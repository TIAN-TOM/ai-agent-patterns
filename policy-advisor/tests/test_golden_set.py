"""Offline tests for the golden-set format and the eval checker logic."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

POLICY_ADVISOR_DIR = Path(__file__).resolve().parents[1]
EVALS_DIR = POLICY_ADVISOR_DIR / "evals"
sys.path.insert(0, str(POLICY_ADVISOR_DIR))
sys.path.insert(0, str(EVALS_DIR))

import run_evals


class GoldenSetFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = run_evals.load_golden_set(EVALS_DIR / "golden_set.json")

    def _app_items(self):
        return [i for i in self.items if i["framework"] == "app"]

    def test_app_has_eight_to_twelve_items(self):
        count = len(self._app_items())
        self.assertGreaterEqual(count, 8)
        self.assertLessEqual(count, 12)

    def test_all_app_items_are_draft_pending_legal_review(self):
        for item in self._app_items():
            self.assertEqual(item["status"], "draft", item["id"])

    def test_app_items_cover_compliant_and_non_compliant_cases(self):
        app_items = self._app_items()
        compliant = [
            i for i in app_items
            if any("covered" in needle.lower() for needle in i["must_contain"])
        ]
        non_compliant = [
            i for i in app_items
            if any("gap" in needle.lower() for needle in i["must_contain"])
            or i["must_not_contain"]
        ]
        self.assertTrue(compliant, "expected at least one compliant-side case")
        self.assertTrue(non_compliant, "expected at least one non-compliant-side case")


class GoldenSetValidationTest(unittest.TestCase):
    def _load(self, items):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.json"
            path.write_text(json.dumps({"items": items}), encoding="utf-8")
            return run_evals.load_golden_set(path)

    def _valid_item(self, **overrides):
        item = {
            "id": "demo-case",
            "framework": "app",
            "status": "draft",
            "description": "demo",
            "fixture": "fixtures/au_sparse_policy.txt",
            "must_contain": ["APP 11:"],
            "must_not_contain": [],
        }
        item.update(overrides)
        return item

    def test_valid_item_loads(self):
        self.assertEqual(len(self._load([self._valid_item()])), 1)

    def test_unknown_framework_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown framework"):
            self._load([self._valid_item(framework="nope")])

    def test_missing_fixture_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixture not found"):
            self._load([self._valid_item(fixture="fixtures/missing.txt")])

    def test_duplicate_item_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate item id"):
            self._load([self._valid_item(), self._valid_item()])

    def test_invalid_status_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid status"):
            self._load([self._valid_item(status="maybe")])

    def test_empty_must_contain_rejected(self):
        with self.assertRaisesRegex(ValueError, "must_contain"):
            self._load([self._valid_item(must_contain=[])])


class CheckItemTest(unittest.TestCase):
    def _item(self, contain, not_contain=()):
        return {"must_contain": list(contain), "must_not_contain": list(not_contain)}

    def test_passes_when_required_text_present(self):
        failures = run_evals.check_item(
            self._item(["APP 11: GAP"]), "APP 11: GAP — nothing found"
        )
        self.assertEqual(failures, [])

    def test_check_is_case_insensitive_and_whitespace_tolerant(self):
        report = "app 11:\n    gap — nothing found"
        failures = run_evals.check_item(self._item(["APP 11: GAP"]), report)
        self.assertEqual(failures, [])

    def test_fails_on_missing_required_text(self):
        failures = run_evals.check_item(self._item(["APP 12: COVERED"]), "APP 12: GAP")
        self.assertEqual(len(failures), 1)
        self.assertIn("missing required text", failures[0])

    def test_fails_on_forbidden_text(self):
        failures = run_evals.check_item(
            self._item(["APP 7:"], ["APP 7: COVERED"]), "APP 7: COVERED — fine"
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("forbidden text", failures[0])


if __name__ == "__main__":
    unittest.main()
