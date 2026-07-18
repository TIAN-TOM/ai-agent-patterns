"""Offline tests for the compliance framework loader and shipped definitions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frameworks

# Frameworks every checkout must ship. Extended as new standards are added.
EXPECTED_IDS = {"gdpr", "hipaa", "iso27001", "app"}


def _minimal_framework(**overrides):
    data = {
        "id": "demo",
        "name": "Demo Framework",
        "source": "Demo Act 2000",
        "description": "Test fixture.",
        "principles": [
            {
                "id": "D1",
                "title": "Demo principle",
                "summary": "Something must be done.",
                "queries": ["demo query"],
                "expects": "The document does something.",
            }
        ],
    }
    data.update(overrides)
    return data


class LoadShippedFrameworksTest(unittest.TestCase):
    def test_all_expected_frameworks_load(self):
        loaded = frameworks.load_frameworks()
        self.assertTrue(EXPECTED_IDS.issubset(loaded.keys()), f"loaded: {sorted(loaded)}")

    def test_every_principle_is_complete(self):
        for framework_id, framework in frameworks.load_frameworks().items():
            self.assertTrue(framework["principles"], framework_id)
            for principle in framework["principles"]:
                for key in ("id", "title", "summary", "queries", "expects"):
                    self.assertTrue(
                        str(principle.get(key, "")).strip(),
                        f"{framework_id}: principle {principle.get('id')} missing {key}",
                    )

    def test_get_framework_is_case_insensitive(self):
        self.assertEqual(frameworks.get_framework("GDPR")["id"], "gdpr")

    def test_get_unknown_framework_lists_available_ids(self):
        with self.assertRaises(ValueError) as ctx:
            frameworks.get_framework("does-not-exist")
        message = str(ctx.exception)
        self.assertIn("does-not-exist", message)
        self.assertIn("gdpr", message)

    def test_describe_frameworks_mentions_each_id(self):
        lines = "\n".join(frameworks.describe_frameworks())
        for framework_id in EXPECTED_IDS:
            self.assertIn(framework_id, lines)


class AustralianPrivacyPrinciplesTest(unittest.TestCase):
    """The APP definition must cover all 13 principles of the Privacy Act 1988."""

    def test_all_thirteen_principles_present_in_order(self):
        framework = frameworks.get_framework("app")
        ids = [principle["id"] for principle in framework["principles"]]
        self.assertEqual(ids, [f"APP {n}" for n in range(1, 14)])

    def test_definition_records_review_status_and_scope_notes(self):
        framework = frameworks.get_framework("app")
        self.assertIn("reviewed", framework.get("review_status", "").lower())
        # Scope notes document which statutory details are kept high-level.
        self.assertTrue(framework.get("review_notes"))


class ValidationTest(unittest.TestCase):
    def _load_single(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "framework.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return frameworks.load_frameworks(Path(tmp))

    def test_valid_minimal_framework_loads(self):
        loaded = self._load_single(_minimal_framework())
        self.assertEqual(list(loaded), ["demo"])

    def test_missing_top_level_key_rejected(self):
        data = _minimal_framework()
        del data["source"]
        with self.assertRaisesRegex(ValueError, "source"):
            self._load_single(data)

    def test_empty_principles_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self._load_single(_minimal_framework(principles=[]))

    def test_duplicate_principle_ids_rejected(self):
        principle = _minimal_framework()["principles"][0]
        with self.assertRaisesRegex(ValueError, "duplicate principle id"):
            self._load_single(_minimal_framework(principles=[principle, dict(principle)]))

    def test_principle_without_queries_rejected(self):
        principle = _minimal_framework()["principles"][0]
        principle["queries"] = []
        with self.assertRaisesRegex(ValueError, "query"):
            self._load_single(_minimal_framework(principles=[principle]))

    def test_duplicate_framework_ids_across_files_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.json", "b.json"):
                (Path(tmp) / name).write_text(
                    json.dumps(_minimal_framework()), encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "duplicate framework id"):
                frameworks.load_frameworks(Path(tmp))

    def test_malformed_json_names_the_file_and_does_not_abort_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "valid.json").write_text(
                json.dumps(_minimal_framework()), encoding="utf-8"
            )
            (Path(tmp) / "broken.json").write_text("{ not valid json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "broken.json"):
                frameworks.load_frameworks(Path(tmp))


if __name__ == "__main__":
    unittest.main()
