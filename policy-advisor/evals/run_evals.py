"""Golden-set eval runner for the policy-advisor gap analysis.

Golden set format (evals/golden_set.json):

    {
      "comment": "free-text notes",
      "items": [
        {
          "id": "unique-item-id",
          "framework": "app",                     // framework id known to frameworks.py
          "status": "draft" | "reviewed",         // draft = expectations not yet legally reviewed
          "description": "what the case checks",
          "fixture": "fixtures/some_policy.txt",  // path relative to evals/
          "must_contain": ["APP 11: GAP"],        // case-insensitive substrings of the report
          "must_not_contain": ["APP 11: COVERED"],
          "notes": "optional"
        }
      ]
    }

Checks are case-insensitive substring matches with whitespace collapsed, so
line wrapping cannot break them. One live LLM call is made per unique
(fixture, framework) pair; items sharing a fixture share that call, and
--max-calls refuses runs that would exceed the cap.

Usage (from the policy-advisor directory):
  ./venv/bin/python evals/run_evals.py --dry-run           # offline: validate items and fixtures
  ./venv/bin/python evals/run_evals.py --framework app     # live run (needs OPENAI_API_KEY)
  ./venv/bin/python evals/run_evals.py --limit 3           # cap the number of items
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR.parent))

# The agents keep their key in policy-advisor/.env; load it regardless of CWD
# so the key check below sees it.
load_dotenv(EVALS_DIR.parent / ".env")

import frameworks

REQUIRED_ITEM_KEYS = {
    "id", "framework", "status", "description", "fixture", "must_contain", "must_not_contain",
}
VALID_STATUSES = {"draft", "reviewed"}


def load_golden_set(path: Path):
    """Load and validate the golden set; returns the list of items."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path.name}: 'items' must be a non-empty list")

    known_frameworks = frameworks.load_frameworks()
    seen_ids = set()
    for index, item in enumerate(items):
        missing = REQUIRED_ITEM_KEYS - item.keys()
        if missing:
            raise ValueError(f"{path.name}: item #{index} missing keys {sorted(missing)}")
        if item["id"] in seen_ids:
            raise ValueError(f"{path.name}: duplicate item id {item['id']!r}")
        seen_ids.add(item["id"])
        if item["status"] not in VALID_STATUSES:
            raise ValueError(f"{path.name}: item {item['id']!r} has invalid status {item['status']!r}")
        if item["framework"].strip().lower() not in known_frameworks:
            raise ValueError(f"{path.name}: item {item['id']!r} references unknown framework {item['framework']!r}")
        if not (EVALS_DIR / item["fixture"]).is_file():
            raise ValueError(f"{path.name}: item {item['id']!r} fixture not found: {item['fixture']}")
        if not isinstance(item["must_contain"], list) or not item["must_contain"]:
            raise ValueError(f"{path.name}: item {item['id']!r} needs a non-empty must_contain list")
        if not isinstance(item["must_not_contain"], list):
            raise ValueError(f"{path.name}: item {item['id']!r} must_not_contain must be a list")
    return items


def normalise(text: str) -> str:
    """Lower-case and collapse whitespace so line wrapping cannot break checks."""
    return re.sub(r"\s+", " ", text).lower()


def check_item(item: dict, report: str):
    """Return the list of failed checks for one item (empty list = pass)."""
    haystack = normalise(report)
    failures = []
    for needle in item["must_contain"]:
        if normalise(needle) not in haystack:
            failures.append(f"missing required text: {needle!r}")
    for needle in item["must_not_contain"]:
        if normalise(needle) in haystack:
            failures.append(f"contains forbidden text: {needle!r}")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Run golden-set evals for policy-advisor")
    parser.add_argument("--golden", type=Path, default=EVALS_DIR / "golden_set.json",
                        help="path to the golden-set JSON file")
    parser.add_argument("--framework", help="only run items for this framework id")
    parser.add_argument("--limit", type=int, help="cap the number of items to run")
    parser.add_argument("--model", default="gpt-5-nano", help="chat model (default: gpt-5-nano)")
    parser.add_argument("--max-calls", type=int, default=12,
                        help="refuse to make more than this many LLM analysis calls (default: 12)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the golden set and fixtures without calling any API")
    args = parser.parse_args()

    items = load_golden_set(args.golden)
    if args.framework:
        items = [i for i in items if i["framework"].lower() == args.framework.lower()]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print("No golden-set items match the filters.")
        return 0

    draft_count = sum(1 for i in items if i["status"] == "draft")

    if args.dry_run:
        for item in items:
            print(f"OK   {item['id']}  [{item['status']}]  {item['framework']}  {item['fixture']}")
        print(f"\nGolden set valid: {len(items)} item(s), {draft_count} draft.")
        return 0

    # Group items so each (fixture, framework) pair costs one analysis call.
    groups = OrderedDict()
    for item in items:
        groups.setdefault((item["fixture"], item["framework"].lower()), []).append(item)

    if len(groups) > args.max_calls:
        print(f"Refusing to run: {len(groups)} analysis calls needed, --max-calls is {args.max_calls}.")
        return 2

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — live evals need it. Use --dry-run for offline checks.")
        return 2

    from agent import PolicyAdvisor  # deferred import keeps --dry-run fast and offline

    print(f"Running {len(items)} item(s) via {len(groups)} gap-analysis call(s) with {args.model}…\n")
    failed_items = 0
    for (fixture, framework_id), group in groups.items():
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(EVALS_DIR / fixture, tmp)
            advisor = PolicyAdvisor(model_name=args.model, policy_dir=tmp)
            report = advisor.run_gap_analysis(framework_id)

        for item in group:
            failures = check_item(item, report)
            tag = " [draft]" if item["status"] == "draft" else ""
            if failures:
                failed_items += 1
                print(f"FAIL {item['id']}{tag}")
                for failure in failures:
                    print(f"       {failure}")
            else:
                print(f"PASS {item['id']}{tag}")

    print(f"\n{len(items) - failed_items}/{len(items)} passed.")
    if draft_count:
        print(f"Note: {draft_count} draft item(s) — expectations pending legal review; treat scores as provisional.")
    return 1 if failed_items else 0


if __name__ == "__main__":
    sys.exit(main())
