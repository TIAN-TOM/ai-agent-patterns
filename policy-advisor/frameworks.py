"""Loader for compliance framework definitions.

Frameworks are plain JSON files in ``frameworks/``. The engine treats them as
config: adding a new standard means adding a data file, not changing agent
code. Each framework lists its principles, and each principle carries the
retrieval queries and expectations the gap-analysis step needs.
"""

import json
from pathlib import Path
from typing import Dict, List

FRAMEWORKS_DIR = Path(__file__).resolve().parent / "frameworks"

_REQUIRED_FRAMEWORK_KEYS = {"id", "name", "source", "description", "principles"}
_REQUIRED_PRINCIPLE_KEYS = {"id", "title", "summary", "queries", "expects"}


def _validate_framework(data: dict, source_name: str) -> None:
    """Raise ValueError if a framework definition is malformed."""
    if not isinstance(data, dict):
        raise ValueError(f"{source_name}: framework definition must be a JSON object")

    missing = _REQUIRED_FRAMEWORK_KEYS - data.keys()
    if missing:
        raise ValueError(f"{source_name}: missing keys {sorted(missing)}")

    principles = data["principles"]
    if not isinstance(principles, list) or not principles:
        raise ValueError(f"{source_name}: 'principles' must be a non-empty list")

    seen_ids = set()
    for index, principle in enumerate(principles):
        if not isinstance(principle, dict):
            raise ValueError(f"{source_name}: principle #{index} must be a JSON object")
        missing = _REQUIRED_PRINCIPLE_KEYS - principle.keys()
        if missing:
            raise ValueError(f"{source_name}: principle #{index} missing keys {sorted(missing)}")
        principle_id = principle["id"]
        if principle_id in seen_ids:
            raise ValueError(f"{source_name}: duplicate principle id {principle_id!r}")
        seen_ids.add(principle_id)
        queries = principle["queries"]
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(q, str) and q.strip() for q in queries)
        ):
            raise ValueError(
                f"{source_name}: principle {principle_id!r} needs a non-empty list of query strings"
            )


def load_frameworks(directory: Path = FRAMEWORKS_DIR) -> Dict[str, dict]:
    """Load and validate every ``*.json`` framework definition in ``directory``."""
    frameworks: Dict[str, dict] = {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name}: invalid JSON ({e})") from e
        _validate_framework(data, path.name)
        framework_id = data["id"].strip().lower()
        if framework_id in frameworks:
            raise ValueError(f"{path.name}: duplicate framework id {framework_id!r}")
        frameworks[framework_id] = data
    return frameworks


def get_framework(framework_id: str, directory: Path = FRAMEWORKS_DIR) -> dict:
    """Return one framework by id (case-insensitive).

    Raises ValueError naming the available ids, so callers can surface a
    helpful message instead of a bare KeyError.
    """
    frameworks = load_frameworks(directory)
    key = framework_id.strip().lower()
    if key not in frameworks:
        available = ", ".join(sorted(frameworks)) or "none loaded"
        raise ValueError(f"Unknown framework {framework_id!r}. Available: {available}")
    return frameworks[key]


def describe_frameworks(directory: Path = FRAMEWORKS_DIR) -> List[str]:
    """One summary line per loaded framework, for menus and agent tools."""
    return [
        f"{fw['id']}: {fw['name']} — {len(fw['principles'])} principles ({fw['source']})"
        for fw in load_frameworks(directory).values()
    ]
