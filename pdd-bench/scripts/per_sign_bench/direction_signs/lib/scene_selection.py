"""Scene review selection (keep/reject) for filter tools and manifest generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sumo_utils import CORE_SCENES_SUBDIR

SELECTION_FILE = "scene_selection.json"
REJECTED_SUBDIR = "_rejected"
CUSTOM_SCENES_SUBDIR = "custom"

VERDICT_PENDING = "pending"
VERDICT_KEEP = "keep"
VERDICT_REJECT = "reject"
VALID_VERDICTS = {VERDICT_PENDING, VERDICT_KEEP, VERDICT_REJECT}

RESERVED_SCENE_DIRS = {CORE_SCENES_SUBDIR, REJECTED_SUBDIR, CUSTOM_SCENES_SUBDIR}


def selection_path(scenes_root: Path) -> Path:
    return scenes_root / SELECTION_FILE


def load_scene_selection(scenes_root: Path) -> dict[str, Any]:
    path = selection_path(scenes_root)
    if not path.is_file():
        return {"updated_at": None, "scenes": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "scenes" not in data:
        data["scenes"] = {}
    return data


def save_scene_selection(scenes_root: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    selection_path(scenes_root).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def scene_verdict(scenes_root: Path, scene_name: str) -> str | None:
    verdicts = load_scene_selection(scenes_root).get("scenes", {})
    verdict = verdicts.get(scene_name)
    if verdict in VALID_VERDICTS:
        return verdict
    return None


def set_scene_verdict(scenes_root: Path, scene_name: str, verdict: str) -> None:
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    selection = load_scene_selection(scenes_root)
    selection.setdefault("scenes", {})[scene_name] = verdict
    save_scene_selection(scenes_root, selection)


def is_reserved_scene_dir(name: str) -> bool:
    return name in RESERVED_SCENE_DIRS


def is_scene_rejected(scenes_root: Path, scene_name: str) -> bool:
    """True if marked reject in scene_selection.json or moved under ``_rejected/``."""
    if scene_verdict(scenes_root, scene_name) == VERDICT_REJECT:
        return True
    # After ``review_junction_scenes.py --apply`` the folder lives only here.
    if (Path(scenes_root) / REJECTED_SUBDIR / scene_name).is_dir():
        return True
    return False


def scene_name_from_manifest_row(row: dict) -> str | None:
    name = row.get("scene_name")
    if name:
        return str(name)
    net_path = row.get("net_path")
    if not net_path:
        return None
    parts = Path(str(net_path)).parts
    return parts[0] if parts else None


def manifest_row_scene_available(
    scenes_root: Path,
    row: dict,
    *,
    respect_scene_selection: bool = True,
) -> bool:
    """False for rejected / missing scene folders (stale manifest rows)."""
    scenes_root = Path(scenes_root)
    scene_name = scene_name_from_manifest_row(row)
    if scene_name and respect_scene_selection and is_scene_rejected(scenes_root, scene_name):
        return False
    net_path = row.get("net_path")
    if not net_path:
        return False
    net_file = Path(str(net_path))
    full = net_file if net_file.is_absolute() else scenes_root / net_file
    return full.is_file()


def rejected_scene_names(scenes_root: Path) -> list[str]:
    scenes_root = Path(scenes_root)
    names = set()
    verdicts = load_scene_selection(scenes_root).get("scenes", {})
    names.update(name for name, verdict in verdicts.items() if verdict == VERDICT_REJECT)
    rejected_dir = scenes_root / REJECTED_SUBDIR
    if rejected_dir.is_dir():
        names.update(
            entry.name
            for entry in rejected_dir.iterdir()
            if entry.is_dir() and (entry / "meta.json").is_file()
        )
    return sorted(names)
