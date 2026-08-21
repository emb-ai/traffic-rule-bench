"""Scene review selection (keep/reject) for filter tools and manifest generation."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CORE_SCENES_SUBDIR = "core"  # keep in sync with traffic_bench.eval.core.sumo.sumo_utils

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
        return {"updated_at": None, "scenes": {}, "reject_reasons": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "scenes" not in data:
        data["scenes"] = {}
    if "reject_reasons" not in data:
        data["reject_reasons"] = {}
    return data


def save_scene_selection(scenes_root: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data.setdefault("scenes", {})
    data.setdefault("reject_reasons", {})
    selection_path(scenes_root).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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
    if verdict != VERDICT_REJECT:
        selection.setdefault("reject_reasons", {}).pop(scene_name, None)
    save_scene_selection(scenes_root, selection)


def set_scene_reject(
    scenes_root: Path,
    scene_name: str,
    *,
    reason: str = "",
    detail: str = "",
) -> None:
    """Mark a scene rejected and optionally record why (for auto-reject / audit)."""
    selection = load_scene_selection(scenes_root)
    selection.setdefault("scenes", {})[scene_name] = VERDICT_REJECT
    reasons = selection.setdefault("reject_reasons", {})
    payload: dict[str, str] = {}
    if reason:
        payload["reason"] = str(reason)
    if detail:
        payload["detail"] = str(detail)
    if payload:
        reasons[scene_name] = payload
    save_scene_selection(scenes_root, selection)


def reject_reason(scenes_root: Path, scene_name: str) -> Optional[dict[str, str]]:
    raw = load_scene_selection(scenes_root).get("reject_reasons", {}).get(scene_name)
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw:
        return {"reason": raw}
    return None


def is_reserved_scene_dir(name: str) -> bool:
    return name in RESERVED_SCENE_DIRS


def is_scene_rejected(scenes_root: Path, scene_name: str) -> bool:
    return scene_verdict(scenes_root, scene_name) == VERDICT_REJECT


def rejected_scene_names(scenes_root: Path) -> list[str]:
    verdicts = load_scene_selection(scenes_root).get("scenes", {})
    return sorted(name for name, verdict in verdicts.items() if verdict == VERDICT_REJECT)


def unapplied_rejected_scenes(scenes_root: Path) -> list[str]:
    """Rejects in scene_selection.json that still exist as top-level scene dirs.

    After review, ``review_scenes.py --apply`` should move these under
    ``_rejected/``. If any remain, manifest generation would still pick them up.
    """
    pending: list[str] = []
    for name in rejected_scene_names(scenes_root):
        if is_reserved_scene_dir(name):
            continue
        path = scenes_root / name
        if path.is_dir():
            pending.append(name)
    return pending


def apply_rejected_scenes(
    scenes_root: Path,
    *,
    dry_run: bool = False,
    only: Optional[list[str]] = None,
) -> tuple[int, int]:
    """Move rejected scene dirs under ``_rejected/``.

    Returns ``(moved, n_rejected)``.
    """
    rejected = rejected_scene_names(scenes_root)
    if only is not None:
        allow = set(only)
        rejected = [name for name in rejected if name in allow]
    if not rejected:
        return 0, 0

    dest_root = scenes_root / REJECTED_SUBDIR
    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    for name in rejected:
        src = scenes_root / name
        dst = dest_root / name
        if not src.is_dir():
            continue
        if dry_run:
            print(f"  would move {name} -> {REJECTED_SUBDIR}/{name}")
            moved += 1
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"  moved {name} -> {REJECTED_SUBDIR}/{name}")
        moved += 1
    return moved, len(rejected)
