"""Manifest IO, scene collection and resume-key helpers."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bench.util import _slug_to_code, _row_seed, _row_sign_code


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _choose_manifest(code_dir: Path, backend: str) -> Path | None:
    if backend == "pgmap":
        p = code_dir / "pgmap_materialized.jsonl"
        return p if p.exists() and p.stat().st_size > 0 else None
    if backend == "paired":
        p = code_dir / "paired_materialized.jsonl"
        return p if p.exists() and p.stat().st_size > 0 else None
    if backend == "citymap":
        p = code_dir / "citymap_materialized.jsonl"
        return p if p.exists() and p.stat().st_size > 0 else None
    if backend == "sumo":
        p1 = code_dir / "sumo" / "sumo_manifest.jsonl"
        p2 = code_dir / "real_manifest.jsonl"
        if p1.exists() and p1.stat().st_size > 0:
            return p1
        if p2.exists() and p2.stat().st_size > 0:
            return p2
        return None
    return None


def collect_rows(
    benchmark_output_dir: Path,
    backends: list[str],
    only_codes: set[str],
    max_scenes_per_sign: int | None,
    unique_scene_id: bool = False,
) -> list[dict]:
    """Iterate manifests and collect rows for evaluation.

    If `unique_scene_id=True`, deduplicate to ONE row per (backend, scene_id) —
    keeps the first encountered row (typically var_idx=0). Used when caller wants
    to cover unique (map × sign) pairs once, not all seed/var_idx variants.
    """
    rows: list[dict] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    seen_scene_ids: set[tuple[str, str]] = set()

    sign_dirs = sorted([d for d in benchmark_output_dir.iterdir() if d.is_dir() and d.name[:1].isdigit()])
    for sign_dir in sign_dirs:
        sign_code = _slug_to_code(sign_dir.name)
        if only_codes and sign_code not in only_codes:
            continue

        for backend in backends:
            manifest = _choose_manifest(sign_dir, backend)
            if manifest is None:
                continue

            for row in _load_jsonl_rows(manifest):
                if "valid" in row and not row["valid"]:
                    continue
                if unique_scene_id:
                    sid_key = (backend, str(row.get("scene_id") or ""))
                    if sid_key in seen_scene_ids:
                        continue
                    seen_scene_ids.add(sid_key)
                key = (backend, sign_code)
                if max_scenes_per_sign is not None and counts[key] >= max_scenes_per_sign:
                    continue
                row["_backend"] = backend
                row["_sign_code"] = sign_code
                rows.append(row)
                counts[key] += 1

    return rows


def load_manifest_rows(manifest_path: Path, backends: list[str],
                        scene_id: str | None = None,
                        scene_uid: str | None = None) -> list[dict]:
    """Read a manifest, tag rows, and apply optional single-scene filters.

    Per-row `_backend` is preferred; falls back to the single value from
    `backends` when a row lacks it (multi-backend manifests must pre-tag rows).
    `_sign_code` is derived from sign_code/pdd_code/sign_type when absent.
    """
    rows: list[dict] = []
    for row in _load_jsonl_rows(manifest_path):
        if "valid" in row and not row["valid"]:
            continue
        if not row.get("_backend"):
            if len(backends) != 1:
                raise ValueError(
                    "--manifest rows lack `_backend` field — pass --backends "
                    "with exactly one backend, or pre-tag rows.")
            row["_backend"] = backends[0]
        if not row.get("_sign_code"):
            row["_sign_code"] = (row.get("sign_code") or row.get("pdd_code")
                                  or row.get("sign_type") or "")
        rows.append(row)

    if scene_id:
        rows = [r for r in rows if str(r.get("scene_id")) == scene_id]
    if scene_uid:
        # _episode_key_from_row → tuple, joined with ":" matches user input format.
        rows = [r for r in rows
                if ":".join(str(x) for x in _episode_key_from_row(r)) == scene_uid]
    return rows


def select_rows_to_run(rows: list[dict], existing_by_key: dict,
                       rerun_failed: bool = False,
                       skip_error_episodes: bool = False) -> tuple[list[dict], int]:
    """Split rows into (to_run, skipped_count) against already-computed results.

    New rows always run. For rows with an existing record: rerun only failed ones
    when `rerun_failed` (unless `skip_error_episodes` keeps prior errors skipped);
    otherwise skip.
    """
    rows_to_run: list[dict] = []
    skipped = 0
    for row in rows:
        old = existing_by_key.get(_episode_key_from_row(row))
        if old is None:
            rows_to_run.append(row)
            continue
        if skip_error_episodes and not bool(old.get("ok", False)):
            skipped += 1
            continue
        if rerun_failed and not bool(old.get("ok", False)):
            rows_to_run.append(row)
            continue
        skipped += 1
    return rows_to_run, skipped


def _episode_key_from_row(row: dict) -> tuple[str, str, str, int]:
    return (
        str(row.get("_backend", "")),
        str(row.get("scene_id", "")),
        str(_row_sign_code(row) or ""),
        _row_seed(row, -1),
    )


def _episode_key_from_result(r: dict) -> tuple[str, str, str, int]:
    return (
        str(r.get("backend", "")),
        str(r.get("scene_id", "")),
        str(r.get("sign_type", "")),
        int(r.get("seed") or -1),
    )


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows
