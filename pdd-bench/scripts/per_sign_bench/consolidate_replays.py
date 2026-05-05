#!/usr/bin/env python3
"""Consolidate per-scene replay.json files into one jsonl per baseline.

Walks the runs tree (supports both chunked and flat layouts) and concatenates
every replay.json for a given baseline into a single file:
   $RUNS_DIR/var_<i>/<baseline>_replays.jsonl   (one row per scene)

Layouts handled:
  Flat:    runs/var_<i>/<baseline>/replays/<sign_slug>/by_sign/<sign>/by_scene/<scene_uid>/<expert>/replay.json
  Chunked: runs/var_<i>/chunk_<k>/<baseline>/replays/<sign_slug>/by_sign/<sign>/by_scene/<scene_uid>/<expert>/replay.json

Usage:
   # All baselines under one var:
   python3 consolidate_replays.py --runs-var /path/to/runs/var_0

   # Single baseline:
   python3 consolidate_replays.py --runs-var /path/to/runs/var_0 --baseline carl

   # Custom output path:
   python3 consolidate_replays.py --runs-var /path/to/runs/var_0 --baseline carl \\
       --out /tmp/carl_consolidated.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _gather_replay_jsons(runs_var: Path, baseline: str) -> list[Path]:
    """Find all replay.json under both flat and chunked layouts for one baseline."""
    paths: list[Path] = []

    # Chunked: chunk_*/baseline/replays/**/replay.json
    for cd in sorted(runs_var.glob("chunk_*")):
        if not cd.is_dir():
            continue
        bd = cd / baseline
        if bd.is_dir():
            paths.extend(bd.rglob("replay.json"))

    # Flat: baseline/replays/**/replay.json
    bd_flat = runs_var / baseline
    if bd_flat.is_dir():
        paths.extend(bd_flat.rglob("replay.json"))

    # Dedup by absolute path (in case both layouts coexist post-merge).
    return sorted(set(p.resolve() for p in paths))


def _discover_baselines(runs_var: Path) -> list[str]:
    """Discover all baseline names across both layouts."""
    names: set[str] = set()
    # Chunked
    for cd in runs_var.glob("chunk_*"):
        if cd.is_dir():
            for bd in cd.iterdir():
                if bd.is_dir():
                    names.add(bd.name)
    # Flat
    for bd in runs_var.iterdir():
        if bd.is_dir() and not bd.name.startswith("chunk_"):
            names.add(bd.name)
    return sorted(names)


def consolidate(runs_var: Path, baseline: str, out_path: Path,
                dedup: bool = False, prefer_chunked: bool = True) -> tuple[int, int, int]:
    """Write one jsonl line per replay.json.

    Args:
        dedup: if True, write only ONE row per scene_uid (or scene_id) — keeps the
            first encountered. Pair `prefer_chunked` to pick chunked-layout files
            preferentially (they were written by the original run, not re-merged).
        prefer_chunked: when dedup'ing, sort paths so chunked-layout entries come
            first (so they "win" the dedup over flat-layout copies).

    Returns: (n_written, n_failed, n_skipped_dupes)
    """
    replays = _gather_replay_jsons(runs_var, baseline)

    if dedup and prefer_chunked:
        # Put chunked-layout paths first, so they win against flat-layout duplicates.
        replays.sort(key=lambda p: (0 if "/chunk_" in str(p) else 1, str(p)))

    n_written = n_failed = n_dupes = 0
    seen: set[str] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for p in replays:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if dedup:
                    key = str(data.get("scene_uid")
                              or data.get("scene_id")
                              or "_no_uid_")
                    if key in seen:
                        n_dupes += 1
                        continue
                    seen.add(key)
                data["_replay_path"] = str(p)
                out.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
                n_written += 1
            except Exception as e:
                print(f"  [warn] {p}: {e}", file=sys.stderr)
                n_failed += 1
    return n_written, n_failed, n_dupes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-var", required=True,
                   help="Path to runs/var_<i> directory.")
    p.add_argument("--baseline", default=None,
                   help="Single baseline name (e.g. carl). Default: process all baselines.")
    p.add_argument("--out", default=None,
                   help="Output jsonl path. Default: $RUNS_VAR/<baseline>_replays.jsonl")
    p.add_argument("--dedup", action="store_true",
                   help="Keep only ONE row per scene_uid (prefers chunked-layout files).")
    args = p.parse_args()

    runs_var = Path(args.runs_var).resolve()
    if not runs_var.is_dir():
        print(f"ERROR: not a dir: {runs_var}", file=sys.stderr)
        sys.exit(2)

    baselines = [args.baseline] if args.baseline else _discover_baselines(runs_var)
    if not baselines:
        print(f"ERROR: no baselines found under {runs_var}", file=sys.stderr)
        sys.exit(2)

    print(f"[scan] {runs_var}: {len(baselines)} baseline(s)")

    grand_total = 0
    for baseline in baselines:
        if args.out and len(baselines) == 1:
            out_path = Path(args.out).resolve()
        else:
            out_path = runs_var / f"{baseline}_replays.jsonl"

        n_written, n_failed, n_dupes = consolidate(
            runs_var, baseline, out_path, dedup=args.dedup,
        )
        grand_total += n_written
        if n_written == 0:
            print(f"  [skip] {baseline}: no replay.json found")
        else:
            note_parts = []
            if n_failed:
                note_parts.append(f"{n_failed} failed")
            if n_dupes:
                note_parts.append(f"{n_dupes} dupes skipped")
            note = f" ({', '.join(note_parts)})" if note_parts else ""
            print(f"  [ok]   {baseline}: {n_written} scenes → {out_path}{note}")

    print()
    print(f"Total scenes written: {grand_total}")


if __name__ == "__main__":
    main()
