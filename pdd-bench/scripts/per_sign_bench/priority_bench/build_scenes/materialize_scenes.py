#!/usr/bin/env python3
"""Materialize Moscow-allocated junction scenes into a sign pool (e.g. 2.4 yield).

Replaces catalog import + Overpass download. Reads
``moscow_junctions/splits/sign_allocations.json``, ensures each allocated
scene exists under ``moscow_junctions/scenes/{T,X,O}/`` (crops on demand),
then symlinks (or copies) them into ``data/<sign>/scenes/`` for review and
``generate_manifest.py``.

Examples:
    # Yield 2.4 — train+test into data/yield/scenes
    python build_scenes/materialize_scenes.py --sign 2.4

    # Only train half; regenerate previews
    python build_scenes/materialize_scenes.py --sign 2.4 --split train --force-preview
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BUILD_SCENES_DIR = Path(__file__).resolve().parent
PRIORITY_BENCH = BUILD_SCENES_DIR.parent
MOSCOW_ROOT = PRIORITY_BENCH.parent / "moscow_junctions"
MOSCOW_SCRIPTS = MOSCOW_ROOT / "scripts"

sys.path.insert(0, str(PRIORITY_BENCH))
sys.path.insert(0, str(MOSCOW_SCRIPTS))

from signs import get_profile, scenes_dir as profile_scenes_dir  # noqa: E402

PREVIEW_NAME = "custom_cropped.png"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_by_scene_id(index_path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["scene_id"])] = row
    return out


def _moscow_scene_dir(moscow_scenes: Path, shape: str, scene_id: str) -> Path:
    return moscow_scenes / shape / scene_id


def _ensure_cropped(
    row: dict,
    *,
    moscow_scenes: Path,
    moscow_net: Path,
    radius_m: float,
) -> Path:
    """Return path to cropped scene dir; crop from city net if missing."""
    from crop_scenes import crop_o_row, crop_tx_row  # type: ignore

    shape = str(row["shape"])
    scene_id = str(row["scene_id"])
    dest = _moscow_scene_dir(moscow_scenes, shape, scene_id)
    if (dest / "map.net.xml").is_file():
        return dest

    print(f"  [crop] {shape}/{scene_id}")
    if shape in {"T", "X"}:
        crop_tx_row(
            row,
            source_net=moscow_net,
            scenes_root=moscow_scenes,
            radius_m=radius_m,
            skip_existing=True,
        )
    elif shape == "O":
        crop_o_row(
            row,
            source_net=moscow_net,
            scenes_root=moscow_scenes,
            radius_m=max(radius_m, 100.0),
            skip_existing=True,
        )
    else:
        raise ValueError(f"Unknown shape {shape!r} for {scene_id}")
    if not (dest / "map.net.xml").is_file():
        raise FileNotFoundError(f"Crop failed: {dest}")
    return dest


def _render_preview(net_path: Path, out_png: Path) -> None:
    """Top-down PNG for review UI (junction fill + lanes via render_map)."""
    import matplotlib

    matplotlib.use("Agg")
    from tools.render_map import parse_sumo_net, render_network

    edges, junctions = parse_sumo_net(net_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    render_network(edges, junctions, out_png, figsize=(6, 6), dpi=120)


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError(f"Unknown mode {mode!r}")


def materialize(
    *,
    sign: str,
    split: str,
    mode: str,
    dest_scenes: Path,
    allocations_path: Path,
    index_path: Path,
    moscow_scenes: Path,
    moscow_net: Path,
    radius_m: float,
    force_preview: bool,
    crop_missing: bool,
) -> dict:
    alloc_doc = _load_json(allocations_path)
    if sign not in alloc_doc.get("signs", {}):
        raise KeyError(
            f"Sign {sign!r} not in {allocations_path}. "
            f"Known: {sorted((alloc_doc.get('signs') or {}))}"
        )
    block = alloc_doc["signs"][sign]
    index = _index_by_scene_id(index_path)

    halves = ["train", "test"] if split == "all" else [split]
    scene_ids: List[str] = []
    half_of: Dict[str, str] = {}
    for half in halves:
        for sid in block[half]["scene_ids"]:
            scene_ids.append(sid)
            half_of[sid] = half

    dest_scenes.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    records: List[dict] = []

    for sid in scene_ids:
        row = index.get(sid)
        if row is None:
            print(f"  [fail] {sid}: not in junction index")
            fail += 1
            continue
        try:
            if crop_missing:
                src = _ensure_cropped(
                    row,
                    moscow_scenes=moscow_scenes,
                    moscow_net=moscow_net,
                    radius_m=radius_m,
                )
            else:
                src = _moscow_scene_dir(moscow_scenes, row["shape"], sid)
                if not (src / "map.net.xml").is_file():
                    raise FileNotFoundError(
                        f"Missing crop {src} (re-run with --crop-missing)"
                    )

            # Shared moscow crops stay sign-agnostic; sign/split live in moscow_pool.json.
            preview = src / PREVIEW_NAME
            if force_preview or not preview.is_file():
                _render_preview(src / "map.net.xml", preview)

            dst = dest_scenes / sid
            _link_or_copy(src, dst, mode=mode)
            records.append(
                {
                    "scene_id": sid,
                    "shape": row["shape"],
                    "split": half_of[sid],
                    "path": str(dst),
                    "moscow_path": str(src),
                }
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {sid}: {exc}")
            fail += 1

    pool_meta = {
        "sign": sign,
        "split": split,
        "mode": mode,
        "allocations_file": str(allocations_path),
        "n_ok": ok,
        "n_fail": fail,
        "scenes": records,
    }
    (dest_scenes / "moscow_pool.json").write_text(
        json.dumps(pool_meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[materialize] {sign}: ok={ok} fail={fail} → {dest_scenes}")
    return pool_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sign", default="2.4", help="Sign code in sign_allocations.json")
    ap.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="Which allocation half to materialize",
    )
    ap.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Link into sign scenes dir (default) or copy",
    )
    ap.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Destination scenes root (default: data/<profile>/scenes)",
    )
    ap.add_argument(
        "--allocations",
        type=Path,
        default=MOSCOW_ROOT / "splits" / "sign_allocations.json",
    )
    ap.add_argument(
        "--index",
        type=Path,
        default=MOSCOW_ROOT / "index" / "junctions.jsonl",
    )
    ap.add_argument(
        "--moscow-scenes",
        type=Path,
        default=MOSCOW_ROOT / "scenes",
    )
    ap.add_argument(
        "--moscow-net",
        type=Path,
        default=MOSCOW_ROOT / "nets" / "moscow.net.xml",
    )
    ap.add_argument("--radius-m", type=float, default=80.0)
    ap.add_argument(
        "--crop-missing",
        action="store_true",
        default=True,
        help="Crop from moscow.net.xml if scene folder missing (default on)",
    )
    ap.add_argument(
        "--no-crop-missing",
        action="store_false",
        dest="crop_missing",
    )
    ap.add_argument("--force-preview", action="store_true")
    args = ap.parse_args()

    profile = get_profile(args.sign)
    dest = args.scenes_dir
    if dest is None:
        dest = profile_scenes_dir(profile)
    else:
        dest = dest.expanduser().resolve()

    if not args.allocations.is_file():
        sys.exit(f"ERROR: allocations not found: {args.allocations}")
    if not args.index.is_file():
        sys.exit(f"ERROR: index not found: {args.index}")
    if args.crop_missing and not args.moscow_net.is_file():
        sys.exit(f"ERROR: moscow net not found: {args.moscow_net}")

    print(f"[materialize] sign={args.sign} → {dest} (mode={args.mode})")
    materialize(
        sign=str(profile.pdd_code),
        split=args.split,
        mode=args.mode,
        dest_scenes=dest,
        allocations_path=args.allocations,
        index_path=args.index,
        moscow_scenes=args.moscow_scenes,
        moscow_net=args.moscow_net,
        radius_m=args.radius_m,
        force_preview=args.force_preview,
        crop_missing=args.crop_missing,
    )


if __name__ == "__main__":
    main()
