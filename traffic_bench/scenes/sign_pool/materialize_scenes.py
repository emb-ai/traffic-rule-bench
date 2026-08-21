#!/usr/bin/env python3
"""Materialize Moscow-allocated scenes into a sign pool (e.g. 2.4 yield / 5.7.1).

Reads ``moscow_scenes/splits/sign_allocations.json``, resolves each allocated
scene under either:

* ``scenes/{T,X,O}/`` for ``crop_kind: junction``
* ``scenes/dual_path/{T,X}/{slot}/`` for ``crop_kind: dual_path`` (5.7 / 4.1 / …)
* ``scenes/segment/{straight,curved}/`` for ``crop_kind: segment`` (3.24 / speed)
* ``scenes/segment_detour/{straight,curved}/`` for ``crop_kind: segment_detour`` (4.2.x)

then symlinks (or copies) them into ``data/<sign>/scenes/`` for review and
``generate_manifest.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SIGN_POOL_DIR = Path(__file__).resolve().parent
SCENES_PKG_DIR = SIGN_POOL_DIR.parent
PACKAGE_DIR = SCENES_PKG_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent
MOSCOW_ROOT = SCENES_PKG_DIR / "map_pool"
BUILD_SCENES_DIR = SIGN_POOL_DIR
PRIORITY_BENCH = REPO_ROOT

from traffic_bench.scenes.map_pool.scripts.allocate_sign_scenes import (
    _counts_for_sign,
    _load_yaml,
    _sample,
)
from traffic_bench.scenes.sign_pool.moscow_pool import (
    load_moscow_pool,
    save_moscow_pool,
    scene_split_map,
)
from traffic_bench.scenes.sign_pool.scene_selection import (
    REJECTED_SUBDIR,
    VERDICT_REJECT,
    is_reserved_scene_dir,
    load_scene_selection,
    set_scene_verdict,
)
from traffic_bench.eval.sign_registry import get_profile, scenes_dir as profile_scenes_dir

PREVIEW_NAME = "custom_cropped.png"
DEFAULT_SIGNS_YAML = MOSCOW_ROOT / "splits" / "signs.yaml"
DEFAULT_TRAIN_IDS = MOSCOW_ROOT / "splits" / "train_ids.json"
DEFAULT_TEST_IDS = MOSCOW_ROOT / "splits" / "test_ids.json"


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


def _is_dual_path_scene_id(scene_id: str) -> bool:
    return str(scene_id).startswith("dual_")


def _index_dual_path_scenes(moscow_scenes: Path) -> Dict[str, dict]:
    """Build scene_id → meta(+path) for ``scenes/dual_path/{T,X}/{slot}/…``."""
    root = moscow_scenes / "dual_path"
    out: Dict[str, dict] = {}
    if not root.is_dir():
        return out
    for shape_dir in sorted(root.iterdir()):
        if not shape_dir.is_dir() or shape_dir.name not in {"T", "X"}:
            continue
        for slot_dir in sorted(shape_dir.iterdir()):
            if not slot_dir.is_dir():
                continue
            for scene_dir in sorted(slot_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                meta_path = scene_dir / "meta.json"
                if not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                sid = str(meta.get("scene_id") or scene_dir.name)
                row = dict(meta)
                row["scene_id"] = sid
                row["shape"] = str(meta.get("shape") or shape_dir.name).upper()
                row["slot"] = str(meta.get("slot") or slot_dir.name)
                row["crop_kind"] = "dual_path"
                row["_path"] = str(scene_dir)
                out[sid] = row
    return out


def _index_segment_scenes(moscow_scenes: Path) -> Dict[str, dict]:
    """Build scene_id → meta(+path) for ``scenes/segment/{straight,curved}/…``."""
    root = moscow_scenes / "segment"
    out: Dict[str, dict] = {}
    if not root.is_dir():
        return out
    for type_dir in sorted(root.iterdir()):
        if not type_dir.is_dir() or type_dir.name not in {"straight", "curved"}:
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            meta_path = scene_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sid = str(meta.get("scene_name") or meta.get("scene_id") or scene_dir.name)
            row = dict(meta)
            row["scene_id"] = sid
            row["shape"] = str(meta.get("segment_type") or type_dir.name)
            row["crop_kind"] = "segment"
            row["_path"] = str(scene_dir)
            out[sid] = row
    return out


def _index_segment_detour_scenes(moscow_scenes: Path) -> Dict[str, dict]:
    """Build scene_id → meta(+path) for ``scenes/segment_detour/{straight,curved}/…``."""
    root = moscow_scenes / "segment_detour"
    out: Dict[str, dict] = {}
    if not root.is_dir():
        return out
    for type_dir in sorted(root.iterdir()):
        if not type_dir.is_dir() or type_dir.name not in {"straight", "curved"}:
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            meta_path = scene_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sid = str(meta.get("scene_name") or meta.get("scene_id") or scene_dir.name)
            row = dict(meta)
            row["scene_id"] = sid
            row["shape"] = str(meta.get("segment_type") or type_dir.name)
            row["crop_kind"] = "segment_detour"
            row["_path"] = str(scene_dir)
            out[sid] = row
    return out


def _resolve_scene_row(
    sid: str,
    *,
    junction_index: Dict[str, dict],
    dual_index: Dict[str, dict],
    detour_index: Dict[str, dict],
    segment_index: Dict[str, dict],
) -> dict:
    if sid in detour_index:
        return detour_index[sid]
    if sid in segment_index:
        return segment_index[sid]
    if sid in dual_index:
        return dual_index[sid]
    if sid in junction_index:
        row = dict(junction_index[sid])
        row.setdefault("crop_kind", "junction")
        return row
    if _is_dual_path_scene_id(sid):
        raise KeyError(
            f"dual_path scene not found under scenes/dual_path/: {sid}"
        )
    if str(sid).startswith("seg_") and "detour" in str(sid):
        raise KeyError(
            f"segment_detour scene not found under scenes/segment_detour/: {sid}"
        )
    if str(sid).startswith("seg_"):
        raise KeyError(
            f"segment scene not found under scenes/segment/: {sid}"
        )
    raise KeyError(f"not in junction index: {sid}")


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
    crop_kind = str(row.get("crop_kind") or "junction")
    if crop_kind == "dual_path" or _is_dual_path_scene_id(scene_id):
        # Dual-path crops are produced by crop_dual_path_scenes.py; never
        # re-crop with junction-only radius here.
        if row.get("_path"):
            dest = Path(str(row["_path"]))
        else:
            slot = str(row.get("slot") or "")
            dest = moscow_scenes / "dual_path" / shape / slot / scene_id
        if not (dest / "map.net.xml").is_file():
            raise FileNotFoundError(
                f"Missing dual_path crop {dest} "
                "(run moscow_scenes/scripts/crop_dual_path_scenes.py)"
            )
        return dest
    if crop_kind == "segment":
        if row.get("_path"):
            dest = Path(str(row["_path"]))
        else:
            dest = moscow_scenes / "segment" / shape / scene_id
        if not (dest / "map.net.xml").is_file():
            raise FileNotFoundError(
                f"Missing segment scene {dest} "
                "(run moscow_scenes/scripts/crop_segment_scenes.py)"
            )
        return dest
    if crop_kind == "segment_detour":
        if row.get("_path"):
            dest = Path(str(row["_path"]))
        else:
            dest = moscow_scenes / "segment_detour" / shape / scene_id
        if not (dest / "map.net.xml").is_file():
            raise FileNotFoundError(
                f"Missing segment_detour scene {dest} "
                "(run moscow_scenes/scripts/prepare_segment_detour.py)"
            )
        return dest

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


def _render_preview(net_path: Path, out_png: Path, *, meta: Optional[dict] = None) -> None:
    """Top-down PNG for review UI (junction fill + lanes via render_map).

    For dual_path scenes, overlays short baseline (red) and long compliant (green).
    """
    import matplotlib

    matplotlib.use("Agg")
    from tools.render_map import (
        parse_sumo_net,
        render_network,
        routes_from_dual_path_meta,
    )

    edges, junctions = parse_sumo_net(net_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    baseline = compliant = None
    if meta:
        baseline, compliant, _spawn = routes_from_dual_path_meta(meta)
    render_network(
        edges,
        junctions,
        out_png,
        figsize=(6, 6),
        dpi=120,
        baseline_edge_ids=baseline,
        compliant_edge_ids=compliant,
    )


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


def _materialize_one(
    sid: str,
    *,
    half: str,
    junction_index: Dict[str, dict],
    dual_index: Dict[str, dict],
    detour_index: Dict[str, dict],
    segment_index: Dict[str, dict],
    dest_scenes: Path,
    moscow_scenes: Path,
    moscow_net: Path,
    radius_m: float,
    force_preview: bool,
    crop_missing: bool,
    mode: str,
) -> dict:
    row = _resolve_scene_row(
        sid,
        junction_index=junction_index,
        dual_index=dual_index,
        detour_index=detour_index,
        segment_index=segment_index,
    )
    crop_kind = str(row.get("crop_kind") or "junction")
    if (
        crop_missing
        or crop_kind == "dual_path"
        or crop_kind == "segment"
        or crop_kind == "segment_detour"
        or _is_dual_path_scene_id(sid)
    ):
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
    preview = src / PREVIEW_NAME
    if force_preview or not preview.is_file():
        _render_preview(src / "map.net.xml", preview, meta=row)
    dst = dest_scenes / sid
    _link_or_copy(src, dst, mode=mode)
    return {
        "scene_id": sid,
        "shape": row["shape"],
        "crop_kind": crop_kind,
        "slot": row.get("slot"),
        "split": half,
        "path": str(dst),
        "moscow_path": str(src),
    }


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
    junction_index = _index_by_scene_id(index_path)
    dual_index = _index_dual_path_scenes(moscow_scenes)
    detour_index = _index_segment_detour_scenes(moscow_scenes)
    segment_index = _index_segment_scenes(moscow_scenes)

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
        try:
            rec = _materialize_one(
                sid,
                half=half_of[sid],
                junction_index=junction_index,
                dual_index=dual_index,
                detour_index=detour_index,
                segment_index=segment_index,
                dest_scenes=dest_scenes,
                moscow_scenes=moscow_scenes,
                moscow_net=moscow_net,
                radius_m=radius_m,
                force_preview=force_preview,
                crop_missing=crop_missing,
                mode=mode,
            )
            records.append(rec)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {sid}: {exc}")
            fail += 1

    pool_meta = {
        "sign": sign,
        "split": split,
        "mode": mode,
        "crop_kind": block.get("crop_kind", "junction"),
        "allocations_file": str(allocations_path),
        "n_ok": ok,
        "n_fail": fail,
        "scenes": records,
    }
    save_moscow_pool(dest_scenes, pool_meta)
    print(f"[materialize] {sign}: ok={ok} fail={fail} → {dest_scenes}")
    return pool_meta


def _sign_quotas(signs_cfg: dict, sign: str) -> Tuple[int, int, List[str], Optional[float], int]:
    """Return (n_train, n_test, shapes, x_share, seed)."""
    seed = int(signs_cfg.get("seed", 42))
    n_train = int(signs_cfg.get("n_train", 115))
    test_frac = float(signs_cfg.get("test_frac", 0.2))
    n_test = int(
        signs_cfg.get("n_test")
        or max(1, round(n_train * test_frac / (1.0 - test_frac)))
    )
    spec = (signs_cfg.get("signs") or {}).get(sign) or {}
    shapes = list(spec.get("shapes") or ["T", "X"])
    x_share = spec.get("x_share", spec.get("x_frac"))
    if "n_train" in spec:
        n_train = int(spec["n_train"])
    if "n_test" in spec:
        n_test = int(spec["n_test"])
    return n_train, n_test, shapes, x_share, seed


def _live_scene_ids(dest_scenes: Path) -> Set[str]:
    ids: Set[str] = set()
    if not dest_scenes.is_dir():
        return ids
    for entry in dest_scenes.iterdir():
        if not entry.is_dir() or is_reserved_scene_dir(entry.name):
            continue
        if (entry / "meta.json").is_file() or (entry / "map.net.xml").is_file():
            ids.add(entry.name)
    return ids


def _rejected_history_ids(dest_scenes: Path) -> Set[str]:
    """IDs marked reject or already moved under _rejected/."""
    out: Set[str] = set()
    selection = load_scene_selection(dest_scenes)
    for name, verdict in (selection.get("scenes") or {}).items():
        if verdict == VERDICT_REJECT:
            out.add(str(name))
    rejected_root = dest_scenes / REJECTED_SUBDIR
    if rejected_root.is_dir():
        for entry in rejected_root.iterdir():
            if entry.is_dir():
                out.add(entry.name)
    return out


def _kept_by_split(dest_scenes: Path) -> Dict[str, Set[str]]:
    """Live scenes that are not rejected, grouped by train/test."""
    pool = load_moscow_pool(dest_scenes) or {}
    split_map = scene_split_map(pool)
    rejected = _rejected_history_ids(dest_scenes)
    live = _live_scene_ids(dest_scenes)
    selection = load_scene_selection(dest_scenes).get("scenes") or {}
    kept: Dict[str, Set[str]] = {"train": set(), "test": set()}
    for sid in live:
        if sid in rejected:
            continue
        if selection.get(sid) == VERDICT_REJECT:
            continue
        half = split_map.get(sid)
        if half in kept:
            kept[half].add(sid)
    return kept


def _excluded_ids(dest_scenes: Path, alloc_block: dict) -> Set[str]:
    """Never re-draw these for refill."""
    excluded = set(_rejected_history_ids(dest_scenes))
    excluded |= _live_scene_ids(dest_scenes)
    for half in ("train", "test"):
        for sid in (alloc_block.get(half) or {}).get("scene_ids") or []:
            excluded.add(str(sid))
    # Also any id ever recorded in moscow_pool (covers removed live links).
    pool = load_moscow_pool(dest_scenes) or {}
    for rec in pool.get("scenes") or []:
        if rec.get("scene_id"):
            excluded.add(str(rec["scene_id"]))
    return excluded


def _pick_refill_ids(
    *,
    need: int,
    shapes: List[str],
    x_share: Optional[float],
    by_shape_pool: Dict[str, List[str]],
    excluded: Set[str],
    seed_key: str,
) -> List[str]:
    if need <= 0:
        return []
    need_by_shape = _counts_for_sign(shapes=shapes, n_total=need, x_share=x_share)
    rng = random.Random(seed_key)
    picked: List[str] = []
    for shape, k in need_by_shape.items():
        pool = [s for s in by_shape_pool.get(shape, []) if s not in excluded]
        chosen = _sample(pool, k, rng)
        if len(chosen) < k:
            print(
                f"  [refill] shortfall {shape}: want {k}, available {len(chosen)}"
            )
        picked.extend(chosen)
        excluded.update(chosen)
    return sorted(set(picked))


def refill(
    *,
    sign: str,
    mode: str,
    dest_scenes: Path,
    allocations_path: Path,
    signs_yaml: Path,
    train_ids_path: Path,
    test_ids_path: Path,
    index_path: Path,
    moscow_scenes: Path,
    moscow_net: Path,
    radius_m: float,
    force_preview: bool,
    crop_missing: bool,
) -> dict:
    """Top up kept train/test counts to signs.yaml quotas with fresh scenes."""
    signs_cfg = _load_yaml(signs_yaml)
    n_train, n_test, shapes, x_share, seed = _sign_quotas(signs_cfg, sign)
    targets = {"train": n_train, "test": n_test}

    alloc_doc = _load_json(allocations_path)
    if sign not in alloc_doc.get("signs", {}):
        raise KeyError(f"Sign {sign!r} not in {allocations_path}")
    block = alloc_doc["signs"][sign]

    train_doc = _load_json(train_ids_path)
    test_doc = _load_json(test_ids_path)
    pools = {
        "train": train_doc.get("by_shape") or {},
        "test": test_doc.get("by_shape") or {},
    }

    kept = _kept_by_split(dest_scenes)
    excluded = _excluded_ids(dest_scenes, block)
    junction_index = _index_by_scene_id(index_path)
    dual_index = _index_dual_path_scenes(moscow_scenes)
    detour_index = _index_segment_detour_scenes(moscow_scenes)
    segment_index = _index_segment_scenes(moscow_scenes)

    print(
        f"[refill] targets train={n_train} test={n_test}; "
        f"kept train={len(kept['train'])} test={len(kept['test'])}"
    )

    new_by_half: Dict[str, List[str]] = {"train": [], "test": []}
    for half in ("train", "test"):
        need = max(0, int(targets[half]) - len(kept[half]))
        if need <= 0:
            print(f"  [{half}] already at quota ({len(kept[half])})")
            continue
        print(f"  [{half}] need {need} more")
        picked = _pick_refill_ids(
            need=need,
            shapes=shapes,
            x_share=x_share,
            by_shape_pool=pools[half],
            excluded=excluded,
            seed_key=f"{seed}|{sign}|refill|{half}|{len(excluded)}",
        )
        new_by_half[half] = picked
        print(f"  [{half}] picked {len(picked)}")

    dest_scenes.mkdir(parents=True, exist_ok=True)
    pool = load_moscow_pool(dest_scenes) or {
        "sign": sign,
        "split": "all",
        "mode": mode,
        "allocations_file": str(allocations_path),
        "scenes": [],
    }
    records_by_id = {
        str(r["scene_id"]): r for r in (pool.get("scenes") or []) if r.get("scene_id")
    }

    ok = fail = 0
    added: List[str] = []
    for half, sids in new_by_half.items():
        for sid in sids:
            try:
                rec = _materialize_one(
                    sid,
                    half=half,
                    junction_index=junction_index,
                    dual_index=dual_index,
                    detour_index=detour_index,
                    segment_index=segment_index,
                    dest_scenes=dest_scenes,
                    moscow_scenes=moscow_scenes,
                    moscow_net=moscow_net,
                    radius_m=radius_m,
                    force_preview=force_preview,
                    crop_missing=crop_missing,
                    mode=mode,
                )
                records_by_id[sid] = rec
                set_scene_verdict(dest_scenes, sid, "pending")
                # Append into allocations for this sign.
                half_block = block.setdefault(half, {"scene_ids": [], "by_shape": {}, "n": 0})
                ids = list(half_block.get("scene_ids") or [])
                if sid not in ids:
                    ids.append(sid)
                    half_block["scene_ids"] = sorted(ids)
                    shape = str(rec["shape"])
                    by_shape = dict(half_block.get("by_shape") or {})
                    by_shape[shape] = int(by_shape.get(shape, 0)) + 1
                    half_block["by_shape"] = by_shape
                    half_block["n"] = len(half_block["scene_ids"])
                added.append(sid)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [fail] {sid}: {exc}")
                fail += 1

    pool["sign"] = sign
    pool["mode"] = mode
    pool["allocations_file"] = str(allocations_path)
    pool["scenes"] = sorted(records_by_id.values(), key=lambda r: str(r.get("scene_id")))
    pool["n_ok"] = len(pool["scenes"])
    pool["n_fail"] = fail
    save_moscow_pool(dest_scenes, pool)

    alloc_doc["signs"][sign] = block
    allocations_path.write_text(
        json.dumps(alloc_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    kept_after = _kept_by_split(dest_scenes)
    print(
        f"[refill] added={ok} fail={fail}; "
        f"kept now train={len(kept_after['train'])} test={len(kept_after['test'])} "
        f"→ {dest_scenes}"
    )
    return {
        "added": added,
        "ok": ok,
        "fail": fail,
        "kept": {k: sorted(v) for k, v in kept_after.items()},
        "targets": targets,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sign", default="2.4", help="Sign code in sign_allocations.json")
    ap.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="Which allocation half to materialize (ignored with --refill)",
    )
    ap.add_argument(
        "--refill",
        action="store_true",
        help="Top up kept train/test counts to signs.yaml quotas from unused ids",
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
        help="Destination scenes root (default: data/scenes/<profile>)",
    )
    ap.add_argument(
        "--allocations",
        type=Path,
        default=MOSCOW_ROOT / "splits" / "sign_allocations.json",
    )
    ap.add_argument("--signs-yaml", type=Path, default=DEFAULT_SIGNS_YAML)
    ap.add_argument("--train-ids", type=Path, default=DEFAULT_TRAIN_IDS)
    ap.add_argument("--test-ids", type=Path, default=DEFAULT_TEST_IDS)
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

    if args.refill:
        if not args.signs_yaml.is_file():
            sys.exit(f"ERROR: signs yaml not found: {args.signs_yaml}")
        if not args.train_ids.is_file() or not args.test_ids.is_file():
            sys.exit("ERROR: train_ids.json / test_ids.json required for --refill")
        print(f"[refill] sign={args.sign} → {dest} (mode={args.mode})")
        refill(
            sign=str(profile.pdd_code),
            mode=args.mode,
            dest_scenes=dest,
            allocations_path=args.allocations,
            signs_yaml=args.signs_yaml,
            train_ids_path=args.train_ids,
            test_ids_path=args.test_ids,
            index_path=args.index,
            moscow_scenes=args.moscow_scenes,
            moscow_net=args.moscow_net,
            radius_m=args.radius_m,
            force_preview=args.force_preview,
            crop_missing=args.crop_missing,
        )
        return

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
