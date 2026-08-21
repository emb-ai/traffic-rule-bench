#!/usr/bin/env python3
"""Allocate n_train / n_test maps per sign from the shared global train/test split.

Shared pool: signs sample independently from the same train/test junction set
(a junction may be assigned to several signs). Within one sign, scene_ids are
unique.

``crop_kind`` in signs.yaml:
  * ``junction`` (default) — sample ``junc_*`` / ``rb_*`` ids from train/test json
  * ``dual_path`` — sample from ``scenes/dual_path/{shape}/{slot}/`` whose
    ``junction_id`` is on the train/test side (via index), filtered by
    ``lib.roles.sign_to_slots`` (+ stem / carriageway for 5.7).
    Within each shape quota, scene_ids are drawn **evenly across slots**
    (e.g. 3.18.1 balances ``r_s`` vs ``r_l``; 3.1 balances all six).
  * ``lane_direction`` — sample from ``scenes/lane_direction/{shape}/``
    (5.15.1 multi-lane LC atoms); within each shape quota, balance by
    ``compliant_dir`` (l / r).

Canonical quotas: ``splits/signs.yaml``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]

from traffic_bench.scenes.map_pool.lib.roles import scenario_matches_sign, sign_shape_policy, sign_to_slots

DEFAULT_SIGNS_YAML = ROOT / "splits" / "signs.yaml"
DEFAULT_TRAIN = ROOT / "splits" / "train_ids.json"
DEFAULT_TEST = ROOT / "splits" / "test_ids.json"
DEFAULT_INDEX = ROOT / "index" / "junctions.jsonl"
DEFAULT_SEGMENTS_INDEX = ROOT / "index" / "segments.jsonl"
DEFAULT_DUAL_ROOT = ROOT / "scenes" / "dual_path"
DEFAULT_LANE_DIR_ROOT = ROOT / "scenes" / "lane_direction"
DEFAULT_SEGMENT_ROOT = ROOT / "scenes" / "segment"
DEFAULT_SEGMENT_DETOUR_ROOT = ROOT / "scenes" / "segment_detour"
DEFAULT_OUT = ROOT / "splits" / "sign_allocations.json"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"PyYAML is required to read {path}. Install with: pip install pyyaml"
        ) from exc
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _counts_for_sign(
    *,
    shapes: List[str],
    n_total: int,
    x_share: Optional[float],
) -> Dict[str, int]:
    shapes = [s.upper() for s in shapes]
    if shapes == ["O"]:
        return {"O": n_total}
    if shapes == ["T"]:
        return {"T": n_total}
    if set(shapes) == {"T", "X"}:
        share = 0.5 if x_share is None else float(x_share)
        n_x = int(round(n_total * share))
        n_x = min(max(n_x, 0), n_total)
        return {"X": n_x, "T": n_total - n_x}
    base, rem = divmod(n_total, len(shapes))
    out = {s: base for s in shapes}
    for s in shapes[:rem]:
        out[s] += 1
    return out


def _sample(pool: List[str], k: int, rng: random.Random) -> List[str]:
    if k <= 0:
        return []
    if k > len(pool):
        return sorted(pool)
    return sorted(rng.sample(pool, k))


def _slot_quotas(slots: List[str], need: int) -> Dict[str, int]:
    """Even split of ``need`` across slots (remainder to first slots)."""
    slots = [s for s in slots if s]
    if not slots or need <= 0:
        return {}
    base, rem = divmod(int(need), len(slots))
    return {s: base + (1 if i < rem else 0) for i, s in enumerate(slots)}


def _sample_balanced_across_slots(
    by_slot: Dict[str, List[str]],
    need: int,
    rng: random.Random,
) -> tuple[List[str], Dict[str, int]]:
    """Sample ``need`` ids with near-equal counts per slot; fill shortfalls from leftovers."""
    slots = sorted(by_slot.keys())
    if need <= 0 or not slots:
        return [], {s: 0 for s in slots}

    quotas = _slot_quotas(slots, need)
    picked: List[str] = []
    got: Dict[str, int] = {s: 0 for s in slots}
    picked_set: Set[str] = set()

    for slot in slots:
        q = int(quotas.get(slot, 0))
        pool = list(by_slot.get(slot) or [])
        chunk = _sample(pool, q, rng)
        for sid in chunk:
            if sid not in picked_set:
                picked.append(sid)
                picked_set.add(sid)
                got[slot] = got.get(slot, 0) + 1

    shortfall = need - len(picked)
    if shortfall > 0:
        leftovers: List[str] = []
        for slot in slots:
            for sid in by_slot.get(slot) or []:
                if sid not in picked_set:
                    leftovers.append(sid)
        extra = _sample(leftovers, shortfall, rng)
        for sid in extra:
            if sid in picked_set:
                continue
            picked.append(sid)
            picked_set.add(sid)
            # Attribute fill to the slot that owns this id when possible.
            for slot in slots:
                if sid in (by_slot.get(slot) or []):
                    got[slot] = got.get(slot, 0) + 1
                    break

    return sorted(picked), got


def _load_index_maps(index_path: Path) -> Dict[str, str]:
    """Return scene_id → junction_id."""
    scene_to_junc: Dict[str, str] = {}
    if not index_path.is_file():
        return scene_to_junc
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("scene_id") or "")
            jid = str(row.get("junction_id") or "")
            if sid and jid:
                scene_to_junc[sid] = jid
    return scene_to_junc


def _junction_ids_for_split(
    scene_ids: List[str],
    scene_to_junc: Dict[str, str],
) -> Set[str]:
    out: Set[str] = set()
    for sid in scene_ids:
        jid = scene_to_junc.get(sid)
        if jid:
            out.add(jid)
        elif sid.startswith("junc_"):
            out.add(sid[len("junc_") :])
    return out


def _scan_dual_path_pool(
    dual_root: Path,
    *,
    allowed_shapes: Set[str],
    allowed_slots: Set[str],
    allowed_junctions: Set[str],
    pdd_code: str,
) -> Dict[str, Dict[str, List[str]]]:
    """Return ``by_shape[shape][slot] = [scene_id, ...]`` for dual_path crops."""
    by_shape: Dict[str, Dict[str, List[str]]] = {s: {} for s in allowed_shapes}
    if not dual_root.is_dir():
        return by_shape
    for shape_dir in sorted(dual_root.iterdir()):
        if not shape_dir.is_dir():
            continue
        shape = shape_dir.name.upper()
        if shape not in allowed_shapes:
            continue
        for slot_dir in sorted(shape_dir.iterdir()):
            if not slot_dir.is_dir() or slot_dir.name not in allowed_slots:
                continue
            slot = slot_dir.name
            for scene_dir in sorted(slot_dir.iterdir()):
                meta_path = scene_dir / "meta.json"
                if not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                jid = str(meta.get("junction_id") or "")
                if allowed_junctions and jid not in allowed_junctions:
                    continue
                if not scenario_matches_sign(meta, pdd_code):
                    continue
                sid = str(meta.get("scene_id") or scene_dir.name)
                by_shape.setdefault(shape, {}).setdefault(slot, []).append(sid)
    for shape, by_slot in list(by_shape.items()):
        for slot in list(by_slot):
            by_slot[slot] = sorted(set(by_slot[slot]))
    return by_shape


def _scan_lane_direction_pool(
    lane_root: Path,
    *,
    allowed_shapes: Set[str],
    allowed_junctions: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """Return ``by_shape[shape][compliant_dir] = [scene_id, ...]``."""
    by_shape: Dict[str, Dict[str, List[str]]] = {s: {} for s in allowed_shapes}
    if not lane_root.is_dir():
        return by_shape
    for shape_dir in sorted(lane_root.iterdir()):
        if not shape_dir.is_dir():
            continue
        shape = shape_dir.name.upper()
        if shape not in allowed_shapes:
            continue
        for scene_dir in sorted(shape_dir.iterdir()):
            meta_path = scene_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            jid = str(meta.get("junction_id") or "")
            if allowed_junctions and jid not in allowed_junctions:
                continue
            dp = meta.get("dual_path") or {}
            if str(dp.get("kind") or "") not in ("lane_change", ""):
                # Accept empty kind for older metas that still have spawn/target.
                if meta.get("target_lane_num") is None and dp.get("target_lane_num") is None:
                    continue
            cdir = str(
                meta.get("compliant_dir")
                or dp.get("compliant_dir")
                or "x"
            ).strip().lower()
            if cdir not in ("l", "r"):
                cdir = "x"
            sid = str(meta.get("scene_id") or scene_dir.name)
            by_shape.setdefault(shape, {}).setdefault(cdir, []).append(sid)
    for shape, by_dir in list(by_shape.items()):
        for d in list(by_dir):
            by_dir[d] = sorted(set(by_dir[d]))
    return by_shape


def _load_segments_index(path: Path) -> Dict[str, Dict]:
    """Load segments.jsonl as {scene_id: row}."""
    index: Dict[str, Dict] = {}
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("scene_id", ""))
            if sid:
                index[sid] = row
    return index


def _scan_segment_pool(
    segment_root: Path,
    segments_index: Dict[str, Dict],
    *,
    allowed_segment_types: Set[str],
    allowed_osm_ways: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """Return ``by_type[segment_type][segment_type] = [scene_id, ...]``.

    For segments, we don't have shape (T/X), so we use segment_type as the key.
    Split is by osm_way_id (not junction_id) to prevent data leakage.
    """
    by_type: Dict[str, Dict[str, List[str]]] = {t: {} for t in allowed_segment_types}
    if not segment_root.is_dir():
        return by_type
    for type_dir in sorted(segment_root.iterdir()):
        if not type_dir.is_dir():
            continue
        seg_type = type_dir.name
        if seg_type not in allowed_segment_types:
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            meta_path = scene_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            osm_way = str(meta.get("osm_way_id") or "")
            if allowed_osm_ways and osm_way not in allowed_osm_ways:
                continue
            sid = str(meta.get("scene_id") or scene_dir.name)
            by_type.setdefault(seg_type, {}).setdefault(seg_type, []).append(sid)
    for seg_type, by_sub in list(by_type.items()):
        for sub in list(by_sub):
            by_sub[sub] = sorted(set(by_sub[sub]))
    return by_type


def _scan_segment_detour_pool(
    segment_detour_root: Path,
    pdd_code: str,
    *,
    allowed_segment_types: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """Return ``by_type[segment_type][segment_type] = [scene_id, ...]`` for segment_detour.

    Scans scenes/segment_detour/{straight,curved}/ for scenes matching the PDD code.
    Uses osm_way_id for train/test split.
    """
    by_type: Dict[str, Dict[str, List[str]]] = {t: {} for t in allowed_segment_types}
    if not segment_detour_root.is_dir():
        return by_type
    for type_dir in sorted(segment_detour_root.iterdir()):
        if not type_dir.is_dir():
            continue
        seg_type = type_dir.name
        if seg_type not in allowed_segment_types:
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            meta_path = scene_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # Filter by pdd_code
            scene_pdd = str(meta.get("pdd_code") or meta.get("detour_code") or "")
            if scene_pdd != pdd_code:
                continue
            osm_way = str(meta.get("osm_way_id") or "")
            sid = str(meta.get("scene_name") or scene_dir.name)
            by_type.setdefault(seg_type, {}).setdefault(seg_type, []).append(sid)
    for seg_type, by_sub in list(by_type.items()):
        for sub in list(by_sub):
            by_sub[sub] = sorted(set(by_sub[sub]))
    return by_type


def _osm_ways_from_segment_detour(
    segment_detour_root: Path,
    pdd_code: str,
    allowed_segment_types: Set[str],
) -> Dict[str, str]:
    """Return scene_id -> osm_way_id mapping for segment_detour scenes."""
    mapping: Dict[str, str] = {}
    if not segment_detour_root.is_dir():
        return mapping
    for type_dir in sorted(segment_detour_root.iterdir()):
        if not type_dir.is_dir():
            continue
        seg_type = type_dir.name
        if seg_type not in allowed_segment_types:
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            meta_path = scene_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scene_pdd = str(meta.get("pdd_code") or meta.get("detour_code") or "")
            if scene_pdd != pdd_code:
                continue
            sid = str(meta.get("scene_name") or scene_dir.name)
            osm_way = str(meta.get("osm_way_id") or "")
            if osm_way:
                mapping[sid] = osm_way
    return mapping


def _osm_ways_for_split(
    segments_index: Dict[str, Dict],
    scene_ids: List[str],
) -> Set[str]:
    """Get osm_way_ids for given scene_ids."""
    ways: Set[str] = set()
    for sid in scene_ids:
        row = segments_index.get(sid)
        if row:
            way = str(row.get("osm_way_id", ""))
            if way:
                ways.add(way)
    return ways


def allocate(
    *,
    signs_cfg: dict,
    train_by_shape: Dict[str, List[str]],
    test_by_shape: Dict[str, List[str]],
    scene_to_junc: Dict[str, str],
    dual_root: Path,
    lane_dir_root: Path,
    segment_root: Path,
    segment_detour_root: Path,
    segments_index: Dict[str, Dict],
) -> dict:
    seed = int(signs_cfg.get("seed", 42))
    n_train = int(signs_cfg.get("n_train", 115))
    test_frac = float(signs_cfg.get("test_frac", 0.2))
    default_n_test = int(
        signs_cfg.get("n_test")
        or max(1, round(n_train * test_frac / (1.0 - test_frac)))
    )
    signs = signs_cfg.get("signs") or {}

    train_juncs = _junction_ids_for_split(
        [sid for ids in train_by_shape.values() for sid in ids], scene_to_junc
    )
    test_juncs = _junction_ids_for_split(
        [sid for ids in test_by_shape.values() for sid in ids], scene_to_junc
    )

    allocations: Dict[str, Any] = {}
    shortfalls: List[str] = []

    for sign_code, spec in sorted(signs.items(), key=lambda kv: str(kv[0])):
        spec = spec or {}
        crop_kind = str(spec.get("crop_kind") or "junction").strip().lower()
        shapes = [str(s).upper() for s in (spec.get("shapes") or ["T", "X"])]
        x_share = spec.get("x_share", spec.get("x_frac"))
        sign_n_train = int(spec.get("n_train", n_train))
        sign_n_test = int(spec.get("n_test", default_n_test))

        train_need = _counts_for_sign(
            shapes=shapes, n_total=sign_n_train, x_share=x_share
        )
        test_need = _counts_for_sign(
            shapes=shapes, n_total=sign_n_test, x_share=x_share
        )

        rng_train = random.Random(f"{seed}|{sign_code}|train")
        rng_test = random.Random(f"{seed}|{sign_code}|test")

        if crop_kind == "dual_path":
            try:
                slots = set(sign_to_slots(str(sign_code)))
                shape_policy = set(sign_shape_policy(str(sign_code)))
            except ValueError as exc:
                shortfalls.append(f"{sign_code}: {exc}")
                continue
            shapes = [s for s in shapes if s in shape_policy] or sorted(shape_policy)
            train_pool = _scan_dual_path_pool(
                dual_root,
                allowed_shapes=set(shapes),
                allowed_slots=slots,
                allowed_junctions=train_juncs,
                pdd_code=str(sign_code),
            )
            test_pool = _scan_dual_path_pool(
                dual_root,
                allowed_shapes=set(shapes),
                allowed_slots=slots,
                allowed_junctions=test_juncs,
                pdd_code=str(sign_code),
            )
            train_need = _counts_for_sign(
                shapes=shapes, n_total=sign_n_train, x_share=x_share
            )
            test_need = _counts_for_sign(
                shapes=shapes, n_total=sign_n_test, x_share=x_share
            )
        elif crop_kind == "lane_direction":
            train_pool = _scan_lane_direction_pool(
                lane_dir_root,
                allowed_shapes=set(shapes),
                allowed_junctions=train_juncs,
            )
            test_pool = _scan_lane_direction_pool(
                lane_dir_root,
                allowed_shapes=set(shapes),
                allowed_junctions=test_juncs,
            )
            train_need = _counts_for_sign(
                shapes=shapes, n_total=sign_n_train, x_share=x_share
            )
            test_need = _counts_for_sign(
                shapes=shapes, n_total=sign_n_test, x_share=x_share
            )
        elif crop_kind == "segment":
            # Segment scenes: split by osm_way_id, not junction_id
            segment_types = set(spec.get("segment_types") or ["straight", "curved"])
            # Build osm_way_id sets from train/test junction splits
            # Use segments whose osm_way_id appears only in train/test junctions
            train_ways = _osm_ways_for_split(segments_index, list(segments_index.keys()))
            test_ways: Set[str] = set()
            # Simple hash-based split by osm_way_id
            for sid, row in segments_index.items():
                osm_way = str(row.get("osm_way_id", ""))
                if osm_way:
                    # 20% test split via hash
                    if hash(osm_way) % 5 == 0:
                        test_ways.add(osm_way)
            train_ways = train_ways - test_ways
            train_pool = _scan_segment_pool(
                segment_root,
                segments_index,
                allowed_segment_types=segment_types,
                allowed_osm_ways=train_ways,
            )
            test_pool = _scan_segment_pool(
                segment_root,
                segments_index,
                allowed_segment_types=segment_types,
                allowed_osm_ways=test_ways,
            )
            # For segments, use segment_type as the "shape" key
            train_need = {t: sign_n_train // len(segment_types) for t in segment_types}
            test_need = {t: sign_n_test // len(segment_types) for t in segment_types}
            # Distribute remainder
            for i, t in enumerate(sorted(segment_types)):
                if i < sign_n_train % len(segment_types):
                    train_need[t] += 1
                if i < sign_n_test % len(segment_types):
                    test_need[t] += 1
        elif crop_kind == "segment_detour":
            # Segment detour scenes: split by osm_way_id, filter by pdd_code
            segment_types = set(spec.get("segment_types") or ["straight", "curved"])
            # Get osm_way mapping for this sign's scenes
            scene_to_way = _osm_ways_from_segment_detour(
                segment_detour_root,
                pdd_code=str(sign_code),
                allowed_segment_types=segment_types,
            )
            # Build train/test osm_way sets (20% test via hash)
            all_ways = set(scene_to_way.values())
            test_ways: Set[str] = set()
            for osm_way in all_ways:
                if osm_way and hash(osm_way) % 5 == 0:
                    test_ways.add(osm_way)
            train_ways = all_ways - test_ways
            # Scan pools filtering by osm_way
            full_pool = _scan_segment_detour_pool(
                segment_detour_root,
                pdd_code=str(sign_code),
                allowed_segment_types=segment_types,
            )
            # Split by osm_way
            train_pool: Dict[str, Dict[str, List[str]]] = {t: {} for t in segment_types}
            test_pool: Dict[str, Dict[str, List[str]]] = {t: {} for t in segment_types}
            for seg_type in segment_types:
                by_sub = full_pool.get(seg_type) or {}
                for sub, sids in by_sub.items():
                    for sid in sids:
                        osm_way = scene_to_way.get(sid, "")
                        if osm_way in train_ways:
                            train_pool.setdefault(seg_type, {}).setdefault(sub, []).append(sid)
                        elif osm_way in test_ways:
                            test_pool.setdefault(seg_type, {}).setdefault(sub, []).append(sid)
            # For segments, use segment_type as the "shape" key
            train_need = {t: sign_n_train // len(segment_types) for t in segment_types}
            test_need = {t: sign_n_test // len(segment_types) for t in segment_types}
            for i, t in enumerate(sorted(segment_types)):
                if i < sign_n_train % len(segment_types):
                    train_need[t] += 1
                if i < sign_n_test % len(segment_types):
                    test_need[t] += 1
        else:
            train_pool = {s: list(train_by_shape.get(s, [])) for s in shapes}
            test_pool = {s: list(test_by_shape.get(s, [])) for s in shapes}

        train_ids: List[str] = []
        test_ids: List[str] = []
        got_train: Dict[str, int] = {}
        got_test: Dict[str, int] = {}
        got_train_slots: Dict[str, Dict[str, int]] = {}
        got_test_slots: Dict[str, Dict[str, int]] = {}

        if crop_kind in ("dual_path", "lane_direction", "segment", "segment_detour"):
            for shape, need in train_need.items():
                by_slot = dict(train_pool.get(shape) or {})
                picked, slot_got = _sample_balanced_across_slots(
                    by_slot, need, rng_train
                )
                got_train[shape] = len(picked)
                got_train_slots[shape] = slot_got
                if len(picked) < need:
                    shortfalls.append(
                        f"{sign_code} train {shape}: want {need}, got {len(picked)}"
                    )
                train_ids.extend(picked)

            for shape, need in test_need.items():
                by_slot = dict(test_pool.get(shape) or {})
                picked, slot_got = _sample_balanced_across_slots(
                    by_slot, need, rng_test
                )
                got_test[shape] = len(picked)
                got_test_slots[shape] = slot_got
                if len(picked) < need:
                    shortfalls.append(
                        f"{sign_code} test {shape}: want {need}, got {len(picked)}"
                    )
                test_ids.extend(picked)
        else:
            for shape, need in train_need.items():
                pool = list(train_pool.get(shape, []))
                picked = _sample(pool, need, rng_train)
                got_train[shape] = len(picked)
                if len(picked) < need:
                    shortfalls.append(
                        f"{sign_code} train {shape}: want {need}, got {len(picked)}"
                    )
                train_ids.extend(picked)

            for shape, need in test_need.items():
                pool = list(test_pool.get(shape, []))
                picked = _sample(pool, need, rng_test)
                got_test[shape] = len(picked)
                if len(picked) < need:
                    shortfalls.append(
                        f"{sign_code} test {shape}: want {need}, got {len(picked)}"
                    )
                test_ids.extend(picked)

        block: Dict[str, Any] = {
            "crop_kind": crop_kind,
            "shapes": shapes,
            "x_share": x_share,
            "train": {
                "scene_ids": sorted(set(train_ids)),
                "by_shape": got_train,
                "n": len(set(train_ids)),
            },
            "test": {
                "scene_ids": sorted(set(test_ids)),
                "by_shape": got_test,
                "n": len(set(test_ids)),
            },
        }
        if crop_kind == "dual_path":
            block["train"]["by_slot"] = got_train_slots
            block["test"]["by_slot"] = got_test_slots
            block["slots"] = sorted(slots)
        elif crop_kind == "lane_direction":
            block["train"]["by_compliant_dir"] = got_train_slots
            block["test"]["by_compliant_dir"] = got_test_slots
        elif crop_kind == "segment":
            block["train"]["by_segment_type"] = got_train_slots
            block["test"]["by_segment_type"] = got_test_slots
            block["segment_types"] = sorted(segment_types)
            block["shapes"] = []  # segments don't have shapes
        elif crop_kind == "segment_detour":
            block["train"]["by_segment_type"] = got_train_slots
            block["test"]["by_segment_type"] = got_test_slots
            block["segment_types"] = sorted(segment_types)
            block["shapes"] = []  # segment_detour doesn't have shapes
        allocations[str(sign_code)] = block

    return {
        "seed": seed,
        "n_train_target": n_train,
        "n_test_target": default_n_test,
        "test_frac": test_frac,
        "shared_pool": True,
        "shortfalls": shortfalls,
        "signs": allocations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signs-yaml", type=Path, default=DEFAULT_SIGNS_YAML)
    ap.add_argument("--train-ids", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--test-ids", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--segments-index", type=Path, default=DEFAULT_SEGMENTS_INDEX)
    ap.add_argument("--dual-root", type=Path, default=DEFAULT_DUAL_ROOT)
    ap.add_argument("--lane-dir-root", type=Path, default=DEFAULT_LANE_DIR_ROOT)
    ap.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    ap.add_argument("--segment-detour-root", type=Path, default=DEFAULT_SEGMENT_DETOUR_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in (args.signs_yaml, args.train_ids, args.test_ids):
        if not p.is_file():
            sys.exit(f"ERROR: missing {p}")

    try:
        signs_cfg = _load_yaml(args.signs_yaml)
    except ImportError as exc:
        sys.exit(f"ERROR: {exc}")

    train_doc = json.loads(args.train_ids.read_text(encoding="utf-8"))
    test_doc = json.loads(args.test_ids.read_text(encoding="utf-8"))
    scene_to_junc = _load_index_maps(args.index)
    segments_index = _load_segments_index(args.segments_index)
    result = allocate(
        signs_cfg=signs_cfg,
        train_by_shape=train_doc.get("by_shape") or {},
        test_by_shape=test_doc.get("by_shape") or {},
        scene_to_junc=scene_to_junc,
        dual_root=args.dual_root,
        lane_dir_root=args.lane_dir_root,
        segment_root=args.segment_root,
        segment_detour_root=args.segment_detour_root,
        segments_index=segments_index,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    twin = args.signs_yaml.with_suffix(".json")
    twin.write_text(
        json.dumps(signs_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[allocate] signs={len(result['signs'])} → {args.out}")
    print(f"[allocate] wrote twin config → {twin}")
    for code, block in result["signs"].items():
        slot_note = ""
        if block.get("crop_kind") == "dual_path":
            tr = block["train"].get("by_slot") or {}
            te = block["test"].get("by_slot") or {}
            slot_note = f"  slots_train={tr}  slots_test={te}"
        elif block.get("crop_kind") == "lane_direction":
            tr = block["train"].get("by_compliant_dir") or {}
            te = block["test"].get("by_compliant_dir") or {}
            slot_note = f"  dirs_train={tr}  dirs_test={te}"
        elif block.get("crop_kind") == "segment":
            tr = block["train"].get("by_segment_type") or {}
            te = block["test"].get("by_segment_type") or {}
            slot_note = f"  types_train={tr}  types_test={te}"
        elif block.get("crop_kind") == "segment_detour":
            tr = block["train"].get("by_segment_type") or {}
            te = block["test"].get("by_segment_type") or {}
            slot_note = f"  types_train={tr}  types_test={te}"
        by_shape = block["train"].get("by_shape") or block["train"].get("by_segment_type") or {}
        by_shape_test = block["test"].get("by_shape") or block["test"].get("by_segment_type") or {}
        print(
            f"  {code} [{block.get('crop_kind', 'junction')}]: "
            f"train={block['train']['n']} {by_shape}  "
            f"test={block['test']['n']} {by_shape_test}"
            f"{slot_note}"
        )
    if result["shortfalls"]:
        print("[allocate] shortfalls:")
        for line in result["shortfalls"]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
