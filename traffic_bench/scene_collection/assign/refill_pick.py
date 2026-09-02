"""Tiered refill picks — same place-reuse policy as ``assign``.

Materialize refill used to draw randomly from junction ``train_ids`` /
``test_ids`` only, ignoring:
  * crop_kind (segment / dual_path got junction IDs)
  * place identity and cross-semantic bans

This module rebuilds per-split ``SplitPlaceRegistry`` from the full
``sign_allocations.json`` and picks replacements with ``pick_tiered``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from traffic_bench.scene_collection.assign.assign import (
    _junction_candidates,
    _junction_ids_for_split,
    _load_segments_index,
    _load_way_split,
    _pick_dual_path,
    _pick_with_shape_balance,
    _scan_dual_path_candidates,
    _scan_segment_candidates,
    counts_for_sign,
    load_signs_yaml,
)
from traffic_bench.scene_collection.assign.places import (
    place_id_for_junction_scene,
    place_id_from_meta,
    place_id_from_way,
)
from traffic_bench.scene_collection.assign.taxonomy import sign_taxonomy
from traffic_bench.scene_collection.assign.tiered import SplitPlaceRegistry
from traffic_bench.scene_collection.collect.dual_path.roles import (
    sign_shape_policy,
    sign_to_slots,
)
from traffic_bench.scene_collection.paths import (
    DUAL_PATH_CROPS,
    JUNCTIONS_INDEX,
    SEGMENT_CROPS,
    SEGMENT_TEST_IDS,
    SEGMENT_TRAIN_IDS,
    SEGMENTS_INDEX,
    SIGNS_YAML,
    TEST_IDS,
    TRAIN_IDS,
)

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_to_junc(index_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not index_path.is_file():
        return out
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("scene_id") or "")
            jid = str(row.get("junction_id") or "")
            if sid and jid:
                out[sid] = jid
    return out


def _dual_meta(dual_root: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not dual_root.is_dir():
        return out
    for shape_dir in dual_root.iterdir():
        if not shape_dir.is_dir():
            continue
        for slot_dir in shape_dir.iterdir():
            if not slot_dir.is_dir():
                continue
            for scene_dir in slot_dir.iterdir():
                meta_path = scene_dir / "meta.json"
                if not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                sid = str(meta.get("scene_id") or scene_dir.name)
                out[sid] = meta
    return out


def _place_for(
    *,
    scene_id: str,
    crop_kind: str,
    scene_to_junc: Dict[str, str],
    segments_index: Dict[str, dict],
    dual_meta: Dict[str, dict],
) -> str:
    if crop_kind == "segment":
        way = str((segments_index.get(scene_id) or {}).get("osm_way_id") or "")
        if way:
            return place_id_from_way(way)
        pid = place_id_from_meta({"scene_id": scene_id}, scene_id=scene_id, crop_kind="segment")
        return pid or f"scene:{scene_id}"
    if crop_kind == "dual_path":
        meta = dual_meta.get(scene_id) or {}
        pid = place_id_from_meta(meta, scene_id=scene_id, crop_kind="dual_path")
        return pid or f"scene:{scene_id}"
    return place_id_for_junction_scene(scene_id, scene_to_junc)


def registries_from_allocations(
    alloc_doc: dict,
    *,
    scene_to_junc: Dict[str, str],
    segments_index: Dict[str, dict],
    dual_meta: Dict[str, dict],
) -> Tuple[SplitPlaceRegistry, SplitPlaceRegistry]:
    """Seed train/test place registries from the full allocation table."""
    train_reg = SplitPlaceRegistry()
    test_reg = SplitPlaceRegistry()
    for code, block in (alloc_doc.get("signs") or {}).items():
        crop = str(block.get("crop_kind") or "junction")
        for split, reg in (("train", train_reg), ("test", test_reg)):
            half = block.get(split) or {}
            for sid in half.get("scene_ids") or []:
                pid = _place_for(
                    scene_id=str(sid),
                    crop_kind=crop,
                    scene_to_junc=scene_to_junc,
                    segments_index=segments_index,
                    dual_meta=dual_meta,
                )
                reg.used.setdefault(pid, set()).add(str(code))
    return train_reg, test_reg


def pick_refill_ids_tiered(
    *,
    pdd_code: str,
    half: str,
    need: int,
    excluded_scene_ids: Set[str],
    alloc_doc: dict,
    signs_yaml: Path = SIGNS_YAML,
    train_ids_path: Path = TRAIN_IDS,
    test_ids_path: Path = TEST_IDS,
    segment_train_ids: Path = SEGMENT_TRAIN_IDS,
    segment_test_ids: Path = SEGMENT_TEST_IDS,
    junctions_index: Path = JUNCTIONS_INDEX,
    segments_index_path: Path = SEGMENTS_INDEX,
    dual_root: Path = DUAL_PATH_CROPS,
    segment_root: Path = SEGMENT_CROPS,
    seed: Optional[int] = None,
) -> Tuple[List[str], Dict[int, int]]:
    """Pick up to ``need`` refill scenes for one sign/half under tiered policy.

    Returns ``(scene_ids, reuse_tier_histogram)``.
    """
    if need <= 0:
        return [], {}

    signs_cfg = load_signs_yaml(signs_yaml)
    spec = (signs_cfg.get("signs") or {}).get(pdd_code) or {}
    tax = sign_taxonomy(str(pdd_code), spec)
    crop_kind = tax.crop_kind
    seed = int(signs_cfg.get("seed", 42) if seed is None else seed)
    rng = random.Random(f"{seed}|{pdd_code}|refill|{half}|{need}|{len(excluded_scene_ids)}")

    scene_to_junc = _scene_to_junc(junctions_index)
    segments_index = _load_segments_index(segments_index_path)
    dual_meta = _dual_meta(dual_root)
    train_reg, test_reg = registries_from_allocations(
        alloc_doc,
        scene_to_junc=scene_to_junc,
        segments_index=segments_index,
        dual_meta=dual_meta,
    )
    registry = train_reg if half == "train" else test_reg

    train_doc = _load_json(train_ids_path) if train_ids_path.is_file() else {}
    test_doc = _load_json(test_ids_path) if test_ids_path.is_file() else {}
    train_by_shape = train_doc.get("by_shape") or {}
    test_by_shape = test_doc.get("by_shape") or {}
    train_ways = _load_way_split(segment_train_ids)
    test_ways = _load_way_split(segment_test_ids)
    train_juncs = _junction_ids_for_split(
        [sid for ids in train_by_shape.values() for sid in ids], scene_to_junc
    )
    test_juncs = _junction_ids_for_split(
        [sid for ids in test_by_shape.values() for sid in ids], scene_to_junc
    )

    shapes = [str(s).upper() for s in (spec.get("shapes") or ["T", "X"])]
    x_share = spec.get("x_share", spec.get("x_frac"))
    need_by_key = counts_for_sign(shapes=shapes, n_total=need, x_share=x_share)

    # Drop candidates already excluded (live / rejected / prior alloc for this sign).
    def _filter_pool(pool_by_key: Dict[str, list]) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for key, cands in pool_by_key.items():
            kept = [c for c in cands if c.scene_id not in excluded_scene_ids]
            if kept:
                out[key] = kept
        return out

    if crop_kind == "dual_path":
        slots = set(sign_to_slots(str(pdd_code)))
        shape_policy = set(sign_shape_policy(str(pdd_code)))
        shapes = [s for s in shapes if s in shape_policy] or sorted(shape_policy)
        need_by_key = counts_for_sign(shapes=shapes, n_total=need, x_share=x_share)
        allowed_juncs = train_juncs if half == "train" else test_juncs
        pool = _scan_dual_path_candidates(
            dual_root,
            allowed_shapes=set(shapes),
            allowed_slots=slots,
            allowed_junctions=allowed_juncs,
            pdd_code=str(pdd_code),
        )
        # Filter excluded inside each slot list.
        for shape, by_slot in list(pool.items()):
            for slot, cands in list(by_slot.items()):
                by_slot[slot] = [c for c in cands if c.scene_id not in excluded_scene_ids]
        picked, _got_shape, _got_slots, tiers = _pick_dual_path(
            pool,
            need_by_key,
            slots,
            registry=registry,
            pdd_code=str(pdd_code),
            rng=rng,
        )
    elif crop_kind == "segment":
        segment_types = set(spec.get("segment_types") or ["straight", "curved"])
        allowed_ways = train_ways if half == "train" else test_ways
        pool = _scan_segment_candidates(
            segment_root,
            segments_index,
            spec=spec,
            allowed_osm_ways=allowed_ways,
        )
        pool = _filter_pool(pool)
        type_need = {t: need // len(segment_types) for t in segment_types}
        for i, t in enumerate(sorted(segment_types)):
            if i < need % len(segment_types):
                type_need[t] += 1
        picked, _got, tiers = _pick_with_shape_balance(
            pool,
            type_need,
            registry=registry,
            pdd_code=str(pdd_code),
            rng=rng,
        )
    else:
        by_shape = train_by_shape if half == "train" else test_by_shape
        pool = _junction_candidates(
            {s: list(by_shape.get(s, [])) for s in shapes},
            scene_to_junc=scene_to_junc,
        )
        pool = _filter_pool(pool)
        picked, _got, tiers = _pick_with_shape_balance(
            pool,
            need_by_key,
            registry=registry,
            pdd_code=str(pdd_code),
            rng=rng,
        )

    # Reflect picks into alloc_doc registries for subsequent halves in the same
    # refill call — caller should also write scene ids into allocations.
    for cand in picked:
        # Ownership already registered inside pick helpers.
        excluded_scene_ids.add(cand.scene_id)

    ids = [c.scene_id for c in picked]
    if len(ids) < need:
        print(
            f"  [refill] tiered shortfall {pdd_code}/{half}: "
            f"want {need}, got {len(ids)} (tiers={tiers})"
        )
    else:
        print(f"  [refill] tiered pick {pdd_code}/{half}: {len(ids)} (tiers={tiers})")
    return ids, {int(k): int(v) for k, v in tiers.items()}
