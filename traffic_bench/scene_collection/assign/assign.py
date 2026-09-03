#!/usr/bin/env python3
"""Allocate n_train / n_test maps per sign with tiered place reuse within each split.

Pipeline:
  1. Global train/test split (``make_split``) — place-disjoint across halves.
  2. Signs processed in taxonomy order; each pick prefers:
       tier 1 — unused physical place in this split
       tier 2 — place already used by the same behavioral family
       tier 3 — place used in the same semantic group (different family)
     Cross-semantic reuse is rejected (shortfall error).

``crop_kind`` in signs.yaml:
  * ``junction`` — ``junc_*`` / ``rb_*`` from train/test json (by shape T/X/O)
  * ``dual_path`` — dual crops whose ``junction_id`` is on the train/test side
  * ``segment`` — corridor crops filtered by query; split by ``osm_way_id``

Canonical quotas: ``splits/signs.yaml``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from traffic_bench.scene_collection.assign.places import (
    place_id_for_junction_scene,
    place_id_from_meta,
    place_id_from_way,
)
from traffic_bench.scene_collection.assign.taxonomy import (
    behavioral_family,
    semantic_group,
    sign_sort_key,
    sign_taxonomy,
)
from traffic_bench.scene_collection.assign.tiered import (
    SceneCandidate,
    SplitPlaceRegistry,
    pick_many_tiered,
    pick_tiered,
)
from traffic_bench.scene_collection.collect.dual_path.roles import (
    scenario_matches_sign,
    sign_shape_policy,
    sign_to_slots,
)
from traffic_bench.scene_collection.collect.segments.metrics import enrich_lane_fields
from traffic_bench.scene_collection.paths import (
    DUAL_PATH_CROPS,
    JUNCTIONS_INDEX,
    SEGMENT_CROPS,
    SEGMENTS_INDEX,
    SEGMENT_TEST_IDS,
    SEGMENT_TRAIN_IDS,
    SIGN_ALLOCATIONS,
    SIGNS_YAML,
    TEST_IDS,
    TRAIN_IDS,
)

DEFAULT_SIGNS_YAML = SIGNS_YAML
DEFAULT_TRAIN = TRAIN_IDS
DEFAULT_TEST = TEST_IDS
DEFAULT_INDEX = JUNCTIONS_INDEX
DEFAULT_SEGMENTS_INDEX = SEGMENTS_INDEX
DEFAULT_DUAL_ROOT = DUAL_PATH_CROPS
DEFAULT_SEGMENT_ROOT = SEGMENT_CROPS
DEFAULT_OUT = SIGN_ALLOCATIONS


def load_signs_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"PyYAML is required to read {path}. Install with: pip install pyyaml"
        ) from exc
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_load_yaml = load_signs_yaml


def counts_for_sign(
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


_counts_for_sign = counts_for_sign


def sample(pool: List[str], k: int, rng: random.Random) -> List[str]:
    if k <= 0:
        return []
    if k > len(pool):
        return sorted(pool)
    return sorted(rng.sample(pool, k))


_sample = sample


def _slot_quotas(slots: List[str], need: int) -> Dict[str, int]:
    slots = [s for s in slots if s]
    if not slots or need <= 0:
        return {}
    base, rem = divmod(int(need), len(slots))
    return {s: base + (1 if i < rem else 0) for i, s in enumerate(slots)}


def _load_index_maps(index_path: Path) -> Dict[str, str]:
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


def _load_segments_index(path: Path) -> Dict[str, Dict]:
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


def _load_way_split(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    doc = json.loads(path.read_text(encoding="utf-8"))
    return set(str(x) for x in (doc.get("all") or []))


def _segment_matches_query(meta: dict, spec: dict) -> bool:
    meta = enrich_lane_fields(meta)
    types = {str(t) for t in (spec.get("segment_types") or ["straight", "curved"])}
    if str(meta.get("segment_type") or "") not in types:
        return False
    min_lanes = spec.get("lane_count_min")
    if min_lanes is not None and int(meta.get("lane_count") or 0) < int(min_lanes):
        return False
    if spec.get("pass_right_ok") and not meta.get("pass_right_ok"):
        return False
    if spec.get("pass_left_ok") and not meta.get("pass_left_ok"):
        return False
    return True


def _iter_segment_scene_dirs(segment_root: Path):
    if not segment_root.is_dir():
        return
    for child in sorted(segment_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"straight", "curved"}:
            for inner in sorted(child.iterdir()):
                if inner.is_dir() and (inner / "meta.json").is_file():
                    yield inner
            continue
        if (child / "meta.json").is_file():
            yield child


def _junction_candidates(
    pool_by_shape: Dict[str, List[str]],
    *,
    scene_to_junc: Dict[str, str],
) -> Dict[str, List[SceneCandidate]]:
    out: Dict[str, List[SceneCandidate]] = {}
    for shape, ids in pool_by_shape.items():
        out[shape] = [
            SceneCandidate(
                scene_id=sid,
                place_id=place_id_for_junction_scene(sid, scene_to_junc),
                shape=shape,
            )
            for sid in sorted(set(ids))
        ]
    return out


def _scan_dual_path_candidates(
    dual_root: Path,
    *,
    allowed_shapes: Set[str],
    allowed_slots: Set[str],
    allowed_junctions: Set[str],
    pdd_code: str,
) -> Dict[str, Dict[str, List[SceneCandidate]]]:
    by_shape: Dict[str, Dict[str, List[SceneCandidate]]] = {
        s: {} for s in allowed_shapes
    }
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
                pid = place_id_from_meta(meta, scene_id=sid, crop_kind="dual_path")
                if pid is None:
                    continue
                by_shape.setdefault(shape, {}).setdefault(slot, []).append(
                    SceneCandidate(
                        scene_id=sid,
                        place_id=pid,
                        shape=shape,
                        slot=slot,
                    )
                )
    for shape, by_slot in list(by_shape.items()):
        for slot in list(by_slot):
            by_slot[slot] = sorted(by_slot[slot], key=lambda c: c.scene_id)
    return by_shape


def _scan_segment_candidates(
    segment_root: Path,
    segments_index: Dict[str, Dict],
    *,
    spec: dict,
    allowed_osm_ways: Set[str],
) -> Dict[str, List[SceneCandidate]]:
    allowed_types = {str(t) for t in (spec.get("segment_types") or ["straight", "curved"])}
    by_type: Dict[str, List[SceneCandidate]] = {t: [] for t in allowed_types}
    for scene_dir in _iter_segment_scene_dirs(segment_root):
        if not (scene_dir / "map.net.xml").is_file():
            continue
        meta_path = scene_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = str(meta.get("scene_name") or meta.get("scene_id") or scene_dir.name)
        index_row = segments_index.get(sid) or {}
        merged = enrich_lane_fields({**index_row, **meta})
        if not _segment_matches_query(merged, spec):
            continue
        osm_way = str(merged.get("osm_way_id") or "")
        if allowed_osm_ways and osm_way not in allowed_osm_ways:
            continue
        seg_type = str(merged.get("segment_type") or "")
        by_type.setdefault(seg_type, []).append(
            SceneCandidate(
                scene_id=sid,
                place_id=place_id_from_way(osm_way),
                shape="segment",
                segment_type=seg_type,
            )
        )
    for seg_type, rows in list(by_type.items()):
        by_type[seg_type] = sorted(rows, key=lambda c: c.scene_id)
    return by_type


def _pick_with_shape_balance(
    candidates_by_key: Dict[str, List[SceneCandidate]],
    need_by_key: Dict[str, int],
    *,
    registry: SplitPlaceRegistry,
    pdd_code: str,
    rng: random.Random,
) -> Tuple[List[SceneCandidate], Dict[str, int], Dict[int, int]]:
    """Pick scenes one-by-one, always taking from the bucket with highest remaining quota."""
    picked: List[SceneCandidate] = []
    remaining = {k: int(v) for k, v in need_by_key.items() if int(v) > 0}
    got: Dict[str, int] = defaultdict(int)
    tier_hist: Dict[int, int] = defaultdict(int)
    used_scenes: Set[str] = set()

    while remaining and sum(remaining.values()) > 0:
        key = max(remaining, key=lambda k: remaining[k])
        if remaining[key] <= 0:
            break
        pool = candidates_by_key.get(key) or []
        result = pick_tiered(
            pool,
            registry=registry,
            pdd_code=pdd_code,
            exclude_scene_ids=used_scenes,
            rng=rng,
        )
        if result is None:
            remaining[key] = 0
            continue
        chosen, tier = result
        picked.append(chosen)
        used_scenes.add(chosen.scene_id)
        registry.register(chosen.place_id, pdd_code, tier=tier)
        tier_hist[int(tier)] += 1
        got[key] += 1
        remaining[key] -= 1
        if remaining[key] <= 0:
            del remaining[key]
    return picked, dict(got), dict(tier_hist)


def _pick_dual_path(
    pool: Dict[str, Dict[str, List[SceneCandidate]]],
    need_by_shape: Dict[str, int],
    slots: Set[str],
    *,
    registry: SplitPlaceRegistry,
    pdd_code: str,
    rng: random.Random,
) -> Tuple[List[SceneCandidate], Dict[str, int], Dict[str, Dict[str, int]], Dict[int, int]]:
    picked: List[SceneCandidate] = []
    got_shape: Dict[str, int] = defaultdict(int)
    got_slot: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tier_hist: Dict[int, int] = defaultdict(int)

    for shape, need in sorted(need_by_shape.items()):
        if need <= 0:
            continue
        slot_list = sorted(slots & set((pool.get(shape) or {}).keys()))
        slot_need = _slot_quotas(slot_list, need)
        flat_need = {f"{shape}/{slot}": slot_need.get(slot, 0) for slot in slot_list}
        flat_cands = {
            f"{shape}/{slot}": list((pool.get(shape) or {}).get(slot) or [])
            for slot in slot_list
        }
        chunk, got_keys, tiers = _pick_with_shape_balance(
            flat_cands,
            flat_need,
            registry=registry,
            pdd_code=pdd_code,
            rng=rng,
        )
        picked.extend(chunk)
        got_shape[shape] += len(chunk)
        for tier, n in tiers.items():
            tier_hist[tier] += n
        for key, n in got_keys.items():
            _, slot = key.split("/", 1)
            got_slot[shape][slot] = got_slot[shape].get(slot, 0) + n
    return picked, dict(got_shape), {k: dict(v) for k, v in got_slot.items()}, dict(tier_hist)


def allocate(
    *,
    signs_cfg: dict,
    train_by_shape: Dict[str, List[str]],
    test_by_shape: Dict[str, List[str]],
    scene_to_junc: Dict[str, str],
    dual_root: Path,
    segment_root: Path,
    segments_index: Dict[str, Dict],
    train_ways: Set[str],
    test_ways: Set[str],
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

    train_registry = SplitPlaceRegistry()
    test_registry = SplitPlaceRegistry()
    allocations: Dict[str, Any] = {}
    shortfalls: List[str] = []

    ordered_signs = sorted(signs.items(), key=lambda kv: sign_sort_key(str(kv[0])))

    for sign_code, spec in ordered_signs:
        spec = spec or {}
        tax = sign_taxonomy(str(sign_code), spec)
        crop_kind = tax.crop_kind
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

        block: Dict[str, Any] = {
            "crop_kind": crop_kind,
            "shapes": shapes,
            "x_share": x_share,
            "semantic_group": tax.semantic_group,
            "behavioral_family": tax.behavioral_family,
            "compatible_topologies": sorted(tax.topologies),
        }

        if crop_kind == "dual_path":
            try:
                slots = set(sign_to_slots(str(sign_code)))
                shape_policy = set(sign_shape_policy(str(sign_code)))
            except ValueError as exc:
                shortfalls.append(f"{sign_code}: {exc}")
                continue
            shapes = [s for s in shapes if s in shape_policy] or sorted(shape_policy)
            train_pool = _scan_dual_path_candidates(
                dual_root,
                allowed_shapes=set(shapes),
                allowed_slots=slots,
                allowed_junctions=train_juncs,
                pdd_code=str(sign_code),
            )
            test_pool = _scan_dual_path_candidates(
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
            train_picked, got_train, got_train_slots, tr_tiers = _pick_dual_path(
                train_pool,
                train_need,
                slots,
                registry=train_registry,
                pdd_code=str(sign_code),
                rng=rng_train,
            )
            test_picked, got_test, got_test_slots, te_tiers = _pick_dual_path(
                test_pool,
                test_need,
                slots,
                registry=test_registry,
                pdd_code=str(sign_code),
                rng=rng_test,
            )
            train_ids = [c.scene_id for c in train_picked]
            test_ids = [c.scene_id for c in test_picked]
            block["slots"] = sorted(slots)
            block["train"] = {
                "scene_ids": sorted(set(train_ids)),
                "by_shape": got_train,
                "by_slot": got_train_slots,
                "reuse_tiers": tr_tiers,
                "n": len(set(train_ids)),
            }
            block["test"] = {
                "scene_ids": sorted(set(test_ids)),
                "by_shape": got_test,
                "by_slot": got_test_slots,
                "reuse_tiers": te_tiers,
                "n": len(set(test_ids)),
            }
            if len(set(train_ids)) < sign_n_train:
                shortfalls.append(
                    f"{sign_code} train: want {sign_n_train}, got {len(set(train_ids))}"
                )
            if len(set(test_ids)) < sign_n_test:
                shortfalls.append(
                    f"{sign_code} test: want {sign_n_test}, got {len(set(test_ids))}"
                )

        elif crop_kind == "segment":
            segment_types = set(spec.get("segment_types") or ["straight", "curved"])
            train_pool = _scan_segment_candidates(
                segment_root,
                segments_index,
                spec=spec,
                allowed_osm_ways=train_ways,
            )
            test_pool = _scan_segment_candidates(
                segment_root,
                segments_index,
                spec=spec,
                allowed_osm_ways=test_ways,
            )
            train_need = {t: sign_n_train // len(segment_types) for t in segment_types}
            test_need = {t: sign_n_test // len(segment_types) for t in segment_types}
            for i, t in enumerate(sorted(segment_types)):
                if i < sign_n_train % len(segment_types):
                    train_need[t] += 1
                if i < sign_n_test % len(segment_types):
                    test_need[t] += 1
            train_picked, got_train, tr_tiers = _pick_with_shape_balance(
                train_pool,
                train_need,
                registry=train_registry,
                pdd_code=str(sign_code),
                rng=rng_train,
            )
            test_picked, got_test, te_tiers = _pick_with_shape_balance(
                test_pool,
                test_need,
                registry=test_registry,
                pdd_code=str(sign_code),
                rng=rng_test,
            )
            train_ids = [c.scene_id for c in train_picked]
            test_ids = [c.scene_id for c in test_picked]
            block["segment_types"] = sorted(segment_types)
            block["shapes"] = []
            block["train"] = {
                "scene_ids": sorted(set(train_ids)),
                "by_segment_type": got_train,
                "reuse_tiers": tr_tiers,
                "n": len(set(train_ids)),
            }
            block["test"] = {
                "scene_ids": sorted(set(test_ids)),
                "by_segment_type": got_test,
                "reuse_tiers": te_tiers,
                "n": len(set(test_ids)),
            }
            if spec.get("prepare"):
                block["prepare"] = spec["prepare"]
            if spec.get("lane_count_min") is not None:
                block["lane_count_min"] = spec["lane_count_min"]
            if spec.get("pass_right_ok"):
                block["pass_right_ok"] = True
            if spec.get("pass_left_ok"):
                block["pass_left_ok"] = True
            if len(set(train_ids)) < sign_n_train:
                shortfalls.append(
                    f"{sign_code} train: want {sign_n_train}, got {len(set(train_ids))}"
                )
            if len(set(test_ids)) < sign_n_test:
                shortfalls.append(
                    f"{sign_code} test: want {sign_n_test}, got {len(set(test_ids))}"
                )

        else:
            train_pool = _junction_candidates(
                {s: list(train_by_shape.get(s, [])) for s in shapes},
                scene_to_junc=scene_to_junc,
            )
            test_pool = _junction_candidates(
                {s: list(test_by_shape.get(s, [])) for s in shapes},
                scene_to_junc=scene_to_junc,
            )
            train_picked, got_train, tr_tiers = _pick_with_shape_balance(
                train_pool,
                train_need,
                registry=train_registry,
                pdd_code=str(sign_code),
                rng=rng_train,
            )
            test_picked, got_test, te_tiers = _pick_with_shape_balance(
                test_pool,
                test_need,
                registry=test_registry,
                pdd_code=str(sign_code),
                rng=rng_test,
            )
            train_ids = [c.scene_id for c in train_picked]
            test_ids = [c.scene_id for c in test_picked]
            block["train"] = {
                "scene_ids": sorted(set(train_ids)),
                "by_shape": got_train,
                "reuse_tiers": tr_tiers,
                "n": len(set(train_ids)),
            }
            block["test"] = {
                "scene_ids": sorted(set(test_ids)),
                "by_shape": got_test,
                "reuse_tiers": te_tiers,
                "n": len(set(test_ids)),
            }
            if len(set(train_ids)) < sign_n_train:
                shortfalls.append(
                    f"{sign_code} train: want {sign_n_train}, got {len(set(train_ids))}"
                )
            if len(set(test_ids)) < sign_n_test:
                shortfalls.append(
                    f"{sign_code} test: want {sign_n_test}, got {len(set(test_ids))}"
                )

        allocations[str(sign_code)] = block

    return {
        "seed": seed,
        "n_train_target": n_train,
        "n_test_target": default_n_test,
        "test_frac": test_frac,
        "allocation_policy": "tiered_place_reuse",
        "tier_rules": [
            "1 unique physical place in split",
            "2 same behavioral family",
            "3 same semantic group, different behavioral family",
            "cross-semantic reuse rejected",
        ],
        "shortfalls": shortfalls,
        "place_registry": {
            "train": train_registry.snapshot(),
            "test": test_registry.snapshot(),
        },
        "signs": allocations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signs-yaml", type=Path, default=DEFAULT_SIGNS_YAML)
    ap.add_argument("--train-ids", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--test-ids", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--segment-train-ids", type=Path, default=SEGMENT_TRAIN_IDS)
    ap.add_argument("--segment-test-ids", type=Path, default=SEGMENT_TEST_IDS)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--segments-index", type=Path, default=DEFAULT_SEGMENTS_INDEX)
    ap.add_argument("--dual-root", type=Path, default=DEFAULT_DUAL_ROOT)
    ap.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
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
    train_ways = _load_way_split(args.segment_train_ids)
    test_ways = _load_way_split(args.segment_test_ids)
    if not train_ways or not test_ways:
        # Fallback: derive from index with same stratified logic as make_split.
        from traffic_bench.scene_collection.collect.make_split import split_segments_by_type

        rows = list(segments_index.values())
        if rows:
            tr, te = split_segments_by_type(
                rows,
                test_frac=float(signs_cfg.get("test_frac", 0.2)),
                seed=int(signs_cfg.get("seed", 42)),
            )
            train_ways = set(w for ways in tr.values() for w in ways)
            test_ways = set(w for ways in te.values() for w in ways)
            print(
                "[allocate] segment split files missing; derived from segments index"
            )

    result = allocate(
        signs_cfg=signs_cfg,
        train_by_shape=train_doc.get("by_shape") or {},
        test_by_shape=test_doc.get("by_shape") or {},
        scene_to_junc=scene_to_junc,
        dual_root=args.dual_root,
        segment_root=args.segment_root,
        segments_index=segments_index,
        train_ways=train_ways,
        test_ways=test_ways,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[allocate] signs={len(result['signs'])} → {args.out}")
    for code, block in result["signs"].items():
        tr = block["train"].get("reuse_tiers") or {}
        te = block["test"].get("reuse_tiers") or {}
        print(
            f"  {code} [{block.get('crop_kind', 'junction')}] "
            f"{block.get('behavioral_family')} "
            f"train={block['train']['n']} tiers={tr}  "
            f"test={block['test']['n']} tiers={te}"
        )
    if result["shortfalls"]:
        print("[allocate] shortfalls:")
        for line in result["shortfalls"]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
