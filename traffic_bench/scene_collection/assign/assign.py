#!/usr/bin/env python3
"""Allocate n_train / n_test maps per sign from the shared global train/test split.

Shared pool: signs sample independently from the same train/test junction set
(a junction may be assigned to several signs). Within one sign, scene_ids are
unique.

``crop_kind`` in signs.yaml:
  * ``junction`` (default) — sample ``junc_*`` / ``rb_*`` ids from train/test json
  * ``dual_path`` — sample from ``crops/dual_path/{shape}/{slot}/`` whose
    ``junction_id`` is on the train/test side (via index), filtered by
    ``collect.dual_path.roles.sign_to_slots`` (+ stem / carriageway for 5.7).
    Within each shape quota, scene_ids are drawn **evenly across slots**
    (e.g. 3.18.1 balances ``r_s`` vs ``r_l``; 3.1 balances all six).
  * ``segment`` — sample from ``crops/segment/<scene_id>/`` using query
    fields in signs.yaml (``segment_types``, ``lane_count_min``,
    ``pass_right_ok`` / ``pass_left_ok``). Split by ``osm_way_id``.

Canonical quotas: ``splits/signs.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def _osm_way_is_test(osm_way: str, test_frac: float) -> bool:
    """Stable train/test stamp on OSM way id (not Python's salted hash())."""
    digest = hashlib.md5(str(osm_way).encode("utf-8")).hexdigest()
    return (int(digest, 16) % 10000) / 10000.0 < float(test_frac)


def _iter_segment_scene_dirs(segment_root: Path):
    """Yield crops/segment/<id>/; also leftover nested straight/curved dirs."""
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
    spec: dict,
    allowed_osm_ways: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """Return ``by_type[segment_type][segment_type] = [scene_id, ...]``."""
    allowed_types = {str(t) for t in (spec.get("segment_types") or ["straight", "curved"])}
    by_type: Dict[str, Dict[str, List[str]]] = {t: {} for t in allowed_types}
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
        by_type.setdefault(seg_type, {}).setdefault(seg_type, []).append(sid)
    for seg_type, by_sub in list(by_type.items()):
        for sub in list(by_sub):
            by_sub[sub] = sorted(set(by_sub[sub]))
    return by_type


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
    segment_root: Path,
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
        elif crop_kind == "segment":
            segment_types = set(spec.get("segment_types") or ["straight", "curved"])
            test_ways: Set[str] = set()
            train_ways: Set[str] = set()
            for row in segments_index.values():
                osm_way = str(row.get("osm_way_id") or "")
                if not osm_way:
                    continue
                if _osm_way_is_test(osm_way, test_frac):
                    test_ways.add(osm_way)
                else:
                    train_ways.add(osm_way)
            train_pool = _scan_segment_pool(
                segment_root,
                segments_index,
                spec=spec,
                allowed_osm_ways=train_ways,
            )
            test_pool = _scan_segment_pool(
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
        else:
            train_pool = {s: list(train_by_shape.get(s, [])) for s in shapes}
            test_pool = {s: list(test_by_shape.get(s, [])) for s in shapes}

        train_ids: List[str] = []
        test_ids: List[str] = []
        got_train: Dict[str, int] = {}
        got_test: Dict[str, int] = {}
        got_train_slots: Dict[str, Dict[str, int]] = {}
        got_test_slots: Dict[str, Dict[str, int]] = {}

        if crop_kind in ("dual_path", "segment"):
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
            block["shapes"] = []
            if spec.get("prepare"):
                block["prepare"] = spec["prepare"]
            if spec.get("lane_count_min") is not None:
                block["lane_count_min"] = spec["lane_count_min"]
            if spec.get("pass_right_ok"):
                block["pass_right_ok"] = True
            if spec.get("pass_left_ok"):
                block["pass_left_ok"] = True
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
    result = allocate(
        signs_cfg=signs_cfg,
        train_by_shape=train_doc.get("by_shape") or {},
        test_by_shape=test_doc.get("by_shape") or {},
        scene_to_junc=scene_to_junc,
        dual_root=args.dual_root,
        segment_root=args.segment_root,
        segments_index=segments_index,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[allocate] signs={len(result['signs'])} → {args.out}")
    for code, block in result["signs"].items():
        slot_note = ""
        if block.get("crop_kind") == "dual_path":
            tr = block["train"].get("by_slot") or {}
            te = block["test"].get("by_slot") or {}
            slot_note = f"  slots_train={tr}  slots_test={te}"
        elif block.get("crop_kind") == "segment":
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
