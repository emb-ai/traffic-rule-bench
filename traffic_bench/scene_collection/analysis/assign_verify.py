"""Verify tiered sign allocations: leak, counts, topology, reuse."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from traffic_bench.scene_collection.assign.places import (
    place_id_for_junction_scene,
    place_id_from_meta,
    place_id_from_way,
)
from traffic_bench.scene_collection.assign.taxonomy import (
    behavioral_family,
    semantic_group,
)
from traffic_bench.scene_collection.paths import (
    DUAL_PATH_CROPS,
    JUNCTIONS_INDEX,
    SEGMENTS_INDEX,
    SIGN_ALLOCATIONS,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_index_maps(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
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


def _load_segments_index(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("scene_id") or "")
            if sid:
                out[sid] = row
    return out


def _load_dual_meta(dual_root: Path) -> Dict[str, dict]:
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
        return place_id_from_way(way) if way else f"scene:{scene_id}"
    if crop_kind == "dual_path":
        meta = dual_meta.get(scene_id) or {}
        pid = place_id_from_meta(meta, scene_id=scene_id, crop_kind="dual_path")
        return pid or f"scene:{scene_id}"
    return place_id_for_junction_scene(scene_id, scene_to_junc)


def _classify_owners(signs: Set[str]) -> str:
    if len(signs) == 1:
        return "unique"
    behs = {behavioral_family(s) for s in signs}
    sems = {semantic_group(s) for s in signs}
    if len(behs) == 1:
        return "within_behavioral"
    if len(sems) == 1:
        return "within_semantic_diff_family"
    return "across_semantic"


def verify(alloc: dict, *, scene_to_junc, segments_index, dual_meta) -> Dict[str, Any]:
    places: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: {"train": set(), "test": set()}
    )
    per_sign: Dict[str, Any] = {}

    for code, block in (alloc.get("signs") or {}).items():
        crop = str(block.get("crop_kind") or "junction")
        row: Dict[str, Any] = {
            "crop_kind": crop,
            "semantic_group": block.get("semantic_group") or semantic_group(code),
            "behavioral_family": block.get("behavioral_family")
            or behavioral_family(code),
        }
        for split in ("train", "test"):
            ids = list((block.get(split) or {}).get("scene_ids") or [])
            for sid in ids:
                places[code][split].add(
                    _place_for(
                        scene_id=sid,
                        crop_kind=crop,
                        scene_to_junc=scene_to_junc,
                        segments_index=segments_index,
                        dual_meta=dual_meta,
                    )
                )
            topo = (block.get(split) or {}).get("by_shape") or (
                block.get(split) or {}
            ).get("by_segment_type") or {}
            row[f"n_{split}"] = int((block.get(split) or {}).get("n") or len(ids))
            row[f"places_{split}"] = len(places[code][split])
            row[f"topo_{split}"] = dict(topo)
            row[f"reuse_tiers_{split}"] = dict(
                (block.get(split) or {}).get("reuse_tiers") or {}
            )
        per_sign[code] = row

    within_leak = {
        code: sorted(places[code]["train"] & places[code]["test"])
        for code in places
        if places[code]["train"] & places[code]["test"]
    }
    global_train: Set[str] = set()
    global_test: Set[str] = set()
    for code in places:
        global_train |= places[code]["train"]
        global_test |= places[code]["test"]
    global_leak = sorted(global_train & global_test)

    counts_ok = all(
        row["n_train"] == int(alloc.get("n_train_target") or 80)
        and row["n_test"] == int(alloc.get("n_test_target") or 20)
        for row in per_sign.values()
    )

    reuse: Dict[str, Any] = {}
    for split in ("train", "test"):
        reg = (alloc.get("place_registry") or {}).get(split) or {}
        bucket: Counter = Counter()
        for _pid, signs in reg.items():
            bucket[_classify_owners(set(signs))] += 1
        total = sum(bucket.values())
        shared = total - bucket["unique"]
        reuse[split] = {
            "n_places": total,
            "unique": bucket["unique"],
            "within_behavioral": bucket["within_behavioral"],
            "within_semantic_diff_family": bucket["within_semantic_diff_family"],
            "across_semantic": bucket["across_semantic"],
            "unique_pct": (100.0 * bucket["unique"] / total) if total else 0.0,
            "shared": shared,
        }

    return {
        "policy": alloc.get("allocation_policy"),
        "n_signs": len(per_sign),
        "train_test": {
            "global_overlap_places": len(global_leak),
            "global_overlap_sample": global_leak[:20],
            "within_sign_leak_signs": sorted(within_leak),
            "within_sign_leak": {k: v[:10] for k, v in within_leak.items()},
            "train_place_union": len(global_train),
            "test_place_union": len(global_test),
            "ok": len(global_leak) == 0 and not within_leak,
        },
        "counts": {
            "target_train": alloc.get("n_train_target"),
            "target_test": alloc.get("n_test_target"),
            "all_signs_hit_target": counts_ok,
            "per_sign": per_sign,
        },
        "reuse": reuse,
        "shortfalls": list(alloc.get("shortfalls") or []),
    }


def _md(summary: Dict[str, Any]) -> str:
    tt = summary["train_test"]
    counts = summary["counts"]
    reuse = summary["reuse"]

    count_rows = []
    for code, row in sorted(
        counts["per_sign"].items(),
        key=lambda kv: (kv[1]["semantic_group"], kv[0]),
    ):
        count_rows.append(
            f"| `{code}` | `{row['behavioral_family']}` | `{row['crop_kind']}` | "
            f"{row['n_train']} | {row['n_test']} | {row['places_train']} | "
            f"{row['places_test']} | `{row['topo_train']}` | `{row['topo_test']}` |"
        )

    def reuse_block(split: str) -> str:
        r = reuse[split]
        shared = r["shared"] or 1
        return "\n".join(
            [
                f"| unique | {r['unique']} | {r['unique_pct']:.1f}% | — |",
                f"| within behavioral family | {r['within_behavioral']} | "
                f"{100.0 * r['within_behavioral'] / r['n_places']:.1f}% | "
                f"{100.0 * r['within_behavioral'] / shared:.1f}% of shared |",
                f"| within semantic, different family | "
                f"{r['within_semantic_diff_family']} | "
                f"{100.0 * r['within_semantic_diff_family'] / r['n_places']:.1f}% | "
                f"{100.0 * r['within_semantic_diff_family'] / shared:.1f}% of shared |",
                f"| across semantic groups | {r['across_semantic']} | "
                f"{100.0 * r['across_semantic'] / r['n_places']:.1f}% | "
                f"{100.0 * r['across_semantic'] / shared:.1f}% of shared |",
            ]
        )

    ok_tt = "PASS" if tt["ok"] else "FAIL"
    ok_counts = "PASS" if counts["all_signs_hit_target"] else "FAIL"
    ok_cross = "PASS" if all(
        reuse[s]["across_semantic"] == 0 for s in ("train", "test")
    ) else "FAIL"

    return f"""# Allocation verification

Policy: `{summary.get("policy")}` · signs: {summary["n_signs"]}

| Check | Result |
| --- | --- |
| Train↔test place overlap = 0 | **{ok_tt}** (global={tt["global_overlap_places"]}, within-sign={len(tt["within_sign_leak_signs"])}) |
| Per-sign scene counts = {counts["target_train"]}/{counts["target_test"]} | **{ok_counts}** |
| Cross-semantic reuse = 0 | **{ok_cross}** |
| Shortfalls | {len(summary["shortfalls"])} |

## Train↔test

- train place union: {tt["train_place_union"]}
- test place union: {tt["test_place_union"]}
- global train∩test: {tt["global_overlap_places"]}
- within-sign leaks: {tt["within_sign_leak_signs"] or "none"}

## Per-sign counts + topology

| PDD | Behavioral family | Crop | Train scenes | Test scenes | Train places | Test places | Train topo | Test topo |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(count_rows)}

## Place reuse (train)

| Bucket | # places | % of all | note |
| --- | ---: | ---: | --- |
{reuse_block("train")}

## Place reuse (test)

| Bucket | # places | % of all | note |
| --- | ---: | ---: | --- |
{reuse_block("test")}

## Reproduce

```bash
python -m traffic_bench.scene_collection assign
python -m traffic_bench.scene_collection analysis assign_verify
# or as part of:
python -m traffic_bench.scene_collection analysis overlap
```
"""


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allocations", type=Path, default=SIGN_ALLOCATIONS)
    ap.add_argument("--index", type=Path, default=JUNCTIONS_INDEX)
    ap.add_argument("--segments-index", type=Path, default=SEGMENTS_INDEX)
    ap.add_argument("--dual-root", type=Path, default=DUAL_PATH_CROPS)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.out_dir is None:
        from traffic_bench.scene_collection.paths import SCENE_COLLECTION

        args.out_dir = SCENE_COLLECTION / "analysis" / "overlap"

    if not args.allocations.is_file():
        raise SystemExit(f"ERROR: missing {args.allocations}")

    alloc = _load_json(args.allocations)
    summary = verify(
        alloc,
        scene_to_junc=_load_index_maps(args.index),
        segments_index=_load_segments_index(args.segments_index),
        dual_meta=_load_dual_meta(args.dual_root),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "allocation_verify.json"
    md_path = args.out_dir / "allocation_verify.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(_md(summary), encoding="utf-8")

    tt = summary["train_test"]
    print(f"[verify] signs={summary['n_signs']}")
    print(
        f"  train↔test places: global={tt['global_overlap_places']} "
        f"within={len(tt['within_sign_leak_signs'])} → "
        f"{'PASS' if tt['ok'] else 'FAIL'}"
    )
    print(
        f"  counts 80/20: "
        f"{'PASS' if summary['counts']['all_signs_hit_target'] else 'FAIL'}"
    )
    for split in ("train", "test"):
        r = summary["reuse"][split]
        print(
            f"  {split} reuse: unique={r['unique']} "
            f"intra_beh={r['within_behavioral']} "
            f"intra_sem={r['within_semantic_diff_family']} "
            f"across={r['across_semantic']}"
        )
    print(f"[verify] wrote {md_path}")
    return 0 if tt["ok"] and summary["counts"]["all_signs_hit_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
