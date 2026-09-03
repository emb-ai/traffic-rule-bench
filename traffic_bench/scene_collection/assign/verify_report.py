#!/usr/bin/env python3
"""Verify tiered sign allocations against the design checklist.

Checks:
  1. train/test physical place overlap = 0
  2. per-sign scene counts (target n_train / n_test)
  3. topology quotas (T/X, O, segment types)
  4. intra-behavioral / within-semantic / across-semantic place reuse

Reads ``sign_allocations.json`` (+ junction/segment indexes for place ids).
Writes a markdown + JSON report under ``analysis/assign/`` by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
    SCENE_COLLECTION,
    SEGMENTS_INDEX,
    SIGN_ALLOCATIONS,
    SIGNS_YAML,
)

DEFAULT_OUT = SCENE_COLLECTION / "analysis" / "assign"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signs_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_scene_to_junc(index_path: Path) -> Dict[str, str]:
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


def build_verification(
    *,
    alloc: dict,
    signs_cfg: dict,
    scene_to_junc: Dict[str, str],
    segments_index: Dict[str, dict],
    dual_meta: Dict[str, dict],
) -> Dict[str, Any]:
    n_train_target = int(signs_cfg.get("n_train", 80))
    n_test_target = int(
        signs_cfg.get("n_test")
        or max(1, round(n_train_target * float(signs_cfg.get("test_frac", 0.2)) / 0.8))
    )

    places: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: {"train": set(), "test": set()}
    )
    per_sign: Dict[str, Any] = {}
    topology_ok: List[str] = []
    topology_warn: List[str] = []

    for code, block in sorted((alloc.get("signs") or {}).items()):
        crop = str(block.get("crop_kind") or "junction")
        for split in ("train", "test"):
            for sid in block[split].get("scene_ids") or []:
                places[code][split].add(
                    _place_for(
                        scene_id=sid,
                        crop_kind=crop,
                        scene_to_junc=scene_to_junc,
                        segments_index=segments_index,
                        dual_meta=dual_meta,
                    )
                )
        topo_train = block["train"].get("by_shape") or block["train"].get(
            "by_segment_type"
        ) or {}
        topo_test = block["test"].get("by_shape") or block["test"].get(
            "by_segment_type"
        ) or {}
        per_sign[code] = {
            "crop_kind": crop,
            "semantic_group": block.get("semantic_group") or semantic_group(code),
            "behavioral_family": block.get("behavioral_family")
            or behavioral_family(code),
            "n_train": int(block["train"]["n"]),
            "n_test": int(block["test"]["n"]),
            "train_places": len(places[code]["train"]),
            "test_places": len(places[code]["test"]),
            "topo_train": dict(topo_train),
            "topo_test": dict(topo_test),
            "reuse_tiers_train": block["train"].get("reuse_tiers") or {},
            "reuse_tiers_test": block["test"].get("reuse_tiers") or {},
            "count_ok": block["train"]["n"] == n_train_target
            and block["test"]["n"] == n_test_target,
        }

        # Topology vs signs.yaml target for T/X signs.
        spec = (signs_cfg.get("signs") or {}).get(code) or {}
        shapes = [str(s).upper() for s in (spec.get("shapes") or [])]
        if set(shapes) == {"T", "X"}:
            for split, topo in (("train", topo_train), ("test", topo_test)):
                total = sum(int(v) for v in topo.values()) or 1
                x_pct = 100.0 * int(topo.get("X", 0)) / total
                if abs(x_pct - 50.0) <= 1.0:
                    topology_ok.append(f"{code} {split} X%={x_pct:.0f}")
                else:
                    topology_warn.append(f"{code} {split} X%={x_pct:.0f} (want ~50)")

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

    reuse: Dict[str, Any] = {}
    for split in ("train", "test"):
        reg = (alloc.get("place_registry") or {}).get(split) or {}
        bucket = Counter()
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
            "unique_pct": round(100.0 * bucket["unique"] / total, 2) if total else 0.0,
            "within_behavioral_pct": round(
                100.0 * bucket["within_behavioral"] / total, 2
            )
            if total
            else 0.0,
            "within_semantic_diff_family_pct": round(
                100.0 * bucket["within_semantic_diff_family"] / total, 2
            )
            if total
            else 0.0,
            "across_semantic_pct": round(
                100.0 * bucket["across_semantic"] / total, 2
            )
            if total
            else 0.0,
            "shared": shared,
        }

    tier_totals = {
        split: dict(
            Counter(
                {
                    int(t): sum(
                        int((block[split].get("reuse_tiers") or {}).get(t, 0))
                        for block in (alloc.get("signs") or {}).values()
                    )
                    for t in ("1", "2", "3", 1, 2, 3)
                }
            )
        )
        for split in ("train", "test")
    }
    # clean tier keys to int-only
    for split in tier_totals:
        cleaned: Dict[int, int] = defaultdict(int)
        for k, v in tier_totals[split].items():
            cleaned[int(k)] += int(v)
        tier_totals[split] = dict(sorted(cleaned.items()))

    checks = {
        "train_test_place_overlap_zero": len(global_leak) == 0
        and not within_leak,
        "per_sign_counts_balanced": all(
            row["count_ok"] for row in per_sign.values()
        ),
        "topology_balanced": len(topology_warn) == 0,
        "across_semantic_reuse_zero": all(
            reuse[s]["across_semantic"] == 0 for s in ("train", "test")
        ),
    }

    return {
        "n_signs": len(per_sign),
        "n_train_target": n_train_target,
        "n_test_target": n_test_target,
        "checks": checks,
        "all_passed": all(checks.values()),
        "train_test": {
            "global_overlap_places": len(global_leak),
            "global_overlap_sample": global_leak[:20],
            "within_sign_leaks": {k: v[:10] for k, v in within_leak.items()},
            "train_place_union": len(global_train),
            "test_place_union": len(global_test),
        },
        "per_sign": per_sign,
        "topology_ok": topology_ok,
        "topology_warn": topology_warn,
        "reuse": reuse,
        "tier_pick_totals": tier_totals,
        "shortfalls": alloc.get("shortfalls") or [],
    }


def write_report(summary: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    checks = summary["checks"]
    tt = summary["train_test"]
    reuse_tr = summary["reuse"]["train"]
    reuse_te = summary["reuse"]["test"]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    rows = []
    for code, row in sorted(
        summary["per_sign"].items(),
        key=lambda kv: (kv[1]["semantic_group"], kv[0]),
    ):
        rows.append(
            f"| `{code}` | {row['behavioral_family']} | {row['crop_kind']} | "
            f"{row['n_train']} | {row['n_test']} | {row['train_places']} | "
            f"{row['test_places']} | `{row['topo_train']}` | `{row['topo_test']}` |"
        )

    md = f"""# Assign verification

Policy: tiered place reuse within each split (unique → same behavioral family →
same semantic group). Cross-semantic reuse is rejected.

## Checklist

| Check | Result |
| --- | --- |
| train/test place overlap = 0 | **{mark(checks['train_test_place_overlap_zero'])}** |
| per-sign scene counts balanced ({summary['n_train_target']}/{summary['n_test_target']}) | **{mark(checks['per_sign_counts_balanced'])}** |
| topology distribution balanced | **{mark(checks['topology_balanced'])}** |
| across-semantic reuse = 0 | **{mark(checks['across_semantic_reuse_zero'])}** |

Overall: **{"PASS" if summary["all_passed"] else "FAIL"}**

## 1. Train ↔ test place overlap

| Metric | Value |
| --- | ---: |
| Global train∩test places | {tt["global_overlap_places"]} |
| Within-sign leaks | {len(tt["within_sign_leaks"])} |
| Train place union | {tt["train_place_union"]} |
| Test place union | {tt["test_place_union"]} |

## 2–3. Per-sign counts and topology

| PDD | Behavioral family | Crop | Train | Test | Train places | Test places | Topo train | Topo test |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(rows)}

Topology notes: {len(summary["topology_warn"])} warnings.
{chr(10).join(f"- {w}" for w in summary["topology_warn"]) or "- none"}

## 4. Place reuse

### Train ({reuse_tr["n_places"]} places)

| Bucket | Count | % |
| --- | ---: | ---: |
| Unique | {reuse_tr["unique"]} | {reuse_tr["unique_pct"]} |
| Within behavioral family | {reuse_tr["within_behavioral"]} | {reuse_tr["within_behavioral_pct"]} |
| Within semantic, different family | {reuse_tr["within_semantic_diff_family"]} | {reuse_tr["within_semantic_diff_family_pct"]} |
| Across semantic groups | {reuse_tr["across_semantic"]} | {reuse_tr["across_semantic_pct"]} |

### Test ({reuse_te["n_places"]} places)

| Bucket | Count | % |
| --- | ---: | ---: |
| Unique | {reuse_te["unique"]} | {reuse_te["unique_pct"]} |
| Within behavioral family | {reuse_te["within_behavioral"]} | {reuse_te["within_behavioral_pct"]} |
| Within semantic, different family | {reuse_te["within_semantic_diff_family"]} | {reuse_te["within_semantic_diff_family_pct"]} |
| Across semantic groups | {reuse_te["across_semantic"]} | {reuse_te["across_semantic_pct"]} |

Allocator tier picks: train `{summary["tier_pick_totals"]["train"]}`,
test `{summary["tier_pick_totals"]["test"]}`.

## Reproduce

```bash
python -m traffic_bench.scene_collection analysis assign
```
"""
    path = out_dir / "README.md"
    path.write_text(md, encoding="utf-8")
    return path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allocations", type=Path, default=SIGN_ALLOCATIONS)
    ap.add_argument("--signs-yaml", type=Path, default=SIGNS_YAML)
    ap.add_argument("--index", type=Path, default=JUNCTIONS_INDEX)
    ap.add_argument("--segments-index", type=Path, default=SEGMENTS_INDEX)
    ap.add_argument("--dual-root", type=Path, default=DUAL_PATH_CROPS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.allocations.is_file():
        raise SystemExit(f"ERROR: missing {args.allocations}")

    alloc = _load_json(args.allocations)
    signs_cfg = _load_signs_yaml(args.signs_yaml)
    summary = build_verification(
        alloc=alloc,
        signs_cfg=signs_cfg,
        scene_to_junc=_load_scene_to_junc(args.index),
        segments_index=_load_segments_index(args.segments_index),
        dual_meta=_load_dual_meta(args.dual_root),
    )
    report = write_report(summary, args.out.expanduser().resolve())
    print(f"[assign-verify] report → {report}")
    print(f"[assign-verify] overall={'PASS' if summary['all_passed'] else 'FAIL'}")
    for name, ok in summary["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
