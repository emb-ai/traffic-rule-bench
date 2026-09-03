"""Markdown README + JSON summary for cross-sign place overlap."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from traffic_bench.scene_collection.analysis.overlap.catalog import (
    SIGN_FAMILY,
    SIGN_SEMANTIC,
    OverlapCatalog,
    degree_histogram,
    family_place_sets,
    global_train_test_place_overlap,
    mean_offdiag,
    pairwise_intersection_matrix,
    per_sign_counts,
    places_for_split,
    reuse_bucket_counts,
    semantic_place_sets,
    shared_places,
    top_overlapping_pairs,
    train_test_leakage_across_signs,
    train_test_leakage_within_sign,
    unique_vs_shared_per_sign,
)


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def _verdict(
    *,
    train_shared: int,
    train_total: int,
    global_tt: int,
    within_leaks: int,
    train_union: int,
    test_union: int,
) -> Tuple[str, List[str]]:
    pct = _pct(train_shared, train_total)
    bullets: List[str] = []
    if pct < 5:
        level = "small"
    elif pct < 20:
        level = "moderate"
    else:
        level = "large"
    bullets.append(
        f"Train cross-sign place reuse is **{level}**: "
        f"{train_shared}/{train_total} places ({pct:.1f}%) appear under ≥2 signs."
    )
    leak_pct = _pct(global_tt, train_union + test_union - global_tt) if (train_union or test_union) else 0.0
    # clearer: fraction of train places that also appear in test
    train_in_test_pct = _pct(global_tt, train_union)
    if global_tt == 0 and within_leaks == 0:
        bullets.append(
            "Train↔test leakage is **clean**: no place appears in both splits "
            "(neither within a sign nor globally)."
        )
    else:
        bullets.append(
            f"Train↔test leakage is **present**: global train∩test = {global_tt} places "
            f"({train_in_test_pct:.1f}% of the train union); "
            f"within-sign leaked place-instances = {within_leaks}."
        )
    bullets.append(
        f"Map inventory size: train union **{train_union}** places, "
        f"test union **{test_union}** places across all signs."
    )
    headline = (
        f"Cross-sign reuse {level} ({pct:.1f}% of train places); "
        f"global train∩test = {global_tt} ({train_in_test_pct:.1f}% of train union)."
    )
    return headline, bullets


def _family_unique_shared(place_sets: Dict[str, set]) -> Dict[str, Dict[str, float]]:
    fam_sets = family_place_sets(place_sets)
    inv: Dict[str, set] = defaultdict(set)
    for fam, places in fam_sets.items():
        for p in places:
            inv[p].add(fam)
    out: Dict[str, Dict[str, float]] = {}
    for fam, places in fam_sets.items():
        unique = sum(1 for p in places if len(inv[p]) == 1)
        shared = sum(1 for p in places if len(inv[p]) >= 2)
        total = len(places)
        out[fam] = {
            "unique": unique,
            "shared": shared,
            "total": total,
            "shared_pct": _pct(shared, total),
        }
    return out


def _place_type_counts(cat: OverlapCatalog) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        "train": defaultdict(int),
        "test": defaultdict(int),
    }
    seen = {"train": set(), "test": set()}
    for rec in cat.records:
        if rec.split not in counts:
            continue
        if rec.place_id in seen[rec.split]:
            continue
        seen[rec.split].add(rec.place_id)
        prefix = rec.place_id.split(":", 1)[0]
        counts[rec.split][prefix] += 1
    return {k: dict(v) for k, v in counts.items()}


def build_summary(cat: OverlapCatalog) -> Dict[str, Any]:
    train_sets = places_for_split(cat, "train")
    test_sets = places_for_split(cat, "test")
    train_shared = shared_places(train_sets, min_signs=2)
    test_shared = shared_places(test_sets, min_signs=2)
    train_deg = degree_histogram(train_sets)
    test_deg = degree_histogram(test_sets)
    within = train_test_leakage_within_sign(cat)
    labels_tt, mat_tt = train_test_leakage_across_signs(cat)
    labels_tr, mat_tr = pairwise_intersection_matrix(train_sets)
    labels_te, mat_te = pairwise_intersection_matrix(test_sets)
    fam_tr_labels, fam_tr_mat = pairwise_intersection_matrix(family_place_sets(train_sets))
    fam_te_labels, fam_te_mat = pairwise_intersection_matrix(family_place_sets(test_sets))
    sem_tr_labels, sem_tr_mat = pairwise_intersection_matrix(semantic_place_sets(train_sets))
    sem_te_labels, sem_te_mat = pairwise_intersection_matrix(semantic_place_sets(test_sets))
    global_tt = sorted(global_train_test_place_overlap(cat))
    train_union = set().union(*train_sets.values()) if train_sets else set()
    test_union = set().union(*test_sets.values()) if test_sets else set()
    train_pairs = top_overlapping_pairs(train_sets, top_n=25)
    test_pairs = top_overlapping_pairs(test_sets, top_n=25)

    same_sign_tt = sum(mat_tt[i][i] for i in range(len(labels_tt)))
    cross_sign_tt = int(sum(sum(row) for row in mat_tt) - same_sign_tt)

    return {
        "n_signs": len(cat.signs),
        "n_records": len(cat.records),
        "signs": cat.signs,
        "per_sign": per_sign_counts(cat),
        "train_unique_vs_shared": unique_vs_shared_per_sign(train_sets),
        "test_unique_vs_shared": unique_vs_shared_per_sign(test_sets),
        "train_family_unique_vs_shared": _family_unique_shared(train_sets),
        "test_family_unique_vs_shared": _family_unique_shared(test_sets),
        "train_reuse_buckets": reuse_bucket_counts(train_sets),
        "test_reuse_buckets": reuse_bucket_counts(test_sets),
        "train_family_place_labels": fam_tr_labels,
        "train_family_place_matrix": fam_tr_mat,
        "test_family_place_labels": fam_te_labels,
        "test_family_place_matrix": fam_te_mat,
        "train_semantic_place_labels": sem_tr_labels,
        "train_semantic_place_matrix": sem_tr_mat,
        "test_semantic_place_labels": sem_te_labels,
        "test_semantic_place_matrix": sem_te_mat,
        "train_place_degree_hist": train_deg,
        "test_place_degree_hist": test_deg,
        "train_places_shared_by_ge2_signs": len(train_shared),
        "test_places_shared_by_ge2_signs": len(test_shared),
        "train_place_union": len(train_union),
        "test_place_union": len(test_union),
        "train_shared_sample": [
            {"place_id": pid, "signs": list(signs)} for pid, signs in train_shared[:40]
        ],
        "test_shared_sample": [
            {"place_id": pid, "signs": list(signs)} for pid, signs in test_shared[:40]
        ],
        "train_top_pairs": [
            {"n_shared": n, "sign_a": a, "sign_b": b} for n, a, b in train_pairs
        ],
        "test_top_pairs": [
            {"n_shared": n, "sign_a": a, "sign_b": b} for n, a, b in test_pairs
        ],
        "within_sign_train_test_leak": {
            sign: sorted(places) for sign, places in within.items()
        },
        "within_sign_train_test_leak_count": sum(len(v) for v in within.values()),
        "train_vs_test_cross_sign_labels": labels_tt,
        "train_vs_test_cross_sign_matrix": mat_tt,
        "train_vs_test_same_sign_sum": same_sign_tt,
        "train_vs_test_cross_sign_sum": cross_sign_tt,
        "train_pairwise_labels": labels_tr,
        "train_pairwise_matrix": mat_tr,
        "train_pairwise_mean_offdiag": mean_offdiag(mat_tr),
        "test_pairwise_labels": labels_te,
        "test_pairwise_matrix": mat_te,
        "test_pairwise_mean_offdiag": mean_offdiag(mat_te),
        "global_train_test_place_overlap_count": len(global_tt),
        "global_train_test_place_overlap_sample": global_tt[:60],
        "place_type_counts": _place_type_counts(cat),
        "sign_family": {s: SIGN_FAMILY.get(s, "other") for s in cat.signs},
        "sign_semantic": {s: SIGN_SEMANTIC.get(s, "other") for s in cat.signs},
    }


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _family_offdiag_rows(labels: List[str], mat: List[List[int]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j:
                continue
            n = int(mat[i][j])
            if n:
                rows.append([f"`{a}`", f"`{b}`", str(n)])
    rows.sort(key=lambda r: -int(r[2]))
    return rows or [["—", "—", "0"]]


def write_report(cat: OverlapCatalog, out_dir: Path, *, figures_rel: str = "figures") -> Path:
    """Write ``summary.json`` + reviewer-facing ``README.md`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(cat)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_shared_n = int(summary["train_places_shared_by_ge2_signs"])
    train_union = int(summary["train_place_union"])
    test_union = int(summary["test_place_union"])
    global_tt = int(summary["global_train_test_place_overlap_count"])
    within_n = int(summary["within_sign_train_test_leak_count"])
    headline, verdict_bullets = _verdict(
        train_shared=train_shared_n,
        train_total=train_union,
        global_tt=global_tt,
        within_leaks=within_n,
        train_union=train_union,
        test_union=test_union,
    )

    hist_rows = [
        [str(k), str(v)]
        for k, v in (summary["train_place_degree_hist"] or {}).items()
    ]
    test_hist_rows = [
        [str(k), str(v)]
        for k, v in (summary["test_place_degree_hist"] or {}).items()
    ]

    per_sign_uvs = summary["train_unique_vs_shared"]
    uvs_rows = [
        [
            f"`{sign}`",
            str(row["family"]),
            str(int(row["unique"])),
            str(int(row["shared"])),
            str(int(row["total"])),
            f"{row['shared_pct']:.1f}%",
        ]
        for sign, row in sorted(
            per_sign_uvs.items(), key=lambda kv: (-kv[1]["shared_pct"], kv[0])
        )
    ]

    test_uvs = summary["test_unique_vs_shared"]
    test_uvs_rows = [
        [
            f"`{sign}`",
            str(int(row["unique"])),
            str(int(row["shared"])),
            str(int(row["total"])),
            f"{row['shared_pct']:.1f}%",
        ]
        for sign, row in sorted(
            test_uvs.items(), key=lambda kv: (-kv[1]["shared_pct"], kv[0])
        )
    ]

    fam_uvs = summary["train_family_unique_vs_shared"]
    fam_rows = [
        [
            f"`{fam}`",
            str(int(row["unique"])),
            str(int(row["shared"])),
            str(int(row["total"])),
            f"{row['shared_pct']:.1f}%",
        ]
        for fam, row in sorted(fam_uvs.items(), key=lambda kv: (-kv[1]["total"], kv[0]))
    ]

    per_sign = summary["per_sign"]
    size_rows = [
        [
            f"`{sign}`",
            str(row["family"]),
            str(row["train_places"]),
            str(row["test_places"]),
            str(row["train_scenes"]),
            str(row["test_scenes"]),
        ]
        for sign, row in sorted(per_sign.items())
    ]

    pair_rows = [
        [str(p["n_shared"]), f"`{p['sign_a']}`", f"`{p['sign_b']}`"]
        for p in summary["train_top_pairs"][:15]
    ]

    within = summary["within_sign_train_test_leak"]
    if within:
        leak_lines = "\n".join(
            f"- `{sign}`: {len(places)} places — "
            + ", ".join(f"`{p}`" for p in places[:8])
            + (" …" if len(places) > 8 else "")
            for sign, places in sorted(within.items())
        )
    else:
        leak_lines = "_None._"

    global_sample = summary["global_train_test_place_overlap_sample"]
    global_sample_md = (
        ", ".join(f"`{p}`" for p in global_sample[:25])
        + (" …" if len(global_sample) > 25 else "")
        if global_sample
        else "_empty_"
    )

    ptypes = summary["place_type_counts"]
    type_rows = []
    all_kinds = sorted(set(ptypes.get("train", {})) | set(ptypes.get("test", {})))
    for k in all_kinds:
        type_rows.append(
            [
                k,
                str(ptypes.get("train", {}).get(k, 0)),
                str(ptypes.get("test", {}).get(k, 0)),
            ]
        )

    same_tt = int(summary["train_vs_test_same_sign_sum"])
    cross_tt = int(summary["train_vs_test_cross_sign_sum"])
    mean_off = float(summary["train_pairwise_mean_offdiag"])

    # Expected-overlap note for direction_control family
    dir_signs = [
        s for s, f in summary["sign_family"].items() if f == "direction_control"
    ]
    dir_shared_pcts = [
        per_sign_uvs[s]["shared_pct"] for s in dir_signs if s in per_sign_uvs
    ]
    dir_avg = sum(dir_shared_pcts) / len(dir_shared_pcts) if dir_shared_pcts else 0.0

    tr_buckets = summary["train_reuse_buckets"]
    te_buckets = summary["test_reuse_buckets"]
    bucket_rows_tr = [
        [
            k,
            str(tr_buckets[k]),
            f"{_pct(tr_buckets[k], train_union):.1f}%",
        ]
        for k in (
            "unique",
            "within_behavioral",
            "within_semantic_diff_family",
            "across_semantic",
        )
    ]
    bucket_rows_te = [
        [
            k,
            str(te_buckets[k]),
            f"{_pct(te_buckets[k], test_union):.1f}%",
        ]
        for k in (
            "unique",
            "within_behavioral",
            "within_semantic_diff_family",
            "across_semantic",
        )
    ]

    fr = figures_rel.rstrip("/")
    verdict_md = "\n".join(f"- {b}" for b in verdict_bullets)

    md = f"""# Map overlap analysis (train / test)

{headline}

Audits **geographic map reuse** under the tiered assign policy
(unique → same behavioral family → same semantic group; no cross-semantic).

Also see [`allocation_verify.md`](allocation_verify.md) for counts / topology checks.

## How a “place” is defined

| Crop family | Place id |
| --- | --- |
| junction / dual_path / roundabout | `junction:<junction_id>` |
| segment (speed, detour, crosswalk, …) | `way:<osm_way_id>` |

Sources: `data/scenes/<sign>/moscow_pool.json`, enriched from `meta.json`.

## Verdict

{verdict_md}

### Interpretation

- **Within behavioral family** reuse (e.g. `direction_control` 4.1.1–4.1.6) is **by design**:
  same place, different ego rule. Avg shared-% in `direction_control` (train): **{dir_avg:.1f}%**.
- **Across semantic groups** should be **0** under the new assign policy.
- **Train↔test** place leak must be **0** (same-sign sum={same_tt}, cross-sign cell sum={cross_tt}).

## Headline numbers

| Metric | Value |
| --- | ---: |
| Signs | {summary["n_signs"]} |
| Pool records | {summary["n_records"]} |
| Train place union | {train_union} |
| Train places shared by ≥2 signs | {train_shared_n} ({_pct(train_shared_n, train_union):.1f}%) |
| Test place union | {test_union} |
| Test places shared by ≥2 signs | {summary["test_places_shared_by_ge2_signs"]} ({_pct(int(summary["test_places_shared_by_ge2_signs"]), test_union):.1f}%) |
| Global train∩test places | {global_tt} |
| Within-sign train∩test places | {within_n} |
| Mean off-diagonal train pairwise | {mean_off:.2f} |

## Reuse buckets (policy taxonomy)

### Train

{_md_table(["Bucket", "# places", "%"], bucket_rows_tr)}

### Test

{_md_table(["Bucket", "# places", "%"], bucket_rows_te)}

## Train place reuse histogram

{_md_table(["# signs sharing place", "# places"], hist_rows)}

### Test

{_md_table(["# signs sharing place", "# places"], test_hist_rows)}

## Per-sign pool sizes

{_md_table(["Sign", "Behavioral family", "Train places", "Test places", "Train scenes", "Test scenes"], size_rows)}

## Per-sign unique vs shared (train)

{_md_table(["Sign", "Behavioral family", "Unique", "Shared", "Total", "Shared %"], uvs_rows)}

## Per-sign unique vs shared (test)

{_md_table(["Sign", "Unique", "Shared", "Total", "Shared %"], test_uvs_rows)}

## Behavioral family roll-up (train)

{_md_table(["Family", "Unique", "Shared across families", "Total", "Shared %"], fam_rows)}

## Behavioral family place overlap (train)

{_md_table(["Family A", "Family B", "# shared places"], _family_offdiag_rows(summary["train_family_place_labels"], summary["train_family_place_matrix"]))}

## Semantic group place overlap (train)

{_md_table(["Group A", "Group B", "# shared places"], _family_offdiag_rows(summary["train_semantic_place_labels"], summary["train_semantic_place_matrix"]))}

## Top overlapping sign pairs (train)

{_md_table(["# shared places", "Sign A", "Sign B"], pair_rows)}

## Train↔test leakage detail

### Within-sign

{leak_lines}

### Global train∩test sample

{global_sample_md}

## Figures

All PNGs under [`{fr}/`]({fr}/).

| File | Meaning |
| --- | --- |
| `{fr}/train_pairwise_intersection.png` | Off-diagonal shared train places |
| `{fr}/test_pairwise_intersection.png` | Off-diagonal shared test places |
| `{fr}/train_top_sign_pairs.png` | Top train sign pairs |
| `{fr}/test_top_sign_pairs.png` | Top test sign pairs |
| `{fr}/train_unique_vs_shared.png` | Per sign unique vs shared (train) |
| `{fr}/test_unique_vs_shared.png` | Per sign unique vs shared (test) |
| `{fr}/train_reuse_buckets.png` | Policy reuse buckets (train) |
| `{fr}/test_reuse_buckets.png` | Policy reuse buckets (test) |
| `{fr}/train_behavioral_pairwise.png` | Shared places between behavioral families |
| `{fr}/test_behavioral_pairwise.png` | Same for test |
| `{fr}/train_semantic_pairwise.png` | Shared places between semantic groups |
| `{fr}/test_semantic_pairwise.png` | Same for test |
| `{fr}/train_behavioral_unique_vs_shared.png` | Behavioral family unique vs shared |
| `{fr}/train_place_degree.png` | Degree histogram (train) |
| `{fr}/test_place_degree.png` | Degree histogram (test) |
| `{fr}/per_sign_pool_sizes.png` | Train/test place counts |
| `{fr}/train_scenes_vs_places.png` | Scenes vs collapsed places |
| `{fr}/train_vs_test_cross_sign.png` | Train(row) ∩ Test(col) |
| `{fr}/within_sign_train_test_leak.png` | Same-sign split leakage |
| `{fr}/top_shared_train_places.png` | Most-reused train places |
| `{fr}/global_train_test_places.png` | Global split coverage |

## Reproduce

```bash
python -m traffic_bench.scene_collection analysis overlap
python -m traffic_bench.scene_collection analysis assign_verify
```
"""
    md_path = out_dir / "README.md"
    md_path.write_text(md, encoding="utf-8")
    return md_path
