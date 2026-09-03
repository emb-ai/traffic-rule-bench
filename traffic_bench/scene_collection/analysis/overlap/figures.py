"""Matplotlib figures for cross-sign place overlap.

Artifacts go under ``overlap/figures/`` (code stays at package root).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from traffic_bench.scene_collection.analysis.overlap.catalog import (
    OverlapCatalog,
    degree_histogram,
    family_place_sets,
    pairwise_intersection_matrix,
    places_for_split,
    reuse_bucket_counts,
    semantic_place_sets,
    shared_places,
    top_overlapping_pairs,
    train_test_leakage_across_signs,
    train_test_leakage_within_sign,
    unique_vs_shared_per_sign,
)

INK = "#2b2f33"
SLATE = "#5c6670"
MIST = "#c8ced4"
STEEL = "#5b7c99"
TAUPE = "#8d7f6c"
ALERT = "#a65d57"
UNIQUE = "#5b7c99"
SHARED = "#a65d57"

_SEQ = LinearSegmentedColormap.from_list(
    "overlap_seq",
    ["#f4f6f8", "#d5dde4", "#9aafc0", "#5b7c99", "#3d5366"],
)
_ALERT_SEQ = LinearSegmentedColormap.from_list(
    "overlap_alert",
    ["#f7f6f3", "#e8d5d3", "#c98b86", "#a65d57", "#6e3834"],
)

_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#8a9098",
    "axes.linewidth": 0.6,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
}


def _short(sign: str) -> str:
    return sign.replace("direction_", "dir_").replace("detour_", "det_").replace("_", "\n")


def _save(fig: plt.Figure, path: Path, pdf: bool) -> List[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    out = [path]
    if pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path)
        out.append(pdf_path)
    plt.close(fig)
    return out


def _heatmap(
    labels: Sequence[str],
    matrix: Sequence[Sequence[float]],
    *,
    title: str,
    cbar_label: str,
    cmap,
    annotate: bool = True,
    fmt: str = "d",
    vmax: Optional[float] = None,
) -> plt.Figure:
    n = len(labels)
    data = np.asarray(matrix, dtype=float)
    fig_w = max(8.0, 0.38 * n + 3.5)
    fig_h = max(7.0, 0.38 * n + 2.8)
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax if vmax is not None else (float(np.nanmax(data)) or 1.0),
            aspect="equal",
        )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([_short(s) for s in labels], rotation=90)
        ax.set_yticklabels([_short(s) for s in labels])
        ax.set_title(title, color=INK, pad=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, color=SLATE)
        if annotate and n <= 30:
            thresh = (vmax if vmax is not None else float(np.nanmax(data) or 1.0)) * 0.55
            for i in range(n):
                for j in range(n):
                    val = data[i, j]
                    if val <= 0:
                        continue
                    text = f"{int(round(val))}" if fmt == "d" else f"{val:.2f}"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="white" if val >= thresh else INK,
                        fontsize=5.5,
                    )
        fig.tight_layout()
    return fig


def fig_pairwise_counts(
    place_sets: Dict[str, set],
    *,
    title: str,
    zero_diag: bool = True,
) -> plt.Figure:
    labels, mat = pairwise_intersection_matrix(place_sets)
    data = [row[:] for row in mat]
    if zero_diag:
        for i in range(len(labels)):
            data[i][i] = 0
    return _heatmap(
        labels,
        data,
        title=title,
        cbar_label="# shared places",
        cmap=_SEQ,
        fmt="d",
    )


def fig_degree_hist(place_sets: Dict[str, set], *, title: str) -> plt.Figure:
    hist = degree_histogram(place_sets)
    xs = sorted(hist.keys())
    ys = [hist[x] for x in xs]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        colors = [STEEL if x == 1 else (TAUPE if x == 2 else ALERT) for x in xs]
        ax.bar(xs, ys, color=colors, width=0.7, edgecolor="white", linewidth=0.4)
        for x, y in zip(xs, ys):
            ax.text(x, y + max(ys) * 0.01, str(y), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xlabel("# signs sharing the same place")
        ax.set_ylabel("# places")
        ax.set_title(title, color=INK)
        ax.set_xticks(xs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_unique_vs_shared(place_sets: Dict[str, set], *, title: str) -> plt.Figure:
    stats = unique_vs_shared_per_sign(place_sets)
    signs = sorted(stats.keys(), key=lambda s: (-stats[s]["shared"], s))
    unique = [stats[s]["unique"] for s in signs]
    shared = [stats[s]["shared"] for s in signs]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(max(9.0, 0.45 * len(signs) + 2), 5.2))
        x = np.arange(len(signs))
        ax.bar(x, unique, color=UNIQUE, label="unique to this sign", width=0.75)
        ax.bar(x, shared, bottom=unique, color=SHARED, label="shared with ≥1 other sign", width=0.75)
        ymax = max((u + sh) for u, sh in zip(unique, shared)) if signs else 1
        for i, s in enumerate(signs):
            tot = stats[s]["total"]
            pct = stats[s]["shared_pct"]
            ax.text(
                i,
                tot + ymax * 0.01,
                f"{pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=ALERT if pct >= 20 else SLATE,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_short(s) for s in signs], rotation=90)
        ax.set_ylabel("# places")
        ax.set_title(title, color=INK)
        ax.legend(frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_pool_sizes(cat: OverlapCatalog) -> plt.Figure:
    signs = cat.signs
    train = [len(cat.places[s]["train"]) for s in signs]
    test = [len(cat.places[s]["test"]) for s in signs]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(max(9.0, 0.45 * len(signs) + 2), 5.0))
        x = np.arange(len(signs))
        w = 0.4
        ax.bar(x - w / 2, train, width=w, color=STEEL, label="train places")
        ax.bar(x + w / 2, test, width=w, color=TAUPE, label="test places")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(s) for s in signs], rotation=90)
        ax.set_ylabel("# places")
        ax.set_title("Place counts per sign (train / test)", color=INK)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_train_vs_test_cross(cat: OverlapCatalog) -> plt.Figure:
    labels, mat = train_test_leakage_across_signs(cat)
    return _heatmap(
        labels,
        mat,
        title="Train(row) ∩ Test(col) place leakage across signs",
        cbar_label="# places",
        cmap=_ALERT_SEQ,
        fmt="d",
    )


def fig_within_sign_leak(cat: OverlapCatalog) -> plt.Figure:
    leaks = train_test_leakage_within_sign(cat)
    signs = cat.signs
    vals = [len(leaks.get(s, ())) for s in signs]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(max(9.0, 0.45 * len(signs) + 2), 4.2))
        colors = [ALERT if v else MIST for v in vals]
        ax.bar(range(len(signs)), vals, color=colors, width=0.75)
        ax.set_xticks(range(len(signs)))
        ax.set_xticklabels([_short(s) for s in signs], rotation=90)
        ax.set_ylabel("# places in train ∩ test")
        ax.set_title("Within-sign train↔test leakage", color=INK)
        total = sum(vals)
        ax.text(
            0.99,
            0.95,
            f"total leaked places: {total}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=ALERT if total else SLATE,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_top_shared(place_sets: Dict[str, set], *, top_n: int = 25) -> plt.Figure:
    rows = shared_places(place_sets, min_signs=2)[:top_n]
    if not rows:
        with plt.rc_context(_RCPARAMS):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            ax.text(0.5, 0.5, "No places shared by ≥2 signs", ha="center", va="center", color=SLATE)
            return fig
    labels = [f"{pid}  ({', '.join(signs)})" for pid, signs in rows]
    vals = [len(signs) for _, signs in rows]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.32 * len(rows) + 1.5)))
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals[::-1], color=TAUPE, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels[::-1], fontsize=6.5)
        ax.set_xlabel("# signs")
        ax.set_title(f"Top {len(rows)} most-reused places (train)", color=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_global_split(cat: OverlapCatalog) -> plt.Figure:
    train: set = set()
    test: set = set()
    for s in cat.signs:
        train |= cat.places[s]["train"]
        test |= cat.places[s]["test"]
    both = train & test
    only_tr = train - test
    only_te = test - train
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        labels = ["train only", "test only", "train ∩ test"]
        vals = [len(only_tr), len(only_te), len(both)]
        colors = [STEEL, TAUPE, ALERT]
        ax.bar(labels, vals, color=colors, width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals + [1]) * 0.01, str(v), ha="center", va="bottom", color=INK)
        ax.set_ylabel("# places (union across signs)")
        ax.set_title("Global place coverage by split", color=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_top_pairs(place_sets: Dict[str, set], *, title: str, top_n: int = 20) -> plt.Figure:
    pairs = top_overlapping_pairs(place_sets, top_n=top_n)
    with plt.rc_context(_RCPARAMS):
        if not pairs:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            ax.text(0.5, 0.5, "No overlapping sign pairs", ha="center", va="center", color=SLATE)
            return fig
        labels = [f"{a} ∩ {b}" for _, a, b in pairs]
        vals = [n for n, _, _ in pairs]
        fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.34 * len(pairs) + 1.5)))
        y = np.arange(len(pairs))[::-1]
        ax.barh(y, vals[::-1], color=STEEL, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels[::-1], fontsize=7)
        for yi, v in zip(y, vals[::-1]):
            ax.text(v + max(vals) * 0.01, yi, str(v), va="center", fontsize=7.5, color=INK)
        ax.set_xlabel("# shared places")
        ax.set_title(title, color=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_family_unique_vs_shared(place_sets: Dict[str, set], *, title: str) -> plt.Figure:
    fam_sets = family_place_sets(place_sets)
    # Recompute unique/shared at family granularity via sign-level degree mapped up.
    # Unique = place appears under exactly one family; shared = ≥2 families.
    inv: Dict[str, set] = defaultdict(set)
    for fam, places in fam_sets.items():
        for p in places:
            inv[p].add(fam)
    families = sorted(fam_sets.keys())
    unique = []
    shared = []
    for fam in families:
        places = fam_sets[fam]
        u = sum(1 for p in places if len(inv[p]) == 1)
        s = sum(1 for p in places if len(inv[p]) >= 2)
        unique.append(u)
        shared.append(s)
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(families))
        ax.bar(x, unique, color=UNIQUE, label="unique to family", width=0.7)
        ax.bar(x, shared, bottom=unique, color=SHARED, label="shared across families", width=0.7)
        ymax = max((u + s) for u, s in zip(unique, shared)) if families else 1
        for i, fam in enumerate(families):
            tot = unique[i] + shared[i]
            pct = 100.0 * shared[i] / tot if tot else 0.0
            ax.text(i, tot + ymax * 0.01, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=25, ha="right")
        ax.set_ylabel("# places")
        ax.set_title(title, color=INK)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_family_pairwise(place_sets: Dict[str, set], *, title: str) -> plt.Figure:
    fam_sets = family_place_sets(place_sets)
    return fig_pairwise_counts(fam_sets, title=title, zero_diag=True)


def fig_semantic_pairwise(place_sets: Dict[str, set], *, title: str) -> plt.Figure:
    sem_sets = semantic_place_sets(place_sets)
    return fig_pairwise_counts(sem_sets, title=title, zero_diag=True)


def fig_reuse_buckets(place_sets: Dict[str, set], *, title: str) -> plt.Figure:
    counts = reuse_bucket_counts(place_sets)
    labels = [
        "unique",
        "within\nbehavioral",
        "within semantic\n≠ family",
        "across\nsemantic",
    ]
    keys = [
        "unique",
        "within_behavioral",
        "within_semantic_diff_family",
        "across_semantic",
    ]
    vals = [counts[k] for k in keys]
    colors = [STEEL, TAUPE, SHARED, ALERT]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors, width=0.7)
        ymax = max(vals + [1])
        for i, v in enumerate(vals):
            ax.text(i, v + ymax * 0.01, str(v), ha="center", va="bottom", color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("# places")
        ax.set_title(title, color=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_place_type_by_split(cat: OverlapCatalog) -> plt.Figure:
    counts = {
        "train": {"junction": 0, "way": 0, "scene": 0, "other": 0},
        "test": {"junction": 0, "way": 0, "scene": 0, "other": 0},
    }
    seen = {"train": set(), "test": set()}
    for rec in cat.records:
        if rec.split not in counts:
            continue
        if rec.place_id in seen[rec.split]:
            continue
        seen[rec.split].add(rec.place_id)
        prefix = rec.place_id.split(":", 1)[0]
        bucket = prefix if prefix in counts[rec.split] else "other"
        counts[rec.split][bucket] += 1

    kinds = ["junction", "way", "scene", "other"]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        x = np.arange(len(kinds))
        w = 0.35
        tr = [counts["train"][k] for k in kinds]
        te = [counts["test"][k] for k in kinds]
        ax.bar(x - w / 2, tr, width=w, color=STEEL, label="train")
        ax.bar(x + w / 2, te, width=w, color=TAUPE, label="test")
        for i, (a, b) in enumerate(zip(tr, te)):
            if a:
                ax.text(i - w / 2, a + 1, str(a), ha="center", va="bottom", fontsize=7.5)
            if b:
                ax.text(i + w / 2, b + 1, str(b), ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(kinds)
        ax.set_ylabel("# unique places")
        ax.set_title("Place identity type by split (union across signs)", color=INK)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def fig_scenes_vs_places(cat: OverlapCatalog) -> plt.Figure:
    """Show that multiple scenes can share one place (augmentation ≠ new map)."""
    signs = cat.signs
    train_scenes = [len(cat.scenes[s]["train"]) for s in signs]
    train_places = [len(cat.places[s]["train"]) for s in signs]
    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=(max(9.0, 0.45 * len(signs) + 2), 5.0))
        x = np.arange(len(signs))
        w = 0.4
        ax.bar(x - w / 2, train_scenes, width=w, color=STEEL, label="train scenes")
        ax.bar(x + w / 2, train_places, width=w, color=TAUPE, label="train places")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(s) for s in signs], rotation=90)
        ax.set_ylabel("count")
        ax.set_title("Train: scenes vs distinct places (place collapses map reuse)", color=INK)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig


def write_all(
    cat: OverlapCatalog,
    out_dir: Path,
    *,
    pdf: bool = False,
) -> List[Path]:
    """Render all overlap figures into ``out_dir`` (typically ``overlap/figures``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    train_sets = places_for_split(cat, "train")
    test_sets = places_for_split(cat, "test")

    jobs = [
        ("train_pairwise_intersection.png", lambda: fig_pairwise_counts(train_sets, title="Train∩train shared places (off-diagonal)")),
        ("test_pairwise_intersection.png", lambda: fig_pairwise_counts(test_sets, title="Test∩test shared places (off-diagonal)")),
        ("train_top_sign_pairs.png", lambda: fig_top_pairs(train_sets, title="Top train sign pairs by shared places")),
        ("test_top_sign_pairs.png", lambda: fig_top_pairs(test_sets, title="Top test sign pairs by shared places")),
        ("train_place_degree.png", lambda: fig_degree_hist(train_sets, title="How many train signs reuse the same place?")),
        ("test_place_degree.png", lambda: fig_degree_hist(test_sets, title="How many test signs reuse the same place?")),
        ("train_unique_vs_shared.png", lambda: fig_unique_vs_shared(train_sets, title="Per sign: unique vs shared train places")),
        ("test_unique_vs_shared.png", lambda: fig_unique_vs_shared(test_sets, title="Per sign: unique vs shared test places")),
        ("train_reuse_buckets.png", lambda: fig_reuse_buckets(train_sets, title="Train place reuse buckets (policy taxonomy)")),
        ("test_reuse_buckets.png", lambda: fig_reuse_buckets(test_sets, title="Test place reuse buckets (policy taxonomy)")),
        ("train_behavioral_pairwise.png", lambda: fig_family_pairwise(train_sets, title="Train: shared places between behavioral families")),
        ("test_behavioral_pairwise.png", lambda: fig_family_pairwise(test_sets, title="Test: shared places between behavioral families")),
        ("train_semantic_pairwise.png", lambda: fig_semantic_pairwise(train_sets, title="Train: shared places between semantic groups")),
        ("test_semantic_pairwise.png", lambda: fig_semantic_pairwise(test_sets, title="Test: shared places between semantic groups")),
        ("train_behavioral_unique_vs_shared.png", lambda: fig_family_unique_vs_shared(train_sets, title="Train: unique vs shared places by behavioral family")),
        ("per_sign_pool_sizes.png", lambda: fig_pool_sizes(cat)),
        ("train_scenes_vs_places.png", lambda: fig_scenes_vs_places(cat)),
        ("train_vs_test_cross_sign.png", lambda: fig_train_vs_test_cross(cat)),
        ("within_sign_train_test_leak.png", lambda: fig_within_sign_leak(cat)),
        ("top_shared_train_places.png", lambda: fig_top_shared(train_sets)),
        ("global_train_test_places.png", lambda: fig_global_split(cat)),
    ]

    for name, make_fig in jobs:
        written.extend(_save(make_fig(), out_dir / name, pdf=pdf))
    return written
