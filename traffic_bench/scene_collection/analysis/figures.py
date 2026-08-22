"""Figures for the harvest inventory (written to analysis/figures/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread

from traffic_bench.scene_collection.analysis.inventory import (
    DUAL_PATH_SHAPES,
    JUNCTION_SHAPES,
    HarvestSnapshot,
    scene_example_dirs,
)
from traffic_bench.scene_collection.collect.dual_path.roles import SLOTS
from traffic_bench.scene_collection.preview import parse_sumo_net, render_network

INK = "#2b2f33"
SLATE = "#5c6670"
MIST = "#c8ced4"
PAPER = "#f7f6f3"
STEEL = "#5b7c99"
TAUPE = "#8d7f6c"
SAGE = "#6d7b72"
INDEX = "#b7bec6"
HARVEST = "#5b7c99"

SHAPE_COLOR = {"T": STEEL, "X": TAUPE, "O": SAGE, "?": SLATE}
TYPE_COLOR = {"straight": STEEL, "curved": TAUPE, "?": SLATE}

_BLUES = LinearSegmentedColormap.from_list(
    "harvest_seq",
    ["#f4f6f8", "#d5dde4", "#9aafc0", "#5b7c99", "#3d5366"],
)

_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.titleweight": "normal",
    "axes.labelcolor": INK,
    "axes.edgecolor": "#8a9098",
    "axes.linewidth": 0.6,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "savefig.facecolor": "white",
}


def _style() -> None:
    plt.rcParams.update(_RCPARAMS)


def _grid(ax: plt.Axes) -> None:
    ax.yaxis.grid(True, color=MIST, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _title(ax: plt.Axes, letter: str, text: str) -> None:
    ax.set_title(f"{letter}   {text}", loc="left", pad=6)


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, pdf: bool = False) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    suffixes = (".png", ".pdf") if pdf else (".png",)
    for suffix in suffixes:
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def inventory_bars(snap: HarvestSnapshot, out_dir: Path, *, pdf: bool = False) -> List[Path]:
    _style()
    names = ("junction", "dual_path", "segment")
    labels = ("Junction", "Dual-path", "Segment")
    catalog = [snap.families()[n].index for n in names]
    disk = [snap.families()[n].on_disk for n in names]
    x = np.arange(len(names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - width / 2, catalog, width, label="Catalog (P)", color=INDEX, zorder=2)
    ax.bar(x + width / 2, disk, width, label="Cropped (H)", color=HARVEST, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Number of maps")
    _title(ax, "a", "Map catalog versus cropped nets")
    _grid(ax)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(catalog + disk) * 1.12)
    return _save(fig, out_dir, "inventory", pdf=pdf)


def junction_shapes(snap: HarvestSnapshot, out_dir: Path, *, pdf: bool = False) -> List[Path]:
    _style()
    idx = snap.junction_index_by_shape()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.85))
    shapes = list(JUNCTION_SHAPES)
    counts = [idx.get(s, 0) for s in shapes]

    axes[0].bar(shapes, counts, color=HARVEST, width=0.55, zorder=2)
    axes[0].set_ylabel("Number of maps")
    axes[0].set_xlabel("Shape")
    _title(axes[0], "a", "Junction topology")
    _grid(axes[0])

    train = [len(snap.train_ids.get(s, [])) for s in shapes]
    test = [len(snap.test_ids.get(s, [])) for s in shapes]
    x = np.arange(len(shapes))
    axes[1].bar(x, train, color=STEEL, width=0.55, label="Train", zorder=2)
    axes[1].bar(x, test, bottom=train, color=TAUPE, width=0.55, label="Test", zorder=2)
    axes[1].set_xticks(x, shapes)
    axes[1].set_xlabel("Shape")
    axes[1].legend(loc="upper right")
    _title(axes[1], "b", "Place-identity split")
    _grid(axes[1])
    fig.tight_layout(w_pad=2.0)
    return _save(fig, out_dir, "junction_shapes", pdf=pdf)


def dual_path_slots(snap: HarvestSnapshot, out_dir: Path, *, pdf: bool = False) -> List[Path]:
    _style()
    disk = snap.dual_path_on_disk
    matrix = np.array(
        [[disk.get((shape, slot), 0) for slot in SLOTS] for shape in DUAL_PATH_SHAPES],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85), gridspec_kw={"width_ratios": [1.2, 1]})

    im = axes[0].imshow(matrix, cmap=_BLUES, aspect="auto")
    axes[0].set_xticks(range(len(SLOTS)), list(SLOTS))
    axes[0].set_yticks(range(len(DUAL_PATH_SHAPES)), list(DUAL_PATH_SHAPES))
    axes[0].tick_params(length=0)
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_color("#8a9098")
        spine.set_linewidth(0.5)
    vmax = float(matrix.max()) if matrix.size else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            shade = "#f7f6f3" if val > 0.55 * vmax else INK
            axes[0].text(j, i, f"{val}", ha="center", va="center", fontsize=7, color=shade)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2, width=0.5, labelsize=7)
    _title(axes[0], "a", "Atoms by shape and slot")

    gains = [float(r["gain_m"]) for r in snap.dual_path_rows if r.get("gain_m") is not None]
    axes[1].hist(gains, bins=32, color=STEEL, edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("Long − short path (m)")
    axes[1].set_ylabel("Atoms")
    _title(axes[1], "b", "Detour gain")
    _grid(axes[1])
    fig.tight_layout(w_pad=2.2)
    return _save(fig, out_dir, "dual_path", pdf=pdf)


def segment_diversity(snap: HarvestSnapshot, out_dir: Path, *, pdf: bool = False) -> List[Path]:
    _style()
    rows = snap.segment_rows
    lengths = [float(r["length_m"]) for r in rows if r.get("length_m") is not None]
    straight = [float(r["straightness"]) for r in rows if r.get("straightness") is not None]
    lanes = [int(r.get("lane_count") or 0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7))
    axes[0].hist(lengths, bins=32, color=STEEL, edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Length (m)")
    axes[0].set_ylabel("Segments")
    _title(axes[0], "a", "Corridor length")
    _grid(axes[0])

    axes[1].hist(straight, bins=32, color=STEEL, edgecolor="white", linewidth=0.4)
    axes[1].axvline(0.99, color=SLATE, lw=0.8, label="Straight ≥ 0.99")
    axes[1].axvline(0.97, color=TAUPE, lw=0.8, ls="--", label="Curved ≥ 0.97")
    axes[1].set_xlabel("Chord / arc")
    axes[1].legend(loc="upper left")
    _title(axes[1], "b", "Straightness")
    _grid(axes[1])

    lane_vals = sorted(set(lanes))
    lane_counts = [lanes.count(v) for v in lane_vals]
    axes[2].bar([str(v) for v in lane_vals], lane_counts, color=STEEL, width=0.6, zorder=2)
    axes[2].set_xlabel("Lanes")
    _title(axes[2], "c", "Vehicle lanes")
    _grid(axes[2])
    fig.tight_layout(w_pad=1.6)
    return _save(fig, out_dir, "segment_diversity", pdf=pdf)


def geo_coverage(snap: HarvestSnapshot, out_dir: Path, *, pdf: bool = False) -> List[Path]:
    _style()
    panels: Sequence[Tuple[str, List[Tuple[float, float, str]], Dict[str, str], str]] = (
        ("Junctions", snap.junction_geo(), SHAPE_COLOR, "a"),
        ("Dual-path junctions", snap.dual_path_geo(), SHAPE_COLOR, "b"),
        ("Segments", snap.segment_geo(), TYPE_COLOR, "c"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.15), sharex=True, sharey=True)
    for ax, (title, points, palette, letter) in zip(axes, panels):
        ax.set_facecolor(PAPER)
        if not points:
            _title(ax, letter, f"{title} (empty)")
            continue
        by_tag: Dict[str, List[Tuple[float, float]]] = {}
        for lat, lon, tag in points:
            by_tag.setdefault(tag, []).append((lon, lat))
        for tag, xy in sorted(by_tag.items()):
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            ax.scatter(
                xs,
                ys,
                s=2.4,
                alpha=0.28,
                c=palette.get(tag, SLATE),
                linewidths=0,
                rasterized=True,
                label=f"{tag}  {len(xy)}",
            )
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(markerscale=4, loc="lower left", handletextpad=0.3, borderaxespad=0.2)
        ax.tick_params(length=2)
        _title(ax, letter, title)
    axes[0].set_ylabel("Latitude")
    fig.tight_layout(w_pad=1.4)
    return _save(fig, out_dir, "geo_coverage", pdf=pdf)


def _render_missing_preview(scene_dir: Path, out_png: Path) -> bool:
    net = scene_dir / "map.net.xml"
    if not net.is_file():
        return False
    meta: dict = {}
    meta_path = scene_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    edges, junctions = parse_sumo_net(net)
    marker = None
    road = meta.get("road_id")
    center = meta.get("center_xy")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        marker = (float(center[0]), float(center[1]))
    render_network(
        edges,
        junctions,
        out_png,
        figsize=(4, 4),
        dpi=120,
        marker_xy=marker,
        compliant_edge_ids=[str(road)] if road else None,
        legend=False,
    )
    return out_png.is_file()


def examples_grid(
    snap: HarvestSnapshot, out_dir: Path, *, n_per_group: int = 2, pdf: bool = False
) -> List[Path]:
    _style()
    groups = scene_example_dirs(snap, n_per_group=n_per_group)
    labels = [k for k, v in groups.items() if v]
    if not labels:
        return []

    n_cols = max(len(groups[k]) for k in labels)
    fig, axes = plt.subplots(
        len(labels),
        n_cols,
        figsize=(2.35 * n_cols, 2.05 * len(labels)),
        facecolor="white",
    )
    if len(labels) == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for r, label in enumerate(labels):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(MIST)
                spine.set_linewidth(0.5)
            if c >= len(groups[label]):
                ax.set_axis_off()
                continue
            scene_dir = groups[label][c]
            png = scene_dir / "custom_cropped.png"
            if not png.is_file():
                tmp = out_dir / "_examples" / f"{scene_dir.name}.png"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                if _render_missing_preview(scene_dir, tmp):
                    png = tmp
            if png.is_file():
                ax.imshow(imread(png))
            ax.set_title(f"{label}  ·  {scene_dir.name}", fontsize=6, color=SLATE, pad=3)

    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    return _save(fig, out_dir, "examples", pdf=pdf)


def write_all(snap: HarvestSnapshot, out_dir: Path, *, pdf: bool = False) -> List[Path]:
    written: List[Path] = []
    for fn in (
        inventory_bars,
        junction_shapes,
        dual_path_slots,
        segment_diversity,
        geo_coverage,
    ):
        written.extend(fn(snap, out_dir, pdf=pdf))
    written.extend(examples_grid(snap, out_dir, pdf=pdf))
    return written
