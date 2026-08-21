#!/usr/bin/env python3
"""Debug / materialize helper: render a scene SUMO net as a static PNG.

Used by ``build_scenes/materialize_scenes.py`` for review-UI previews
(``custom_cropped.png``). Also runnable ad-hoc:

  python -m tools.render_map <scene> --scenes-dir data/stop/scenes

For dual_path scenes, overlays short baseline (red) and long compliant (green)
routes from ``meta.json`` when present.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

TOOLS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TOOLS_DIR.parent

from traffic_bench.eval.core.sumo.sumo_utils import resolve_net_file, load_scene_meta, resolve_scene_dir

SCENES_DIR_DEFAULT = PACKAGE_DIR / "data" / "scenes" / "yield"
Point = Tuple[float, float]


def parse_sumo_net(net_path: Path):
    """Parse SUMO net.xml and extract edges/lanes for rendering."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(net_path)
    root = tree.getroot()

    edges = []
    junctions = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id.startswith(":"):
            continue

        for lane in edge.findall("lane"):
            shape_str = lane.get("shape")
            if shape_str:
                points = []
                for p in shape_str.strip().split():
                    x, y = map(float, p.split(","))
                    points.append((x, y))
                if len(points) >= 2:
                    edges.append(
                        {
                            "id": edge_id,
                            "lane_id": lane.get("id"),
                            "points": points,
                            "width": float(lane.get("width", 3.2)),
                        }
                    )

    for junction in root.findall("junction"):
        junc_type = junction.get("type")
        if junc_type in ("internal", "dead_end"):
            continue
        x = float(junction.get("x", 0))
        y = float(junction.get("y", 0))
        shape_str = junction.get("shape")
        if shape_str:
            points = []
            for p in shape_str.strip().split():
                coords = p.split(",")
                if len(coords) >= 2:
                    points.append((float(coords[0]), float(coords[1])))
            if points:
                junctions.append(
                    {"id": junction.get("id"), "points": points, "x": x, "y": y}
                )

    return edges, junctions


def _edge_centerline_map(edges: Sequence[dict]) -> Dict[str, List[Point]]:
    """Pick one representative polyline per edge id (prefer middle lane)."""
    by_edge: Dict[str, List[List[Point]]] = {}
    for edge in edges:
        eid = str(edge.get("id") or "")
        pts = edge.get("points") or []
        if not eid or len(pts) < 2:
            continue
        by_edge.setdefault(eid, []).append(list(pts))
    out: Dict[str, List[Point]] = {}
    for eid, variants in by_edge.items():
        out[eid] = variants[len(variants) // 2]
    return out


def polyline_for_edge_ids(
    edges: Sequence[dict],
    edge_ids: Iterable[str],
) -> List[Point]:
    """Concatenate lane shapes for an ordered edge sequence into one polyline."""
    centers = _edge_centerline_map(edges)
    poly: List[Point] = []
    for eid in edge_ids:
        pts = centers.get(str(eid))
        if not pts:
            continue
        if not poly:
            poly.extend(pts)
            continue
        # Avoid duplicating the shared vertex between consecutive edges.
        if abs(poly[-1][0] - pts[0][0]) < 1e-3 and abs(poly[-1][1] - pts[0][1]) < 1e-3:
            poly.extend(pts[1:])
        else:
            poly.extend(pts)
    return poly


def routes_from_dual_path_meta(meta: dict) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str]]:
    """Return (baseline_edges, compliant_edges, spawn_edge) from dual_path meta."""
    dp = meta.get("dual_path") if isinstance(meta, dict) else None
    if not isinstance(dp, dict):
        return None, None, None
    baseline = [str(e) for e in (dp.get("baseline_path") or dp.get("turn_path") or []) if e]
    compliant = [
        str(e) for e in (dp.get("compliant_path") or dp.get("straight_path") or []) if e
    ]
    spawn = str(meta.get("road_id") or "") or None
    # Include spawn approach on both routes for a continuous visual from ego.
    if spawn:
        if baseline and baseline[0] != spawn:
            baseline = [spawn, *baseline]
        if compliant and compliant[0] != spawn:
            compliant = [spawn, *compliant]
    return (baseline or None), (compliant or None), spawn


def render_network(
    edges,
    junctions,
    out_path: Path,
    figsize=(12, 12),
    dpi=150,
    marker_xy: tuple[float, float] | None = None,
    baseline_edge_ids: Optional[Sequence[str]] = None,
    compliant_edge_ids: Optional[Sequence[str]] = None,
    legend: bool = True,
):
    """Render the road network to an image.

    Optional dual-path overlays:
      * baseline (short / forbidden) — red
      * compliant (long / allowed) — green
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#f0f0f0")

    for junc in junctions:
        if len(junc["points"]) >= 3:
            polygon = mpatches.Polygon(
                junc["points"],
                closed=True,
                facecolor="#909090",
                edgecolor="#707070",
                linewidth=0.5,
                alpha=0.7,
            )
            ax.add_patch(polygon)

    lines = []
    for edge in edges:
        pts = edge["points"]
        if len(pts) >= 2:
            lines.append(pts)

    if lines:
        lc = LineCollection(lines, colors="#404040", linewidths=2.0, alpha=0.9)
        ax.add_collection(lc)

        lc_center = LineCollection(
            lines, colors="#ffffff", linewidths=0.5, alpha=0.5, linestyles="dashed"
        )
        ax.add_collection(lc_center)

    legend_handles = []
    if compliant_edge_ids:
        poly = polyline_for_edge_ids(edges, compliant_edge_ids)
        if len(poly) >= 2:
            (h,) = ax.plot(
                [p[0] for p in poly],
                [p[1] for p in poly],
                color="#2e7d32",
                linewidth=3.5,
                alpha=0.95,
                zorder=8,
                label="compliant (long)",
            )
            legend_handles.append(h)
    if baseline_edge_ids:
        poly = polyline_for_edge_ids(edges, baseline_edge_ids)
        if len(poly) >= 2:
            (h,) = ax.plot(
                [p[0] for p in poly],
                [p[1] for p in poly],
                color="#c62828",
                linewidth=3.0,
                alpha=0.95,
                zorder=9,
                label="baseline (short)",
            )
            legend_handles.append(h)

    if marker_xy is not None:
        ax.plot(
            marker_xy[0],
            marker_xy[1],
            "o",
            color="red",
            markersize=14,
            markeredgecolor="#8b0000",
            markeredgewidth=1.5,
            zorder=10,
        )
    elif baseline_edge_ids or compliant_edge_ids:
        # Mark spawn at the start of the approach edge when routes are shown.
        spawn_ids = baseline_edge_ids or compliant_edge_ids or []
        if spawn_ids:
            spawn_poly = polyline_for_edge_ids(edges, [spawn_ids[0]])
            if spawn_poly:
                ax.plot(
                    spawn_poly[0][0],
                    spawn_poly[0][1],
                    "o",
                    color="#1565c0",
                    markersize=10,
                    markeredgecolor="#0d47a1",
                    markeredgewidth=1.2,
                    zorder=11,
                    label="spawn",
                )
                legend_handles.append(ax.lines[-1])

    if legend and legend_handles:
        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
            fancybox=False,
            edgecolor="#666666",
        )

    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Image saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Render a scene SUMO network as a static map image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scene", help="Scene folder name under scenes/ (e.g. junc_100012502)"
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Scenes root directory (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: scenes/<scene>/custom.png)",
    )
    parser.add_argument("--figsize", type=float, default=12, help="Figure size (default: 12)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI (default: 150)")
    parser.add_argument(
        "--no-routes",
        action="store_true",
        help="Do not overlay dual_path baseline/compliant routes from meta",
    )
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    scene_dir = resolve_scene_dir(scenes_dir, args.scene)

    out_path = args.out if args.out is not None else scene_dir / "custom.png"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene: {scene_dir}")
    meta = load_scene_meta(scene_dir)

    net_file = resolve_net_file(scene_dir, meta)
    net_path = scene_dir / net_file
    print(f"  net.xml: {net_path}")

    print("\nParsing network...")
    edges, junctions = parse_sumo_net(net_path)
    print(f"  {len(edges)} lanes, {len(junctions)} junctions")

    baseline = compliant = None
    if not args.no_routes:
        baseline, compliant, _spawn = routes_from_dual_path_meta(meta)
        if baseline or compliant:
            print(
                f"  dual_path overlays: baseline={len(baseline or [])} edges, "
                f"compliant={len(compliant or [])} edges"
            )

    print("\nRendering...")
    render_network(
        edges,
        junctions,
        out_path,
        figsize=(args.figsize, args.figsize),
        dpi=args.dpi,
        baseline_edge_ids=baseline,
        compliant_edge_ids=compliant,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
