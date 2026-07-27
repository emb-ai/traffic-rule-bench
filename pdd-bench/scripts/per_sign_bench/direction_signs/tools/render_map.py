#!/usr/bin/env python3
"""Debug: Render a scene map as static image."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Path setup - tools/ is inside direction_signs/
TOOLS_DIR = Path(__file__).resolve().parent
DIRECTION_SIGNS_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(DIRECTION_SIGNS_DIR))

from lib.sumo_utils import resolve_net_file, load_scene_meta, resolve_scene_dir

SCENES_DIR_DEFAULT = DIRECTION_SIGNS_DIR / "scenes"


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
                    edges.append({
                        "id": edge_id,
                        "lane_id": lane.get("id"),
                        "points": points,
                        "width": float(lane.get("width", 3.2)),
                    })
    
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
                junctions.append({"id": junction.get("id"), "points": points, "x": x, "y": y})
    
    return edges, junctions


def edge_shapes_by_id(edges) -> dict[str, list[tuple[float, float]]]:
    """Map SUMO edge id -> representative lane polyline (longest lane)."""
    best: dict[str, list[tuple[float, float]]] = {}
    for edge in edges:
        eid = edge.get("id")
        pts = edge.get("points") or []
        if not eid or len(pts) < 2:
            continue
        prev = best.get(eid)
        if prev is None or len(pts) > len(prev):
            best[eid] = list(pts)
    return best


def polylines_for_edge_path(
    edge_shapes: dict[str, list[tuple[float, float]]],
    edge_ids: list[str] | tuple[str, ...],
) -> list[list[tuple[float, float]]]:
    """Collect polylines for a sequence of SUMO edge ids."""
    out: list[list[tuple[float, float]]] = []
    for eid in edge_ids:
        pts = edge_shapes.get(eid)
        if pts and len(pts) >= 2:
            out.append(pts)
    return out


def offset_polyline(
    pts: list[tuple[float, float]],
    offset_m: float,
) -> list[tuple[float, float]]:
    """Shift a polyline left/right by ``offset_m`` (sign = side)."""
    if len(pts) < 2 or abs(offset_m) < 1e-9:
        return list(pts)
    out: list[tuple[float, float]] = []
    for i in range(len(pts)):
        if i == 0:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
        elif i == len(pts) - 1:
            x0, y0 = pts[-2]
            x1, y1 = pts[-1]
        else:
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-9:
            out.append(pts[i])
            continue
        nx, ny = -dy / length, dx / length
        x, y = pts[i]
        out.append((x + nx * offset_m, y + ny * offset_m))
    return out


def offset_polylines(
    polylines: list[list[tuple[float, float]]],
    offset_m: float,
) -> list[list[tuple[float, float]]]:
    return [offset_polyline(pts, offset_m) for pts in polylines]


def continuous_route_polyline(
    edge_shapes: dict[str, list[tuple[float, float]]],
    edge_ids: list[str] | tuple[str, ...],
    *,
    gap_bridge: bool = True,
) -> list[tuple[float, float]]:
    """Stitch edge shapes into one polyline (bridge omitted junction internals)."""
    route: list[tuple[float, float]] = []
    prev_end: tuple[float, float] | None = None
    for eid in edge_ids:
        pts = edge_shapes.get(eid)
        if not pts or len(pts) < 2:
            continue
        if prev_end is not None and gap_bridge:
            gap = (
                (pts[0][0] - prev_end[0]) ** 2 + (pts[0][1] - prev_end[1]) ** 2
            ) ** 0.5
            if gap > 0.5:
                if not route or route[-1] != prev_end:
                    route.append(prev_end)
                route.append(pts[0])
        if route and abs(route[-1][0] - pts[0][0]) < 1e-6 and abs(route[-1][1] - pts[0][1]) < 1e-6:
            route.extend(pts[1:])
        else:
            route.extend(pts)
        prev_end = pts[-1]
    return route


def dual_path_overlays(
    ego_edge_id: str,
    turn_path: list[str] | tuple[str, ...],
    straight_path: list[str] | tuple[str, ...],
    *,
    turn_dir: str,
    turn_length_m: float,
    straight_length_m: float,
) -> list[dict]:
    """Build overlays for full spawn→dest routes (both include shared tail).

    Continuous polylines bridge junction gaps; lateral offsets keep overlapping
    final edges visible so both colors clearly arrive at destination.
    """
    turn_full = [ego_edge_id, *turn_path]
    straight_full = [ego_edge_id, *straight_path]
    return [
        {
            "label": f"straight ({straight_length_m:.0f}m) → dest",
            "color": "#1f77b4",
            "edge_ids": straight_full,
            "continuous": True,
            "linewidth": 4.5,
            "zorder": 6,
            "offset_m": 1.8,
            "mark_end": True,
        },
        {
            "label": f"turn/{turn_dir} ({turn_length_m:.0f}m) → dest",
            "color": "#ff7f0e",
            "edge_ids": turn_full,
            "continuous": True,
            "linewidth": 4.5,
            "zorder": 7,
            "offset_m": -1.8,
            "mark_end": True,
        },
    ]


def point_on_edge(
    edge_shapes: dict[str, list[tuple[float, float]]],
    edge_id: str,
    *,
    at: str = "start",
) -> tuple[float, float] | None:
    """Return start or end point of an edge polyline."""
    pts = edge_shapes.get(edge_id)
    if not pts:
        return None
    return pts[0] if at == "start" else pts[-1]


def render_network(
    edges,
    junctions,
    out_path: Path,
    figsize=(12, 12),
    dpi=150,
    marker_xy: tuple[float, float] | None = None,
    path_overlays: list[dict] | None = None,
    spawn_xy: tuple[float, float] | None = None,
    dest_xy: tuple[float, float] | None = None,
    legend: bool = False,
):
    """Render the road network to an image.

    ``path_overlays`` entries: ``{"label", "color", "edge_ids"}`` or
    ``{"label", "color", "polylines"}``. Optional ``linewidth``, ``zorder``,
    ``offset_m``, ``linestyle``.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#f0f0f0")
    
    for junc in junctions:
        if len(junc["points"]) >= 3:
            polygon = mpatches.Polygon(junc["points"], closed=True, 
                                       facecolor="#909090", edgecolor="#707070", 
                                       linewidth=0.5, alpha=0.7)
            ax.add_patch(polygon)
    
    lines = []
    widths = []
    for edge in edges:
        pts = edge["points"]
        if len(pts) >= 2:
            lines.append(pts)
            widths.append(edge["width"])
    
    if lines:
        lc = LineCollection(lines, colors="#404040", linewidths=2.0, alpha=0.9)
        ax.add_collection(lc)
        
        lc_center = LineCollection(lines, colors="#ffffff", linewidths=0.5, 
                                   alpha=0.5, linestyles="dashed")
        ax.add_collection(lc_center)

    edge_shapes = edge_shapes_by_id(edges)
    legend_handles = []
    if path_overlays:
        for overlay in path_overlays:
            color = overlay.get("color", "#1f77b4")
            label = overlay.get("label")
            polylines = overlay.get("polylines")
            if polylines is None:
                edge_ids = overlay.get("edge_ids") or ()
                if overlay.get("continuous"):
                    stitched = continuous_route_polyline(edge_shapes, edge_ids)
                    polylines = [stitched] if len(stitched) >= 2 else []
                else:
                    polylines = polylines_for_edge_path(edge_shapes, edge_ids)
            offset_m = float(overlay.get("offset_m") or 0.0)
            if offset_m:
                polylines = offset_polylines(polylines, offset_m)
            if not polylines:
                continue
            lc_path = LineCollection(
                polylines,
                colors=color,
                linewidths=float(overlay.get("linewidth") or 3.5),
                alpha=0.95,
                zorder=int(overlay.get("zorder") or 6),
                linestyles=overlay.get("linestyle") or "solid",
            )
            ax.add_collection(lc_path)
            if overlay.get("mark_end"):
                end_pts = [pts[-1] for pts in polylines if pts]
                if end_pts:
                    xs = [p[0] for p in end_pts]
                    ys = [p[1] for p in end_pts]
                    ax.scatter(
                        xs,
                        ys,
                        s=90,
                        c=color,
                        marker="^",
                        edgecolors="black",
                        linewidths=0.6,
                        zorder=12,
                    )
            if label and legend:
                legend_handles.append(
                    mpatches.Patch(color=color, label=label)
                )

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
            label="junction" if legend else None,
        )

    if spawn_xy is not None:
        ax.plot(
            spawn_xy[0],
            spawn_xy[1],
            "s",
            color="#2ca02c",
            markersize=12,
            markeredgecolor="#145214",
            markeredgewidth=1.2,
            zorder=11,
            label="spawn" if legend else None,
        )
    if dest_xy is not None:
        ax.plot(
            dest_xy[0],
            dest_xy[1],
            "*",
            color="#d62728",
            markersize=18,
            markeredgecolor="#7f0000",
            markeredgewidth=1.0,
            zorder=11,
            label="destination" if legend else None,
        )

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = legend_handles + handles
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
    
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
    parser.add_argument("scene", help="Scene folder name under scenes/ (e.g. savvinskaya_3)")
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

    print("\nRendering...")
    render_network(edges, junctions, out_path, 
                   figsize=(args.figsize, args.figsize), dpi=args.dpi)

    print("\nDone!")


if __name__ == "__main__":
    main()
