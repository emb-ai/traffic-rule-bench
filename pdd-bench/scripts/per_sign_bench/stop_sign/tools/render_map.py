#!/usr/bin/env python3
"""Debug: Render a scene map as static image."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Path setup - tools/ is inside stop_sign/
TOOLS_DIR = Path(__file__).resolve().parent
STOP_SIGN_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(STOP_SIGN_DIR))

from lib.sumo_utils import resolve_net_file, load_scene_meta, resolve_scene_dir

SCENES_DIR_DEFAULT = STOP_SIGN_DIR / "scenes"


def parse_net_location(net_path: Path) -> dict | None:
    """Read SUMO net ``location`` bounds for lat/lon → local XY conversion."""
    import xml.etree.ElementTree as ET

    loc = ET.parse(net_path).getroot().find("location")
    if loc is None:
        return None
    try:
        orig = tuple(float(v) for v in loc.get("origBoundary", "").split(","))
        conv = tuple(float(v) for v in loc.get("convBoundary", "").split(","))
    except (TypeError, ValueError):
        return None
    if len(orig) != 4 or len(conv) != 4:
        return None
    return {"orig": orig, "conv": conv}


def latlon_to_net_xy(lat: float, lon: float, location: dict) -> tuple[float, float]:
    """Map WGS84 point to local net coordinates using ``origBoundary`` / ``convBoundary``."""
    lon_min, lat_min, lon_max, lat_max = location["orig"]
    x_min, y_min, x_max, y_max = location["conv"]
    if lon_max == lon_min or lat_max == lat_min:
        return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x = (lon - lon_min) / (lon_max - lon_min) * (x_max - x_min) + x_min
    y = (lat - lat_min) / (lat_max - lat_min) * (y_max - y_min) + y_min
    return x, y


def _points_within_radius(points, cx: float, cy: float, radius_m: float) -> bool:
    r2 = radius_m * radius_m
    return any((x - cx) ** 2 + (y - cy) ** 2 <= r2 for x, y in points)


def filter_network_by_radius(
    edges,
    junctions,
    cx: float,
    cy: float,
    radius_m: float,
):
    """Keep lanes / junction polygons that intersect the crop circle."""
    kept_edges = [e for e in edges if _points_within_radius(e["points"], cx, cy, radius_m)]
    kept_junctions = []
    for junc in junctions:
        pts = junc.get("points") or [(junc.get("x", 0.0), junc.get("y", 0.0))]
        if _points_within_radius(pts, cx, cy, radius_m):
            kept_junctions.append(junc)
    return kept_edges, kept_junctions


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


def render_network(
    edges,
    junctions,
    out_path: Path,
    figsize=(12, 12),
    dpi=150,
    *,
    center: tuple[float, float] | None = None,
    radius_m: float | None = None,
    show_center: bool = True,
):
    """Render the road network to an image."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#f0f0f0")
    
    for junc in junctions:
        if len(junc["points"]) >= 3:
            polygon = mpatches.Polygon(junc["points"], closed=True, 
                                       facecolor="#909090", edgecolor="#707070", 
                                       linewidth=0.5, alpha=0.7)
            ax.add_patch(polygon)
    
    lines = []
    for edge in edges:
        pts = edge["points"]
        if len(pts) >= 2:
            lines.append(pts)
    
    if lines:
        lc = LineCollection(lines, colors="#404040", linewidths=2.0, alpha=0.9)
        ax.add_collection(lc)
        
        lc_center = LineCollection(lines, colors="#ffffff", linewidths=0.5, 
                                   alpha=0.5, linestyles="dashed")
        ax.add_collection(lc_center)

    if center is not None and radius_m is not None:
        cx, cy = center
        ax.set_xlim(cx - radius_m, cx + radius_m)
        ax.set_ylim(cy - radius_m, cy + radius_m)
        circle = mpatches.Circle(
            (cx, cy),
            radius_m,
            fill=False,
            edgecolor="#cc0000",
            linewidth=1.0,
            linestyle="--",
            alpha=0.35,
        )
        ax.add_patch(circle)
        if show_center:
            ax.plot(cx, cy, marker="+", color="#cc0000", markersize=14, markeredgewidth=2)
    else:
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
