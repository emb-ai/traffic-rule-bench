#!/usr/bin/env python3
"""Debug / materialize helper: render a scene SUMO net as a static PNG.

Used by ``build_scenes/materialize_scenes.py`` for review-UI previews
(``custom_cropped.png``). Also runnable ad-hoc:

  python -m tools.render_map <scene> --scenes-dir data/stop/scenes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Path setup — tools/ is inside priority_bench/
TOOLS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from core.sumo.sumo_utils import resolve_net_file, load_scene_meta, resolve_scene_dir

SCENES_DIR_DEFAULT = PACKAGE_DIR / "data" / "yield" / "scenes"


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


def render_network(
    edges,
    junctions,
    out_path: Path,
    figsize=(12, 12),
    dpi=150,
    marker_xy: tuple[float, float] | None = None,
):
    """Render the road network to an image."""
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
    render_network(
        edges, junctions, out_path, figsize=(args.figsize, args.figsize), dpi=args.dpi
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
