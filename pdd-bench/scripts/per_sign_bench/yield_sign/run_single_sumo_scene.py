#!/usr/bin/env python3
"""Render a SUMO network (net.xml) as an image or GIF.

Usage:
    python run_single_sumo_scene.py \
        --scene-dir scenes/check \
        --out scenes/check/output.png

    python run_single_sumo_scene.py \
        --scene-dir scenes/savvinskaya_3 \
        --out scenes/savvinskaya_3/output.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCENES_ROOT = SCRIPT_DIR / "scenes"


def load_scene_meta(scene_dir: Path) -> dict:
    """Load meta.json from a scene directory."""
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {scene_dir}")
    with open(meta_path) as f:
        return json.load(f)


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


def render_network(edges, junctions, out_path: Path, figsize=(12, 12), dpi=150):
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
    
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    
    print(f"Image saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Render SUMO network as image")
    parser.add_argument("--scene-dir", required=True, help="Path to scene folder")
    parser.add_argument("--out", required=True, help="Output image path (png/jpg/pdf)")
    parser.add_argument("--figsize", type=float, default=12, help="Figure size (default: 12)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI (default: 150)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    if not scene_dir.is_absolute():
        if (SCENES_ROOT / args.scene_dir).exists():
            scene_dir = SCENES_ROOT / args.scene_dir
    scene_dir = scene_dir.resolve()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene: {scene_dir}")
    meta = load_scene_meta(scene_dir)
    
    net_file = meta.get("net_file")
    if not net_file:
        net_files = list(scene_dir.glob("*.net.xml"))
        if net_files:
            net_file = net_files[0].name
        else:
            raise FileNotFoundError(f"No .net.xml file found in {scene_dir}")
    
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
