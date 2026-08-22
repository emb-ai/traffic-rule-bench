#!/usr/bin/env python3
"""Ad-hoc CLI: render a scene SUMO net as a static PNG.

Library code lives in ``traffic_bench.scene_collection.preview``.

  python -m tools.render_map <scene> --scenes-dir data/scenes/yield
"""
from __future__ import annotations

import argparse
from pathlib import Path

from traffic_bench.eval.core.sumo.sumo_utils import load_scene_meta, resolve_net_file, resolve_scene_dir
from traffic_bench.scene_collection.paths import DATA_SCENES
from traffic_bench.scene_collection.preview import (
    crosswalk_xy_from_meta,
    parse_sumo_net,
    render_network,
    routes_from_dual_path_meta,
)

SCENES_DIR_DEFAULT = DATA_SCENES / "yield"


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
    n_cross = sum(1 for e in edges if e.get("kind") == "crossing")
    print(f"  {len(edges)} lanes, {len(junctions)} junctions, {n_cross} crossings")

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
        crosswalk_xy=None if n_cross else crosswalk_xy_from_meta(junctions, meta),
    )
    print(f"Image saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("\nDone!")


if __name__ == "__main__":
    main()
