"""Top-down PNG previews of a cropped SUMO net (review UI / harvest).

Backfill missing ``custom_cropped.png`` under a crop tree::

    python -m traffic_bench.scene_collection.preview \\
        --root traffic_bench/scene_collection/maps/crops/junction \\
        --skip-existing --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

Point = Tuple[float, float]


def _parse_shape(shape_str: str | None) -> List[Point]:
    if not shape_str:
        return []
    points: List[Point] = []
    for token in shape_str.strip().split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    return points


def parse_sumo_net(net_path: Path):
    """Parse SUMO net.xml and extract edges/lanes for rendering.

    Crossing edges (``function="crossing"``) are kept even though their ids
    start with ``:``; ``render_network`` draws them as a zebra.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(net_path)
    root = tree.getroot()

    edges = []
    junctions = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id") or ""
        kind = "crossing" if edge.get("function") == "crossing" else "road"
        if kind != "crossing" and edge_id.startswith(":"):
            continue

        for lane in edge.findall("lane"):
            points = _parse_shape(lane.get("shape"))
            if kind == "crossing":
                outline = _parse_shape(lane.get("outlineShape"))
                if _polyline_length(outline) > _polyline_length(points):
                    points = outline
            if len(points) >= 2:
                edges.append(
                    {
                        "id": edge_id,
                        "lane_id": lane.get("id"),
                        "points": points,
                        "width": float(lane.get("width", 3.2)),
                        "kind": kind,
                    }
                )

    for junction in root.findall("junction"):
        junc_type = junction.get("type")
        if junc_type in ("internal", "dead_end"):
            continue
        x = float(junction.get("x", 0))
        y = float(junction.get("y", 0))
        points = _parse_shape(junction.get("shape"))
        junctions.append(
            {"id": junction.get("id"), "points": points, "x": x, "y": y}
        )

    return edges, junctions


def _polyline_length(pts: Sequence[Point]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])
    )


def _point_along(pts: Sequence[Point], dist: float) -> Optional[Point]:
    if len(pts) < 2:
        return None
    remain = max(0.0, dist)
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 1e-9:
            continue
        if remain <= seg:
            t = remain / seg
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        remain -= seg
    return pts[-1]


def _centroid(pts: Sequence[Point]) -> Point:
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _crossing_axes(pts: Sequence[Point]) -> tuple[Point, Point, Point]:
    """Return (center, unit across-road, unit along-road) from a crossing polyline."""
    cx, cy = _centroid(pts)
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    if math.hypot(dx, dy) < 1e-6:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    return (cx, cy), (ux, uy), (-uy, ux)


def _crossing_mark_poly(
    pts: Sequence[Point], *, along_m: float = 14.0, across_min_m: float = 12.0
) -> List[Point]:
    """Preview rectangle around a SUMO crossing so it stays visible on a long crop."""
    if len(pts) < 2:
        return []
    (cx, cy), (ux, uy), (vx, vy) = _crossing_axes(pts)
    half_across = max(_polyline_length(pts), across_min_m) / 2.0
    half_along = along_m / 2.0
    return [
        (cx - ux * half_across - vx * half_along, cy - uy * half_across - vy * half_along),
        (cx + ux * half_across - vx * half_along, cy + uy * half_across - vy * half_along),
        (cx + ux * half_across + vx * half_along, cy + uy * half_across + vy * half_along),
        (cx - ux * half_across + vx * half_along, cy - uy * half_across + vy * half_along),
    ]


def _zebra_hatch_lines(
    pts: Sequence[Point], *, period: float = 2.2, half_len: float = 6.0
) -> List[List[Point]]:
    """Stripes along the road, across a crossing polyline."""
    if len(pts) < 2:
        return []
    (cx, cy), (ux, uy), (vx, vy) = _crossing_axes(pts)
    half_across = max(_polyline_length(pts), 12.0) / 2.0
    ticks: List[List[Point]] = []
    t = -half_across
    while t <= half_across + 1e-6:
        px, py = cx + ux * t, cy + uy * t
        ticks.append(
            [
                (px - vx * half_len, py - vy * half_len),
                (px + vx * half_len, py + vy * half_len),
            ]
        )
        t += period
    return ticks


def crosswalk_xy_from_meta(junctions: Sequence[dict], meta: Optional[dict]) -> Optional[Point]:
    """Fallback mark when netconvert dropped the crossing edge but kept the node."""
    if not meta:
        return None
    node_id = meta.get("crosswalk_node_id")
    if not node_id:
        return None
    want = str(node_id)
    for junc in junctions:
        if str(junc.get("id") or "") == want:
            return (float(junc["x"]), float(junc["y"]))
    return None


def attach_crosswalk_overlay(edges: List[dict], junctions: Sequence[dict], meta: Optional[dict]) -> List[dict]:
    """If the net has no crossing edge, draw the injected split-node outline instead."""
    if any(e.get("kind") == "crossing" for e in edges):
        return edges
    poly = crosswalk_poly_from_meta(junctions, meta)
    if not poly:
        return edges
    return [*edges, {"id": "crosswalk", "points": poly, "kind": "crossing"}]


def crosswalk_poly_from_meta(junctions: Sequence[dict], meta: Optional[dict]) -> Optional[List[Point]]:
    """Junction outline of the injected split node (when SUMO omitted the crossing edge)."""
    if not meta:
        return None
    node_id = meta.get("crosswalk_node_id")
    if not node_id:
        return None
    want = str(node_id)
    for junc in junctions:
        if str(junc.get("id") or "") != want:
            continue
        pts = list(junc.get("points") or [])
        if len(pts) >= 3:
            return pts
    return None


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
    crosswalk_xy: tuple[float, float] | None = None,
):
    """Render the road network to an image.

    Optional dual-path overlays:
      * baseline (short / forbidden) — red
      * compliant (long / allowed) — green

    Crossing edges from the net are drawn as a zebra; ``crosswalk_xy`` is a
    fallback mark when the crossing edge is missing.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#f0f0f0")

    road_edges = [e for e in edges if e.get("kind") != "crossing"]
    crossing_edges = [e for e in edges if e.get("kind") == "crossing"]

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
    for edge in road_edges:
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
    drew_crossing = False
    hatch: List[List[Point]] = []
    for edge in crossing_edges:
        pts = edge.get("points") or []
        mark = _crossing_mark_poly(pts) if len(pts) >= 2 else []
        if len(mark) >= 3:
            ax.add_patch(
                mpatches.Polygon(
                    mark,
                    closed=True,
                    facecolor="#fff8e1",
                    edgecolor="#1a1a1a",
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=12,
                )
            )
            hatch.extend(_zebra_hatch_lines(pts))
            cx, cy = _centroid(pts)
            ax.plot(
                cx,
                cy,
                marker="s",
                markersize=9,
                markerfacecolor="#fff8e1",
                markeredgecolor="#1a1a1a",
                markeredgewidth=1.1,
                zorder=15,
                linestyle="None",
            )
            drew_crossing = True
    if hatch:
        ax.add_collection(
            LineCollection(
                hatch,
                colors="#1a1a1a",
                linewidths=2.0,
                alpha=0.95,
                zorder=14,
                capstyle="butt",
            )
        )
    if drew_crossing:
        legend_handles.append(
            mpatches.Patch(facecolor="#fff8e1", edgecolor="#1a1a1a", label="crosswalk")
        )
    elif crosswalk_xy is not None:
        ax.plot(
            crosswalk_xy[0],
            crosswalk_xy[1],
            marker="s",
            markersize=11,
            markerfacecolor="#fff8e1",
            markeredgecolor="#1a1a1a",
            markeredgewidth=1.4,
            zorder=12,
            linestyle="None",
            label="crosswalk",
        )
        legend_handles.append(ax.lines[-1])

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
            handles=legend_handles,
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


PREVIEW_NAME = "custom_cropped.png"


def iter_scene_dirs(root: Path) -> List[Path]:
    """Directories that already have a cropped ``map.net.xml``."""
    if not root.is_dir():
        return []
    return sorted(
        path.parent
        for path in root.rglob("map.net.xml")
        if path.is_file()
    )


def render_scene_preview(scene_dir: Path) -> Path:
    """Write ``custom_cropped.png`` next to ``map.net.xml``."""
    net = scene_dir / "map.net.xml"
    if not net.is_file():
        raise FileNotFoundError(f"no map.net.xml in {scene_dir}")
    meta: dict = {}
    meta_path = scene_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    road = meta.get("road_id")
    baseline, compliant, _spawn = routes_from_dual_path_meta(meta)
    if compliant is None and road:
        compliant = [str(road)]
    out_png = scene_dir / PREVIEW_NAME
    edges, junctions = parse_sumo_net(net)
    edges = attach_crosswalk_overlay(edges, junctions, meta)
    has_crossing = any(e.get("kind") == "crossing" for e in edges)
    render_network(
        edges,
        junctions,
        out_png,
        figsize=(6, 6),
        dpi=120,
        baseline_edge_ids=baseline,
        compliant_edge_ids=compliant,
        legend=bool(baseline or compliant or has_crossing),
        crosswalk_xy=None if has_crossing else crosswalk_xy_from_meta(junctions, meta),
    )
    return out_png


def _render_one(scene_dir: Path) -> Tuple[str, str, str]:
    try:
        render_scene_preview(scene_dir)
        return ("ok", scene_dir.name, "")
    except Exception as exc:  # noqa: BLE001
        return ("fail", scene_dir.name, str(exc))


def backfill_previews(
    root: Path,
    *,
    skip_existing: bool = False,
    workers: int = 4,
) -> None:
    """Render previews for cropped scenes under ``root``."""
    jobs: List[Path] = []
    skipped = 0
    for scene_dir in iter_scene_dirs(root):
        if skip_existing and (scene_dir / PREVIEW_NAME).is_file():
            skipped += 1
            continue
        jobs.append(scene_dir)

    print(
        f"[preview] {len(jobs)} to render under {root} "
        f"(skip_existing={skip_existing}, already_have={skipped}, workers={workers})"
    )
    ok = fail = 0
    t0 = time.time()
    workers = max(1, int(workers))

    def _consume(i: int, status: str, scene_id: str, detail: str) -> None:
        nonlocal ok, fail
        if status == "ok":
            ok += 1
        else:
            fail += 1
            print(f"  [fail] {scene_id}: {detail}")
        if i % 50 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail}")

    if not jobs:
        print("[preview] nothing to do")
        return
    if workers == 1:
        for i, scene_dir in enumerate(jobs, 1):
            status, scene_id, detail = _render_one(scene_dir)
            _consume(i, status, scene_id, detail)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_render_one, scene_dir) for scene_dir in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                status, scene_id, detail = fut.result()
                _consume(i, status, scene_id, detail)
    print(f"[preview] done in {time.time() - t0:.1f}s: ok={ok} fail={fail}")


def main(argv: Optional[List[str]] = None) -> int:
    from traffic_bench.scene_collection.paths import CROPS

    ap = argparse.ArgumentParser(
        prog="python -m traffic_bench.scene_collection.preview",
        description="Write custom_cropped.png for cropped SUMO scenes.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=CROPS,
        help=f"crop tree (default: {CROPS})",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip scenes that already have custom_cropped.png",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    backfill_previews(args.root, skip_existing=args.skip_existing, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
