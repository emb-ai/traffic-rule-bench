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


def lane_shapes_by_id(edges) -> dict[tuple[str, int], list[tuple[float, float]]]:
    """Map (edge_id, lane_num) -> lane polyline."""
    out: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for edge in edges:
        eid = edge.get("id")
        lid = edge.get("lane_id") or ""
        pts = edge.get("points") or []
        if not eid or len(pts) < 2:
            continue
        try:
            ln = int(str(lid).rsplit("_", 1)[1])
        except (ValueError, IndexError):
            ln = 0
        out[(eid, ln)] = list(pts)
    return out


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


def _polyline_fraction(
    pts: list[tuple[float, float]],
    *,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> list[tuple[float, float]]:
    """Keep the portion of a polyline between cumulative-length fractions."""
    if len(pts) < 2:
        return list(pts)
    start_frac = max(0.0, min(1.0, float(start_frac)))
    end_frac = max(0.0, min(1.0, float(end_frac)))
    if end_frac <= start_frac:
        return [pts[0]]
    segs: list[float] = []
    total = 0.0
    for i in range(len(pts) - 1):
        d = (
            (pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2
        ) ** 0.5
        segs.append(d)
        total += d
    if total < 1e-9:
        return list(pts)
    t0, t1 = start_frac * total, end_frac * total

    def _at(dist: float) -> tuple[float, float]:
        acc = 0.0
        for i, seg in enumerate(segs):
            if acc + seg >= dist - 1e-9 or i == len(segs) - 1:
                if seg < 1e-9:
                    return pts[i]
                u = max(0.0, min(1.0, (dist - acc) / seg))
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                return (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
            acc += seg
        return pts[-1]

    out: list[tuple[float, float]] = [_at(t0)]
    acc = 0.0
    for i, seg in enumerate(segs):
        next_acc = acc + seg
        if next_acc <= t0 + 1e-9:
            acc = next_acc
            continue
        if acc >= t1 - 1e-9:
            break
        if next_acc < t1 - 1e-9:
            out.append(pts[i + 1])
        acc = next_acc
    end_pt = _at(t1)
    if abs(out[-1][0] - end_pt[0]) > 1e-6 or abs(out[-1][1] - end_pt[1]) > 1e-6:
        out.append(end_pt)
    return out


def _stitch_polylines(
    parts: list[list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    route: list[tuple[float, float]] = []
    for pts in parts:
        if not pts or len(pts) < 2:
            if pts and (not route or route[-1] != pts[0]):
                route.extend(pts)
            continue
        if not route:
            route.extend(pts)
            continue
        # Bridge gap between consecutive parts (lane-change / junction).
        gap = (
            (pts[0][0] - route[-1][0]) ** 2 + (pts[0][1] - route[-1][1]) ** 2
        ) ** 0.5
        if gap > 0.5:
            route.append(pts[0])
        if abs(route[-1][0] - pts[0][0]) < 1e-6 and abs(route[-1][1] - pts[0][1]) < 1e-6:
            route.extend(pts[1:])
        else:
            route.extend(pts)
    return route


def point_on_lane(
    lane_shapes: dict[tuple[str, int], list[tuple[float, float]]],
    edge_id: str,
    lane_num: int,
    *,
    distance_before_end_m: float = 20.0,
) -> tuple[float, float] | None:
    """Point on a lane ~``distance_before_end_m`` before the lane end."""
    pts = lane_shapes.get((edge_id, int(lane_num)))
    if not pts or len(pts) < 2:
        return None
    # Walk back from the end.
    remain = float(distance_before_end_m)
    if remain <= 0:
        return pts[-1]
    acc = 0.0
    for i in range(len(pts) - 1, 0, -1):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if acc + seg >= remain:
            # interpolate from end backward
            need = remain - acc
            if seg < 1e-9:
                return pts[i]
            u = need / seg  # from end toward start
            return (x1 - (x1 - x0) * u, y1 - (y1 - y0) * u)
        acc += seg
    return pts[0]


def dual_path_overlays(
    ego_edge_id: str,
    turn_path: list[str] | tuple[str, ...],
    straight_path: list[str] | tuple[str, ...],
    *,
    turn_dir: str,
    turn_length_m: float,
    straight_length_m: float,
    lane_shapes: dict[tuple[str, int], list[tuple[float, float]]] | None = None,
    spawn_lane_num: int | None = None,
    target_lane_num: int | None = None,
    show_illegal_to_dest: bool = True,
    spawn_distance_before_end_m: float = 20.0,
) -> list[dict]:
    """Overlays for 5.15.1.

    Blue = compliant: spawn lane → lane-change onto target → dest.
    Orange = baseline: spawn lane → forbidden first exit → dest (when
    ``show_illegal_to_dest``, post connector-injection), else wrong spur.
    """
    spawn_ln = 0 if spawn_lane_num is None else int(spawn_lane_num)
    target_ln = spawn_ln if target_lane_num is None else int(target_lane_num)
    dest_edges = list(straight_path)

    # Prefer lane-level approach geometry when available.
    spawn_lane_pts = (
        (lane_shapes or {}).get((ego_edge_id, spawn_ln)) if lane_shapes else None
    )
    target_lane_pts = (
        (lane_shapes or {}).get((ego_edge_id, target_ln)) if lane_shapes else None
    )
    use_lanes = bool(
        spawn_lane_pts
        and target_lane_pts
        and len(spawn_lane_pts) >= 2
        and len(target_lane_pts) >= 2
    )

    if use_lanes:
        # Fraction along approach where ego spawns (distance before lane end).
        total_len = 0.0
        for i in range(len(spawn_lane_pts) - 1):
            total_len += (
                (spawn_lane_pts[i + 1][0] - spawn_lane_pts[i][0]) ** 2
                + (spawn_lane_pts[i + 1][1] - spawn_lane_pts[i][1]) ** 2
            ) ** 0.5
        if total_len < 1e-6:
            spawn_frac = 0.0
        else:
            spawn_frac = max(
                0.0,
                min(0.85, 1.0 - float(spawn_distance_before_end_m) / total_len),
            )
        # Lane-change happens AFTER spawn, finishing before the junction.
        lc_start = min(spawn_frac + 0.05, 0.80)
        lc_end = min(max(lc_start + 0.25, 0.90), 0.98)

        blue_approach = _stitch_polylines(
            [
                _polyline_fraction(spawn_lane_pts, start_frac=0.0, end_frac=lc_start),
                _polyline_fraction(target_lane_pts, start_frac=lc_end, end_frac=1.0),
            ]
        )
        orange_approach = list(spawn_lane_pts)

        if show_illegal_to_dest and dest_edges:
            orange_overlay = {
                "label": (
                    f"wrong / illegal from L{spawn_ln} "
                    f"→ dest ({straight_length_m:.0f}m)"
                ),
                "color": "#ff7f0e",
                "route_parts": {
                    "approach": orange_approach,
                    "edge_ids": dest_edges,
                },
                "linewidth": 4.5,
                "zorder": 7,
                "offset_m": -1.2,
                "mark_end": True,
                "linestyle": "dashed",
            }
        else:
            orange_overlay = {
                "label": f"wrong / no change (L{spawn_ln}, {turn_dir}, {turn_length_m:.0f}m)",
                "color": "#ff7f0e",
                "route_parts": {
                    "approach": orange_approach,
                    "edge_ids": list(turn_path),
                },
                "linewidth": 4.5,
                "zorder": 7,
                "offset_m": -1.2,
                "mark_end": True,
                "linestyle": "dashed",
            }

        blue_overlay = {
            "label": (
                f"correct / L{spawn_ln}→L{target_ln} "
                f"({straight_length_m:.0f}m) → dest"
            ),
            "color": "#1f77b4",
            "route_parts": {
                "approach": blue_approach,
                "edge_ids": dest_edges,
            },
            "linewidth": 4.5,
            "zorder": 6,
            "offset_m": 1.2,
            "mark_end": True,
        }
        lc_seg = _stitch_polylines(
            [
                _polyline_fraction(spawn_lane_pts, start_frac=lc_start, end_frac=min(lc_start + 0.08, lc_end)),
                _polyline_fraction(target_lane_pts, start_frac=max(lc_end - 0.08, lc_start), end_frac=lc_end),
            ]
        )
        lc_overlay = {
            "label": f"lane-change L{spawn_ln}→L{target_ln}",
            "color": "#2ca02c",
            "polylines": [lc_seg] if len(lc_seg) >= 2 else [],
            "linewidth": 5.5,
            "zorder": 8,
            "linestyle": "solid",
        }
        return [blue_overlay, orange_overlay, lc_overlay]

    # Fallback: edge-centerline overlays (legacy).
    turn_full = [ego_edge_id, *turn_path]
    straight_full = [ego_edge_id, *straight_path]
    if show_illegal_to_dest and dest_edges:
        orange_edges = [ego_edge_id, *dest_edges]
        orange_label = f"wrong / illegal ({turn_dir}→dest, {straight_length_m:.0f}m)"
    else:
        orange_edges = turn_full
        orange_label = f"wrong / no change ({turn_dir}, {turn_length_m:.0f}m)"
    return [
        {
            "label": f"correct / lane-change ({straight_length_m:.0f}m) → dest",
            "color": "#1f77b4",
            "edge_ids": straight_full,
            "continuous": True,
            "linewidth": 4.5,
            "zorder": 6,
            "offset_m": 1.8,
            "mark_end": True,
        },
        {
            "label": orange_label,
            "color": "#ff7f0e",
            "edge_ids": orange_edges,
            "continuous": True,
            "linewidth": 4.5,
            "zorder": 7,
            "offset_m": -1.8,
            "mark_end": True,
            "linestyle": "dashed",
        },
    ]


def _arrow_polygon(origin, heading_rad: float, length: float = 4.0, width: float = 1.6):
    import math

    ux, uy = math.cos(heading_rad), math.sin(heading_rad)
    px, py = -uy, ux
    tip = (origin[0] + ux * length, origin[1] + uy * length)
    left = (
        origin[0] + ux * (length * 0.35) + px * width,
        origin[1] + uy * (length * 0.35) + py * width,
    )
    right = (
        origin[0] + ux * (length * 0.35) - px * width,
        origin[1] + uy * (length * 0.35) - py * width,
    )
    base_l = (origin[0] + px * (width * 0.35), origin[1] + py * (width * 0.35))
    base_r = (origin[0] - px * (width * 0.35), origin[1] - py * (width * 0.35))
    return [base_l, left, tip, right, base_r]


def lane_direction_arrow_overlays(
    lane_shapes: dict[tuple[str, int], list[tuple[float, float]]],
    approach_lane_dirs: dict[str, list[str]] | dict[int, list[str]] | None,
    ego_edge_id: str,
    *,
    pullback_m: float = 8.0,
) -> list[dict]:
    """Draw per-lane allowed-direction arrows near the end of the ego approach."""
    import math

    if not approach_lane_dirs:
        return []
    overlays: list[dict] = []
    dir_heading_offset = {"s": 0.0, "l": math.radians(55), "r": math.radians(-55)}
    color = "#f5f5f5"
    for raw_ln, dirs in approach_lane_dirs.items():
        try:
            ln = int(raw_ln)
        except (TypeError, ValueError):
            continue
        shape = lane_shapes.get((ego_edge_id, ln))
        if not shape or len(shape) < 2:
            continue
        # Anchor ~pullback_m before lane end.
        end = shape[-1]
        prev = shape[-2]
        seg_dx, seg_dy = end[0] - prev[0], end[1] - prev[1]
        seg_len = math.hypot(seg_dx, seg_dy) or 1.0
        heading = math.atan2(seg_dy, seg_dx)
        frac = min(pullback_m / seg_len, 0.9)
        ax = end[0] - seg_dx * frac
        ay = end[1] - seg_dy * frac
        # Lateral fan so multiple dirs on one lane do not fully overlap.
        unique_dirs = []
        for d in dirs:
            d = str(d).lower()
            if d in dir_heading_offset and d not in unique_dirs:
                unique_dirs.append(d)
        n = max(len(unique_dirs), 1)
        for i, d in enumerate(unique_dirs):
            lateral = (i - (n - 1) / 2.0) * 1.2
            px, py = -math.sin(heading), math.cos(heading)
            origin = (ax + px * lateral, ay + py * lateral)
            poly = _arrow_polygon(origin, heading + dir_heading_offset[d])
            overlays.append(
                {
                    "label": f"lane {ln}: {','.join(unique_dirs)}" if i == 0 else None,
                    "color": color,
                    "polylines": [poly],
                    "linewidth": 1.8,
                    "zorder": 9,
                    "arrow_fill": True,
                    "dir": d,
                }
            )
    return overlays


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
            if polylines is None and overlay.get("route_parts"):
                parts = overlay["route_parts"]
                approach = list(parts.get("approach") or [])
                dest_poly = continuous_route_polyline(
                    edge_shapes, parts.get("edge_ids") or ()
                )
                # Lateral offset only on post-junction edges so approach stays
                # on true lane geometry (spawn vs target).
                offset_m = float(overlay.get("offset_m") or 0.0)
                if offset_m and dest_poly:
                    dest_poly = offset_polyline(dest_poly, offset_m)
                stitched = _stitch_polylines(
                    [p for p in (approach, dest_poly) if p and len(p) >= 1]
                )
                polylines = [stitched] if len(stitched) >= 2 else []
                # Prevent a second full-route offset below.
                overlay = dict(overlay)
                overlay["offset_m"] = 0.0
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
            if overlay.get("arrow_fill"):
                for poly in polylines:
                    if len(poly) >= 3:
                        ax.add_patch(
                            mpatches.Polygon(
                                poly,
                                closed=True,
                                facecolor=color,
                                edgecolor="#222222",
                                linewidth=0.6,
                                alpha=0.95,
                                zorder=int(overlay.get("zorder") or 9),
                            )
                        )
                if label and legend:
                    legend_handles.append(mpatches.Patch(color=color, label=label))
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
