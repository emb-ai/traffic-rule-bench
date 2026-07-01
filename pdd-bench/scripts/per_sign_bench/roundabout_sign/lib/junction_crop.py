"""Crop SUMO nets to a traffic circle plus attached spoke roads for PDD 4.3."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .junction_priority_layout import JunctionLayoutError, SumoEdge, _load_net


def _find_netconvert() -> str:
    for path in (
        shutil.which("netconvert"),
        "/home/jovyan/.local/bin/netconvert",
        "/usr/bin/netconvert",
    ):
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("netconvert not found on PATH")


def parse_net_location(net_path: Path) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Return (conv_boundary, orig_boundary) from a SUMO net location tag."""
    root = ET.parse(net_path).getroot()
    loc = root.find("location")
    if loc is None:
        raise JunctionLayoutError(f"No <location> tag in {net_path}")

    conv = tuple(float(x) for x in loc.get("convBoundary", "").split(","))
    orig = tuple(float(x) for x in loc.get("origBoundary", "").split(","))
    if len(conv) != 4 or len(orig) != 4:
        raise JunctionLayoutError(f"Invalid location boundaries in {net_path}")
    return conv, orig  # type: ignore[return-value]


def net_xy_to_latlon(
    x: float,
    y: float,
    conv_boundary: Tuple[float, float, float, float],
    orig_boundary: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Map SUMO network XY to WGS84 lat/lon via linear conv/orig bounds."""
    min_x, min_y, max_x, max_y = conv_boundary
    min_lon, min_lat, max_lon, max_lat = orig_boundary
    if max_x == min_x or max_y == min_y:
        raise JunctionLayoutError("Degenerate convBoundary in net.xml")
    lon = min_lon + (x - min_x) / (max_x - min_x) * (max_lon - min_lon)
    lat = min_lat + (y - min_y) / (max_y - min_y) * (max_lat - min_lat)
    return lat, lon


def _parse_shape_str(shape_str: str) -> List[Tuple[float, float]]:
    if not shape_str:
        return []
    points: List[Tuple[float, float]] = []
    for token in shape_str.strip().split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points


def _shape_str(points: List[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _polyline_length_pts(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
    return total


def _trim_polyline(
    points: List[Tuple[float, float]],
    max_length: float,
    *,
    from_end: bool,
) -> List[Tuple[float, float]]:
    """Keep at most ``max_length`` meters along a polyline from start or end."""
    if len(points) < 2 or max_length <= 0:
        return points[:1] if points else []
    if _polyline_length_pts(points) <= max_length:
        return points

    if from_end:
        acc = 0.0
        kept: List[Tuple[float, float]] = [points[-1]]
        for i in range(len(points) - 2, -1, -1):
            p0, p1 = points[i], points[i + 1]
            seg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if acc + seg <= max_length:
                kept.append(p0)
                acc += seg
            else:
                need = max_length - acc
                if seg > 0:
                    t = need / seg
                    kept.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
                break
        kept.reverse()
        return kept

    acc = 0.0
    kept = [points[0]]
    for i in range(1, len(points)):
        p0, p1 = points[i - 1], points[i]
        seg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if acc + seg <= max_length:
            kept.append(p1)
            acc += seg
        else:
            need = max_length - acc
            if seg > 0:
                t = need / seg
                kept.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
            break
    return kept


def _edge_max_lane_length(edge: SumoEdge) -> float:
    if not edge.lanes:
        return 0.0
    return max(lane.length for lane in edge.lanes)


def collect_roundabout_keep_edge_ids(
    net_path: Path,
    pick: "RoundaboutPick",
    *,
    spoke_extension_m: float,
    extend_spoke_edge_id: Optional[str] = None,
) -> List[str]:
    """Ring edges, all spoke approaches, and upstream chain on every spoke."""
    _, edges, _, _ = _load_net(net_path)
    ring_juncs = set(pick.ring_junction_ids)
    ring_edges = set(pick.ring_edge_ids)
    keep: Set[str] = set(ring_edges) | set(pick.spoke_edge_ids)

    incoming_to: Dict[str, List[str]] = defaultdict(list)
    for edge_id, edge in edges.items():
        incoming_to[edge.to_node].append(edge_id)

    def walk_upstream(spoke_id: str) -> None:
        spoke = edges.get(spoke_id)
        if spoke is None:
            return
        remaining = spoke_extension_m - _edge_max_lane_length(spoke)
        if remaining <= 0:
            return
        stack: List[Tuple[str, float]] = [(spoke.from_node, remaining)]
        visited: Set[str] = {spoke_id}
        while stack:
            node_id, budget = stack.pop()
            for pred_id in incoming_to.get(node_id, []):
                if pred_id in visited or pred_id in ring_edges:
                    continue
                pred = edges.get(pred_id)
                if pred is None:
                    continue
                keep.add(pred_id)
                visited.add(pred_id)
                pred_len = _edge_max_lane_length(pred)
                next_budget = budget - pred_len
                if next_budget > 0:
                    stack.append((pred.from_node, next_budget))

    for spoke_id in pick.spoke_edge_ids:
        walk_upstream(spoke_id)

    pruned = {
        eid
        for eid in keep
        if eid in ring_edges or (edges.get(eid) and edges[eid].to_node in ring_juncs)
    }
    upstream_keep: Set[str] = set()
    for eid in sorted(keep):
        if eid in pruned:
            continue
        edge = edges.get(eid)
        if edge is None:
            continue
        if any(pred in pruned or pred in upstream_keep for pred in incoming_to.get(edge.from_node, [])):
            upstream_keep.add(eid)
    pruned |= upstream_keep

    if not pruned:
        raise JunctionLayoutError("No roundabout edges to keep after pruning")
    return sorted(pruned)


def _trim_edge_lanes(
    edge_el: ET.Element,
    *,
    max_arm_length_m: float,
    trim_from_end: bool,
) -> None:
    lane_els = edge_el.findall("lane")
    shapes: List[List[Tuple[float, float]]] = []
    for lane_el in lane_els:
        pts = _trim_polyline(
            _parse_shape_str(lane_el.get("shape", "")),
            max_arm_length_m,
            from_end=trim_from_end,
        )
        shapes.append(pts)
        lane_el.set("shape", _shape_str(pts))
        lane_el.set("length", f"{_polyline_length_pts(pts):.2f}")

    if shapes:
        edge_el.set("shape", _shape_str(shapes[0]))


def _trim_roundabout_spokes_in_net(
    root: ET.Element,
    *,
    ring_junction_ids: Set[str],
    ring_edge_ids: Set[str],
    max_spoke_length_m: float,
) -> None:
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge.get("function", "normal") == "internal":
            continue
        if edge_id in ring_edge_ids:
            continue
        to_junction = edge.get("to", "")
        if to_junction in ring_junction_ids:
            _trim_edge_lanes(edge, max_arm_length_m=max_spoke_length_m, trim_from_end=True)
        else:
            _trim_edge_lanes(edge, max_arm_length_m=max_spoke_length_m, trim_from_end=False)


def _update_net_bounds(root: ET.Element) -> None:
    xs: List[float] = []
    ys: List[float] = []
    for lane in root.findall(".//lane"):
        for x, y in _parse_shape_str(lane.get("shape", "")):
            xs.append(x)
            ys.append(y)
    if not xs:
        return
    pad = 1.0
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    loc = root.find("location")
    if loc is None:
        return
    conv = [float(v) for v in loc.get("convBoundary", "0,0,0,0").split(",")]
    orig = [float(v) for v in loc.get("origBoundary", "0,0,0,0").split(",")]
    if len(conv) == 4 and len(orig) == 4 and conv[2] != conv[0] and conv[3] != conv[1]:
        min_lon, min_lat, max_lon, max_lat = orig
        cmin_x, cmin_y, cmax_x, cmax_y = conv
        o_min_lon = min_lon + (min_x - cmin_x) / (cmax_x - cmin_x) * (max_lon - min_lon)
        o_max_lon = min_lon + (max_x - cmin_x) / (cmax_x - cmin_x) * (max_lon - min_lon)
        o_min_lat = min_lat + (min_y - cmin_y) / (cmax_y - cmin_y) * (max_lat - min_lat)
        o_max_lat = min_lat + (max_y - cmin_y) / (cmax_y - cmin_y) * (max_lat - min_lat)
        loc.set("origBoundary", f"{o_min_lon},{o_min_lat},{o_max_lon},{o_max_lat}")
    loc.set("convBoundary", f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}")


def crop_net_to_roundabout_only(
    net_path: Path,
    pick: "RoundaboutPick",
    out_path: Path,
    *,
    spoke_extension_m: float,
    max_spoke_length_m: float,
    extend_spoke_edge_id: Optional[str] = None,
) -> None:
    """Keep only the traffic circle ring, spokes, and upstream chain on one sign spoke."""
    edge_ids = collect_roundabout_keep_edge_ids(
        net_path,
        pick,
        spoke_extension_m=spoke_extension_m,
        extend_spoke_edge_id=extend_spoke_edge_id,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(edge_ids))
        edge_list_file = Path(handle.name)

    try:
        cmd = [
            _find_netconvert(),
            "--sumo-net-file",
            str(net_path),
            "-o",
            str(out_path),
            "--keep-edges.input-file",
            str(edge_list_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise JunctionLayoutError(
                f"netconvert roundabout crop failed for {net_path}:\n"
                f"{result.stderr or result.stdout}"
            )
    finally:
        edge_list_file.unlink(missing_ok=True)

    if not out_path.is_file():
        raise JunctionLayoutError(f"netconvert did not write {out_path}")

    tree = ET.parse(out_path)
    root = tree.getroot()
    _trim_roundabout_spokes_in_net(
        root,
        ring_junction_ids=set(pick.ring_junction_ids),
        ring_edge_ids=set(pick.ring_edge_ids),
        max_spoke_length_m=max_spoke_length_m,
    )
    _update_net_bounds(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)


def resolve_full_source_net(scene_dir: Path, meta: dict) -> Path:
    """Return the uncropped SUMO net for a scene (backup or catalog net)."""
    from .sumo_utils import resolve_net_file

    scene_dir = scene_dir.resolve()
    scene_name = meta.get("scene_name", scene_dir.name)
    candidates: List[Path] = []

    for path in sorted(scene_dir.glob("*.net.xml.full.bak")):
        candidates.append(path)

    named = scene_dir / f"{scene_name}.net.xml"
    if named.is_file():
        candidates.append(named)

    for path in sorted(scene_dir.glob("*.net.xml")):
        if path.name != "map.net.xml":
            candidates.append(path)

    if candidates:
        return max(candidates, key=lambda path: path.stat().st_size)

    return scene_dir / resolve_net_file(scene_dir, meta)


def roundabout_scene_name(core_scene_name: str) -> str:
    """Single cropped roundabout folder per core map (e.g. sign_144460_rb)."""
    return f"{core_scene_name}_rb"


def roundabout_spoke_scene_name(core_scene_name: str, spoke_rank: int) -> str:
    return f"{core_scene_name}_rb_s{spoke_rank:02d}"


def resolve_catalog_sign_spoke(
    pick: "RoundaboutPick",
    catalog_road_id: Optional[str],
    spokes: Iterable[str],
) -> str:
    """Pick the spoke edge that matches the catalog sign road."""
    spoke_list = list(spokes)
    if not spoke_list:
        raise JunctionLayoutError("No spokes on traffic circle")

    if pick.approach_edge_id in spoke_list:
        return pick.approach_edge_id

    if catalog_road_id:
        if catalog_road_id in spoke_list:
            return catalog_road_id
        prefix = catalog_road_id.split("#", 1)[0]
        chain = [
            sid
            for sid in spoke_list
            if sid == catalog_road_id or sid.startswith(prefix + "#")
        ]
        if chain:
            return chain[0]
        stripped = catalog_road_id.lstrip("-")
        for candidate in (f"-{stripped}", stripped):
            if candidate in spoke_list:
                return candidate

    return spoke_list[0]


def crop_scene_to_roundabout(
    scene_dir: Path,
    *,
    ego_spoke_edge_id: str,
    spoke_rank: int = 0,
    spoke_extension_m: float = 80.0,
    max_spoke_length_m: float = 80.0,
    min_lane_length_m: float = 10.0,
    output_net_name: str = "map.net.xml",
    backup_original: bool = True,
    output_dir: Optional[Path] = None,
    output_scene_name: Optional[str] = None,
    source_pick: Optional["RoundaboutPick"] = None,
    sumo_roundabout: Optional["SumoRoundabout"] = None,
) -> "RoundaboutPick":
    """Crop to ring+spokes only; place the 4.3 sign on ``ego_spoke_edge_id``."""
    from .roundabout_topology import RoundaboutPick, detect_roundabout, resolve_sumo_roundabout
    from .roundabout_fingerprint import sumo_roundabout_record
    from .sumo_utils import load_scene_meta

    scene_dir = scene_dir.resolve()
    meta = load_scene_meta(scene_dir)
    source_net = resolve_full_source_net(scene_dir, meta)
    catalog_sign_road = meta.get("road_id")
    pick = source_pick or detect_roundabout(source_net, sign_edge_id=catalog_sign_road)

    if ego_spoke_edge_id not in pick.spoke_edge_ids:
        prefix = ego_spoke_edge_id.split("#", 1)[0]
        spoke_match = any(
            eid == ego_spoke_edge_id or eid.startswith(prefix + "#")
            for eid in pick.spoke_edge_ids
        )
        if not spoke_match:
            raise JunctionLayoutError(
                f"Spoke edge {ego_spoke_edge_id!r} is not attached to the traffic circle "
                f"(spokes={list(pick.spoke_edge_ids)})"
            )

    output_dir = (output_dir or scene_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if backup_original and output_dir == scene_dir:
        backup_path = scene_dir / f"{source_net.name}.full.bak"
        if not backup_path.exists():
            shutil.copy2(source_net, backup_path)

    out_net = output_dir / output_net_name
    crop_net_to_roundabout_only(
        source_net,
        pick,
        out_net,
        spoke_extension_m=spoke_extension_m,
        max_spoke_length_m=max_spoke_length_m,
        extend_spoke_edge_id=ego_spoke_edge_id,
    )

    cropped_pick = detect_roundabout(
        out_net,
        sign_edge_id=ego_spoke_edge_id,
        ego_spoke_edge_id=ego_spoke_edge_id,
    )

    conv, orig = parse_net_location(out_net)
    center_lat, center_lon = net_xy_to_latlon(
        cropped_pick.center_xy[0],
        cropped_pick.center_xy[1],
        conv,
        orig,
    )

    center_path = output_dir / "center.json"
    center_path.write_text(
        json_dumps({"lat": center_lat, "lon": center_lon}) + "\n",
        encoding="utf-8",
    )

    core_name = meta.get("core_scene_name") or meta.get("scene_name", scene_dir.name)
    scene_name = output_scene_name or roundabout_spoke_scene_name(core_name, spoke_rank)

    meta = dict(meta)
    meta.update(
        {
            "scene_name": scene_name,
            "scene_kind": "roundabout",
            "core_scene_name": core_name,
            "net_file": output_net_name,
            "latitude": center_lat,
            "longitude": center_lon,
            "roundabout_spoke_rank": spoke_rank,
            "roundabout_sign_spoke_edge": ego_spoke_edge_id,
            "spoke_extension_m": spoke_extension_m,
            "roundabout_entry_junction": cropped_pick.entry_junction_id,
            "roundabout_center_xy": [cropped_pick.center_xy[0], cropped_pick.center_xy[1]],
            "roundabout_ring_edges": list(cropped_pick.ring_edge_ids),
            "roundabout_spoke_edges": list(cropped_pick.spoke_edge_ids),
            "roundabout_approach_edge": ego_spoke_edge_id,
        }
    )

    from .scene_augmentation import pick_default_yield_spawn_meta_for_net

    spawn_meta = pick_default_yield_spawn_meta_for_net(
        out_net,
        prefer_ego_edge_id=ego_spoke_edge_id,
        min_lane_length=min_lane_length_m,
        ring_edge_ids=cropped_pick.ring_edge_ids,
        spoke_edge_ids=cropped_pick.spoke_edge_ids,
        entry_junction_id=cropped_pick.entry_junction_id,
    )
    if spawn_meta:
        meta.update(spawn_meta)
    else:
        meta.pop("destination_lane_id", None)
        meta.pop("destination_edge_id", None)

    meta["catalog_sign_road_id"] = catalog_sign_road
    meta["roundabout_approach_edge"] = ego_spoke_edge_id

    meta.pop("distance_from_start", None)
    meta.pop("crop_radius_m", None)
    meta.pop("junction_id", None)
    meta.pop("junction_arm_count", None)
    meta.pop("junction_rank", None)
    meta["sign_spawn_distance"] = 30.0

    if sumo_roundabout is None:
        try:
            sumo_roundabout = resolve_sumo_roundabout(
                source_net,
                sign_edge_id=catalog_sign_road,
            )
        except Exception:
            sumo_roundabout = None
    if sumo_roundabout is not None:
        meta.update(sumo_roundabout_record(sumo_roundabout))

    meta_path = output_dir / "meta.json"
    meta_path.write_text(json_dumps(meta) + "\n", encoding="utf-8")
    return cropped_pick


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2)
