"""Find a 3- or 4-arm junction in a SUMO net and crop the network around it."""

from __future__ import annotations

import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .junction_priority_layout import (
    INTERSECTION_JUNCTION_TYPES,
    JunctionLayoutError,
    _incoming_edges_for_junction,
    _load_net,
)


@dataclass(frozen=True)
class JunctionPick:
    junction_id: str
    center_xy: Tuple[float, float]
    total_lanes: int
    incoming_edge_ids: Tuple[str, ...]
    arm_count: int


# Backward-compatible alias
FourArmJunctionPick = JunctionPick


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


def meters_to_degrees(meters: float, lat: float) -> Tuple[float, float]:
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(lat))
    return meters / meters_per_degree_lat, meters / meters_per_degree_lon


def geo_boundary_for_center(
    lat: float,
    lon: float,
    radius_m: float,
) -> str:
    """SUMO --keep-edges.in-geo-boundary string: west,south,east,north."""
    delta_lat, delta_lon = meters_to_degrees(radius_m, lat)
    west = lon - delta_lon
    east = lon + delta_lon
    south = lat - delta_lat
    north = lat + delta_lat
    return f"{west},{south},{east},{north}"


def find_best_junction_with_arm_count(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 10.0,
    require_arm_count: int = 4,
) -> JunctionPick:
    """Pick the junction with ``require_arm_count`` incoming arms and the most lanes.

    Each incoming arm must have at least one lane longer than ``min_lane_length_m``.
    Raises ``JunctionLayoutError`` when no suitable junction exists.
    """
    net_path = Path(net_path)
    junctions, edges, _, _ = _load_net(net_path)

    candidates: List[JunctionPick] = []
    for jid, info in junctions.items():
        if info["type"] not in INTERSECTION_JUNCTION_TYPES:
            continue
        incoming = _incoming_edges_for_junction(jid, edges)
        if len(incoming) != require_arm_count:
            continue
        if not all(any(lane.length > min_lane_length_m for lane in arm.lanes) for arm in incoming):
            continue
        total_lanes = sum(len(arm.lanes) for arm in incoming)
        candidates.append(
            JunctionPick(
                junction_id=jid,
                center_xy=info["center"],
                total_lanes=total_lanes,
                incoming_edge_ids=tuple(arm.edge_id for arm in incoming),
                arm_count=require_arm_count,
            )
        )

    if not candidates:
        raise JunctionLayoutError(
            f"No {require_arm_count}-arm junction with all arms having a lane "
            f"> {min_lane_length_m}m in {net_path}"
        )

    candidates.sort(key=lambda pick: pick.total_lanes, reverse=True)
    return candidates[0]


def find_best_four_arm_junction(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 10.0,
    require_arm_count: int = 4,
) -> JunctionPick:
    """Pick the 4-arm junction with the most incoming lanes."""
    return find_best_junction_with_arm_count(
        net_path,
        min_lane_length_m=min_lane_length_m,
        require_arm_count=require_arm_count,
    )


def collect_intersection_junction_candidates(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 10.0,
    arm_counts: Tuple[int, ...] = (4, 3),
) -> List[JunctionPick]:
    """Return all qualifying 3- and 4-arm junctions in a net."""
    net_path = Path(net_path)
    allowed = set(arm_counts)
    junctions, edges, _, _ = _load_net(net_path)

    candidates: List[JunctionPick] = []
    for jid, info in junctions.items():
        if info["type"] not in INTERSECTION_JUNCTION_TYPES:
            continue
        incoming = _incoming_edges_for_junction(jid, edges)
        arm_count = len(incoming)
        if arm_count not in allowed:
            continue
        if not all(any(lane.length > min_lane_length_m for lane in arm.lanes) for arm in incoming):
            continue
        total_lanes = sum(len(arm.lanes) for arm in incoming)
        candidates.append(
            JunctionPick(
                junction_id=jid,
                center_xy=info["center"],
                total_lanes=total_lanes,
                incoming_edge_ids=tuple(arm.edge_id for arm in incoming),
                arm_count=arm_count,
            )
        )
    return candidates


def find_ranked_intersection_junctions(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 10.0,
    arm_counts: Tuple[int, ...] = (4, 3),
    max_junctions: int = 5,
) -> List[JunctionPick]:
    """Rank junctions: 4-arm before 3-arm, then by total incoming lanes (descending)."""
    if max_junctions < 1:
        raise ValueError("max_junctions must be at least 1")

    candidates = collect_intersection_junction_candidates(
        net_path,
        min_lane_length_m=min_lane_length_m,
        arm_counts=arm_counts,
    )
    if not candidates:
        raise JunctionLayoutError(
            f"No 3- or 4-arm junction with all arms having a lane "
            f"> {min_lane_length_m}m in {net_path}"
        )

    candidates.sort(key=lambda pick: (-pick.arm_count, -pick.total_lanes, pick.junction_id))
    return candidates[:max_junctions]


def find_best_intersection_junction(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 10.0,
    arm_counts: Tuple[int, ...] = (4, 3),
) -> JunctionPick:
    """Prefer a 4-arm junction; fall back to 3-arm when none qualify."""
    return find_ranked_intersection_junctions(
        net_path,
        min_lane_length_m=min_lane_length_m,
        arm_counts=arm_counts,
        max_junctions=1,
    )[0]


def try_find_junction_with_arm_count(
    net_path: Path | str,
    *,
    arm_count: int,
    min_lane_length_m: float = 10.0,
) -> Optional[JunctionPick]:
    """Return the best junction pick for ``arm_count`` incoming arms, or None."""
    try:
        return find_best_junction_with_arm_count(
            net_path,
            min_lane_length_m=min_lane_length_m,
            require_arm_count=arm_count,
        )
    except JunctionLayoutError:
        return None


def try_find_junction_for_arm_counts(
    net_path: Path | str,
    *,
    arm_counts: Tuple[int, ...] = (4,),
    min_lane_length_m: float = 10.0,
) -> Optional[JunctionPick]:
    """Return the best junction pick for the first qualifying arm count (highest first)."""
    for arm_count in sorted(set(arm_counts), reverse=True):
        pick = try_find_junction_with_arm_count(
            net_path,
            arm_count=arm_count,
            min_lane_length_m=min_lane_length_m,
        )
        if pick is not None:
            return pick
    return None


def try_find_four_arm_junction(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 10.0,
) -> Optional[JunctionPick]:
    """Return the best 4-arm junction pick, or None if none qualify."""
    return try_find_junction_with_arm_count(
        net_path,
        arm_count=4,
        min_lane_length_m=min_lane_length_m,
    )


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


def collect_junction_arm_edge_ids(net_path: Path, junction_id: str) -> List[str]:
    """Return incoming + outgoing edge ids for one intersection."""
    root = ET.parse(net_path).getroot()
    ids: List[str] = []
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge.get("function", "normal") == "internal":
            continue
        if edge.get("to") == junction_id or edge.get("from") == junction_id:
            ids.append(edge_id)
    if not ids:
        raise JunctionLayoutError(f"No external edges found for junction {junction_id}")
    return sorted(set(ids))


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


def _trim_arms_in_net(root: ET.Element, junction_id: str, max_arm_length_m: float) -> None:
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge.get("function", "normal") == "internal":
            continue
        if edge.get("to") == junction_id:
            _trim_edge_lanes(edge, max_arm_length_m=max_arm_length_m, trim_from_end=True)
        elif edge.get("from") == junction_id:
            _trim_edge_lanes(edge, max_arm_length_m=max_arm_length_m, trim_from_end=False)


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


def crop_net_to_junction_only(
    net_path: Path,
    junction_id: str,
    out_path: Path,
    *,
    arm_length_m: float,
) -> None:
    """Keep only the picked junction, its internal links, and incoming/outgoing arms."""
    import tempfile

    edge_ids = collect_junction_arm_edge_ids(net_path, junction_id)
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
                f"netconvert junction crop failed for {net_path}:\n"
                f"{result.stderr or result.stdout}"
            )
    finally:
        edge_list_file.unlink(missing_ok=True)

    if not out_path.is_file():
        raise JunctionLayoutError(f"netconvert did not write {out_path}")

    tree = ET.parse(out_path)
    root = tree.getroot()
    _trim_arms_in_net(root, junction_id, arm_length_m)
    _update_net_bounds(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)


def crop_net_around_latlon(
    net_path: Path,
    center_lat: float,
    center_lon: float,
    out_path: Path,
    *,
    radius_m: float,
) -> None:
    """Crop a SUMO net to a geo boundary using netconvert."""
    boundary = geo_boundary_for_center(center_lat, center_lon, radius_m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _find_netconvert(),
        "--sumo-net-file",
        str(net_path),
        "-o",
        str(out_path),
        "--keep-edges.in-geo-boundary",
        boundary,
        "--geometry.remove",
        "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise JunctionLayoutError(
            f"netconvert crop failed for {net_path}:\n{result.stderr or result.stdout}"
        )
    if not out_path.is_file():
        raise JunctionLayoutError(f"netconvert did not write {out_path}")


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


def crop_scene_to_junction_pick(
    scene_dir: Path,
    pick: JunctionPick,
    *,
    source_net: Path,
    radius_m: float = 80.0,
    min_lane_length_m: float = 10.0,
    output_dir: Optional[Path] = None,
    output_scene_name: Optional[str] = None,
    output_net_name: str = "map.net.xml",
    base_meta: Optional[dict] = None,
    backup_original: bool = True,
    junction_rank: Optional[int] = None,
    core_scene_name: Optional[str] = None,
) -> JunctionPick:
    """Crop ``source_net`` around ``pick``; write center.json and meta into ``output_dir``."""
    from .sumo_utils import load_scene_meta

    scene_dir = scene_dir.resolve()
    output_dir = (output_dir or scene_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(base_meta if base_meta is not None else load_scene_meta(scene_dir))
    source_net = source_net.resolve()
    core_name = core_scene_name or meta.get("scene_name", scene_dir.name)

    conv, orig = parse_net_location(source_net)
    center_lat, center_lon = net_xy_to_latlon(
        pick.center_xy[0],
        pick.center_xy[1],
        conv,
        orig,
    )

    if backup_original and output_dir == scene_dir and source_net.name != output_net_name:
        backup_path = scene_dir / f"{source_net.name}.full.bak"
        if not backup_path.exists():
            shutil.copy2(source_net, backup_path)

    out_net = output_dir / output_net_name
    crop_net_to_junction_only(
        source_net,
        pick.junction_id,
        out_net,
        arm_length_m=radius_m,
    )

    center_path = output_dir / "center.json"
    center_path.write_text(
        json_dumps({"lat": center_lat, "lon": center_lon}) + "\n",
        encoding="utf-8",
    )

    scene_name = output_scene_name or meta.get("scene_name", scene_dir.name)
    if output_scene_name is None and output_dir != scene_dir:
        scene_name = f"{core_name}_j{pick.junction_id}"

    meta.update(
        {
            "scene_name": scene_name,
            "scene_kind": "junction",
            "core_scene_name": core_name,
            "net_file": output_net_name,
            "latitude": center_lat,
            "longitude": center_lon,
            "crop_radius_m": radius_m,
            "junction_id": pick.junction_id,
            "junction_arm_count": pick.arm_count,
            "junction_center_xy": [pick.center_xy[0], pick.center_xy[1]],
        }
    )
    if junction_rank is not None:
        meta["junction_rank"] = junction_rank

    from .scene_augmentation import pick_default_main_spawn_meta_for_net

    spawn_meta = pick_default_main_spawn_meta_for_net(
        out_net,
        prefer_ego_edge_id=meta.get("road_id"),
        min_lane_length=min_lane_length_m,
    )
    if spawn_meta:
        meta.update(spawn_meta)
    else:
        meta.pop("destination_lane_id", None)
        meta.pop("destination_edge_id", None)

    meta.pop("distance_from_start", None)
    meta["sign_spawn_distance"] = 30.0

    meta_path = output_dir / "meta.json"
    meta_path.write_text(json_dumps(meta) + "\n", encoding="utf-8")

    return pick


def crop_scene_to_junction(
    scene_dir: Path,
    *,
    radius_m: float = 80.0,
    min_lane_length_m: float = 10.0,
    output_net_name: str = "map.net.xml",
    backup_original: bool = True,
) -> JunctionPick:
    """Crop scene net around the best 4- or 3-arm junction; write center.json."""
    from .sumo_utils import load_scene_meta

    scene_dir = scene_dir.resolve()
    meta = load_scene_meta(scene_dir)
    source_net = resolve_full_source_net(scene_dir, meta)
    pick = find_best_intersection_junction(source_net, min_lane_length_m=min_lane_length_m)
    return crop_scene_to_junction_pick(
        scene_dir,
        pick,
        source_net=source_net,
        radius_m=radius_m,
        min_lane_length_m=min_lane_length_m,
        output_net_name=output_net_name,
        base_meta=meta,
        backup_original=backup_original,
    )


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2)
