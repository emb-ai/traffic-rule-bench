"""Find pedestrian crossings in a SUMO net and crop scenes around each one."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lib.crosswalk_layout import net_has_crossings, parse_crossing_junction_id
from lib.junction_crop import (
    JunctionLayoutError,
    crop_net_around_latlon,
    crop_net_to_junction_only,
    net_xy_to_latlon,
    parse_net_location,
    resolve_full_source_net,
)
from lib.manifest_config import DEFAULT_CROSSWALK_CROP_RADIUS_M
from lib.sumo_utils import is_vehicle_drivable_lane


@dataclass(frozen=True)
class CrosswalkPick:
    crosswalk_id: str
    junction_id: str
    center_xy: Tuple[float, float]
    crossed_edge_ids: Tuple[str, ...]
    approach_edge_ids: Tuple[str, ...]
    max_approach_lane_m: float
    approach_lane_count: int


def json_dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def crosswalk_scene_name(core_scene_name: str, rank: int) -> str:
    """Build crop folder name, e.g. sign_71853 + 0 -> sign_71853_cw0."""
    return f"{core_scene_name}_cw{rank}"


def is_crosswalk_scene_meta(meta: dict) -> bool:
    return meta.get("scene_kind") == "crosswalk" or meta.get("crosswalk_id")


def _net_contains_crosswalk(net_path: Path, crosswalk_id: str) -> bool:
    root = ET.parse(net_path).getroot()
    for edge in root.findall("edge"):
        if edge.get("id") == crosswalk_id and edge.get("function") == "crossing":
            return True
    return False


def _crop_crosswalk_net(
    source_net: Path,
    out_net: Path,
    pick: CrosswalkPick,
    *,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    crop_mode: str,
) -> str:
    """Crop source net around a crossing. Returns the crop mode actually used."""
    mode = (crop_mode or "geo").strip().lower()
    if mode not in {"geo", "junction"}:
        raise ValueError(f"Unsupported crop_mode: {crop_mode!r}")

    if mode == "geo":
        crop_net_around_latlon(source_net, center_lat, center_lon, out_net, radius_m=radius_m)
        if _net_contains_crosswalk(out_net, pick.crosswalk_id):
            return "geo"
        print(
            f"  [crop] geo crop lost crossing {pick.crosswalk_id}; "
            f"falling back to junction crop"
        )

    crop_net_to_junction_only(
        source_net,
        pick.junction_id,
        out_net,
        arm_length_m=radius_m,
    )
    return "junction"


def _parse_shape_center(shape_str: str) -> Tuple[float, float]:
    points: list[tuple[float, float]] = []
    for token in (shape_str or "").strip().split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        points.append((float(x_str), float(y_str)))
    if not points:
        return 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _lane_length_from_el(lane_el: ET.Element) -> float:
    length = float(lane_el.get("length", 0) or 0)
    if length > 0:
        return length
    shape = (lane_el.get("shape") or "").strip().split()
    coords = [tuple(map(float, p.split(","))) for p in shape if "," in p]
    if len(coords) < 2:
        return 0.0
    return sum(
        ((coords[i + 1][0] - coords[i][0]) ** 2 + (coords[i + 1][1] - coords[i][1]) ** 2) ** 0.5
        for i in range(len(coords) - 1)
    )


def _load_edge_endpoints(root: ET.Element) -> dict[str, tuple[str, str]]:
    endpoints: dict[str, tuple[str, str]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge.get("function") in {"internal", "crossing", "walkingarea"}:
            continue
        if edge_id.startswith(":"):
            continue
        endpoints[edge_id] = (edge.get("from", ""), edge.get("to", ""))
    return endpoints


def collect_crosswalk_candidates(
    net_path: Path | str,
    *,
    min_approach_lane_m: float = 10.0,
) -> List[CrosswalkPick]:
    """Return all crossings with at least one viable vehicle approach lane."""
    net_path = Path(net_path)
    root = ET.parse(net_path).getroot()
    endpoints = _load_edge_endpoints(root)

    edge_max_lane: dict[str, float] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge.get("function") not in {None, "normal", ""}:
            if edge.get("function") not in {None, "normal", ""}:
                continue
        if edge_id.startswith(":"):
            continue
        best = 0.0
        for lane in edge.findall("lane"):
            if not is_vehicle_drivable_lane(lane):
                continue
            best = max(best, _lane_length_from_el(lane))
        if best > 0:
            edge_max_lane[edge_id] = best

    picks: list[CrosswalkPick] = []
    for edge in root.findall("edge"):
        if edge.get("function") != "crossing":
            continue
        crosswalk_id = edge.get("id", "")
        crossed_raw = (edge.get("crossingEdges") or "").strip()
        if not crosswalk_id or not crossed_raw:
            continue

        junction_id = parse_crossing_junction_id(crosswalk_id) or ""
        if not junction_id:
            continue

        crossed_edge_ids = tuple(e for e in crossed_raw.split() if e)
        approach_edges = [
            eid
            for eid in crossed_edge_ids
            if endpoints.get(eid, ("", ""))[1] == junction_id
        ]
        viable_approaches = [
            eid for eid in approach_edges if edge_max_lane.get(eid, 0.0) >= min_approach_lane_m
        ]
        if not viable_approaches:
            continue

        lane_el = edge.find("lane")
        center_xy = _parse_shape_center(lane_el.get("shape", "") if lane_el is not None else "")
        max_lane = max(edge_max_lane.get(eid, 0.0) for eid in viable_approaches)
        picks.append(
            CrosswalkPick(
                crosswalk_id=crosswalk_id,
                junction_id=junction_id,
                center_xy=center_xy,
                crossed_edge_ids=crossed_edge_ids,
                approach_edge_ids=tuple(viable_approaches),
                max_approach_lane_m=max_lane,
                approach_lane_count=len(viable_approaches),
            )
        )

    return picks


def find_ranked_crosswalks(
    net_path: Path | str,
    *,
    min_approach_lane_m: float = 10.0,
    max_crosswalks: int = 8,
) -> List[CrosswalkPick]:
    """Rank crossings by longest approach lane, then number of approach arms."""
    if max_crosswalks < 1:
        raise ValueError("max_crosswalks must be at least 1")

    candidates = collect_crosswalk_candidates(net_path, min_approach_lane_m=min_approach_lane_m)
    if not candidates:
        raise JunctionLayoutError(
            f"No pedestrian crossing with vehicle approach lane >= {min_approach_lane_m}m in {net_path}"
        )

    candidates.sort(
        key=lambda pick: (-pick.max_approach_lane_m, -pick.approach_lane_count, pick.crosswalk_id)
    )
    return candidates[:max_crosswalks]


def crop_scene_to_crosswalk_pick(
    scene_dir: Path,
    pick: CrosswalkPick,
    *,
    source_net: Path,
    radius_m: float = DEFAULT_CROSSWALK_CROP_RADIUS_M,
    crop_mode: str = "geo",
    output_dir: Optional[Path] = None,
    output_scene_name: Optional[str] = None,
    output_net_name: str = "map.net.xml",
    base_meta: Optional[dict] = None,
    backup_original: bool = True,
    crosswalk_rank: Optional[int] = None,
    core_scene_name: Optional[str] = None,
) -> CrosswalkPick:
    """Crop ``source_net`` around a crossing junction; write meta into ``output_dir``."""
    from lib.sumo_utils import load_scene_meta

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
    used_crop_mode = _crop_crosswalk_net(
        source_net,
        out_net,
        pick,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=radius_m,
        crop_mode=crop_mode,
    )

    center_path = output_dir / "center.json"
    center_path.write_text(
        json_dumps({"lat": center_lat, "lon": center_lon}) + "\n",
        encoding="utf-8",
    )

    scene_name = output_scene_name or meta.get("scene_name", scene_dir.name)
    if output_scene_name is None and output_dir != scene_dir and crosswalk_rank is not None:
        scene_name = crosswalk_scene_name(core_name, crosswalk_rank)

    meta.update(
        {
            "scene_name": scene_name,
            "scene_kind": "crosswalk",
            "pdd_code": "5.19",
            "sign_type": "crosswalk",
            "core_scene_name": core_name,
            "net_file": output_net_name,
            "latitude": center_lat,
            "longitude": center_lon,
            "crop_radius_m": radius_m,
            "crop_mode": used_crop_mode,
            "crosswalk_id": pick.crosswalk_id,
            "crosswalk_rank": crosswalk_rank,
            "junction_id": pick.junction_id,
            "crossed_edge_ids": list(pick.crossed_edge_ids),
            "approach_edge_ids": list(pick.approach_edge_ids),
        }
    )
    (output_dir / "meta.json").write_text(json_dumps(meta) + "\n", encoding="utf-8")
    return pick
