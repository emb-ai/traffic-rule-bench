"""Find pedestrian crossings in a SUMO net and crop scenes around each one."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lib.crosswalk_layout import count_net_crossings, net_has_crossings, parse_crossing_junction_id
from lib.junction_crop import (
    JunctionLayoutError,
    _find_netconvert,
    _update_net_bounds,
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


def prune_net_to_single_crosswalk(
    net_path: Path,
    *,
    keep_crosswalk_id: str,
    junction_id: str,
) -> int:
    """Keep exactly one pedestrian crossing in a cropped SUMO net.

  Removes every other ``function="crossing"`` edge, walking areas on other
  junctions, and pedestrian connections that referenced them. Returns the
  number of removed crossing edges.
    """
    net_path = Path(net_path)
    tree = ET.parse(net_path)
    root = tree.getroot()
    removed_crossings = 0
    remove_ids: set[str] = set()
    junction_prefix = f":{junction_id}"

    for edge in list(root.findall("edge")):
        edge_id = edge.get("id", "")
        fn = edge.get("function")
        if fn == "crossing":
            if edge_id != keep_crosswalk_id:
                remove_ids.add(edge_id)
                root.remove(edge)
                removed_crossings += 1
        elif fn == "walkingarea" and not edge_id.startswith(junction_prefix):
            remove_ids.add(edge_id)
            root.remove(edge)

    for connection in list(root.findall("connection")):
        from_id = connection.get("from", "")
        to_id = connection.get("to", "")
        via_id = connection.get("via", "")
        if from_id in remove_ids or to_id in remove_ids or via_id in remove_ids:
            root.remove(connection)

    _update_net_bounds(root)
    ET.indent(tree, space="  ")

    with tempfile.NamedTemporaryFile("w", suffix=".net.xml", delete=False) as handle:
        edited_path = Path(handle.name)
    tree.write(edited_path, encoding="unicode", xml_declaration=True)

    with tempfile.NamedTemporaryFile(suffix=".net.xml", delete=False) as handle:
        cleaned_path = Path(handle.name)

    try:
        cmd = [
            _find_netconvert(),
            "--sumo-net-file",
            str(edited_path),
            "-o",
            str(cleaned_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise JunctionLayoutError(
                f"netconvert prune failed for {net_path}:\n{result.stderr or result.stdout}"
            )
        shutil.copy2(cleaned_path, net_path)
    finally:
        edited_path.unlink(missing_ok=True)
        cleaned_path.unlink(missing_ok=True)

    if count_net_crossings(net_path) != 1:
        raise JunctionLayoutError(
            f"Expected exactly one crossing ({keep_crosswalk_id}) in {net_path}, "
            f"found {count_net_crossings(net_path)}"
        )
    if not _net_contains_crosswalk(net_path, keep_crosswalk_id):
        raise JunctionLayoutError(
            f"Target crossing {keep_crosswalk_id} missing after prune in {net_path}"
        )
    return removed_crossings


def _crop_crosswalk_net(
    source_net: Path,
    out_net: Path,
    pick: CrosswalkPick,
    *,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    crop_mode: str,
    trim_geometry: bool = False,
) -> str:
    """Crop source net around a crossing. Returns the crop mode actually used."""
    mode = (crop_mode or "geo").strip().lower()
    if mode not in {"geo", "junction"}:
        raise ValueError(f"Unsupported crop_mode: {crop_mode!r}")

    if mode == "geo":
        crop_net_around_latlon(
            source_net,
            center_lat,
            center_lon,
            out_net,
            radius_m=radius_m,
            trim_geometry=trim_geometry,
            junction_id=pick.junction_id,
        )
        if _net_contains_crosswalk(out_net, pick.crosswalk_id):
            removed = prune_net_to_single_crosswalk(
                out_net,
                keep_crosswalk_id=pick.crosswalk_id,
                junction_id=pick.junction_id,
            )
            if removed:
                print(
                    f"  [crop] pruned {removed} extra crossing(s); "
                    f"kept only {pick.crosswalk_id}"
                )
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
    used_mode = "junction"

    removed = prune_net_to_single_crosswalk(
        out_net,
        keep_crosswalk_id=pick.crosswalk_id,
        junction_id=pick.junction_id,
    )
    if removed:
        print(
            f"  [crop] pruned {removed} extra crossing(s); "
            f"kept only {pick.crosswalk_id}"
        )
    return used_mode


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
    trim_geometry: bool = False,
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
        trim_geometry=trim_geometry,
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
            "crop_trim_geometry": trim_geometry,
            "single_crosswalk_only": True,
            "crosswalk_id": pick.crosswalk_id,
            "crosswalk_rank": crosswalk_rank,
            "junction_id": pick.junction_id,
            "crossed_edge_ids": list(pick.crossed_edge_ids),
            "approach_edge_ids": list(pick.approach_edge_ids),
        }
    )
    (output_dir / "meta.json").write_text(json_dumps(meta) + "\n", encoding="utf-8")
    return pick
