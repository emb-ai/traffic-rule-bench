"""Inject pedestrian crosswalks into SUMO segment networks.

This module provides functions to add mid-block pedestrian crossings to
segment maps that originally have no pedestrian infrastructure. The approach:

1. Split the main road edge(s) at the crosswalk position to create a junction
2. Add sidewalk lanes to the edges
3. Define a crossing at the new junction
4. Run netconvert to generate walkingareas and connections
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Crosswalk position presets (meters from segment start)
POSITION_NEAR_START_M = 40.0
POSITION_NEAR_END_OFFSET_M = 40.0  # from segment end
MIN_POSITION_FROM_START_M = 30.0
MIN_POSITION_FROM_END_M = 40.0

# Default infrastructure widths
DEFAULT_CROSSWALK_WIDTH_M = 4.0
DEFAULT_SIDEWALK_WIDTH_M = 2.0


@dataclass
class CrosswalkInjection:
    """Configuration for injecting a crosswalk into a segment network."""

    source_net: Path
    crosswalk_position_m: float
    edge_ids: Tuple[str, ...]  # edges to split and cross
    crosswalk_width_m: float = DEFAULT_CROSSWALK_WIDTH_M
    sidewalk_width_m: float = DEFAULT_SIDEWALK_WIDTH_M
    priority: bool = True  # zebra crossing (vehicles must yield)

    @property
    def crosswalk_node_id(self) -> str:
        """Generate a unique node ID for the crosswalk junction."""
        pos_int = int(self.crosswalk_position_m)
        return f"cw_node_{pos_int}"


@dataclass
class CrosswalkInjectionResult:
    """Result of crosswalk injection."""

    success: bool
    output_net: Optional[Path] = None
    crosswalk_node_id: Optional[str] = None
    crosswalk_edge_id: Optional[str] = None
    crossed_edge_ids: Tuple[str, ...] = field(default_factory=tuple)
    error: Optional[str] = None


def _find_netconvert() -> str:
    """Find netconvert executable."""
    for path in (
        shutil.which("netconvert"),
        str(Path.home() / ".local" / "bin" / "netconvert"),
        "/usr/local/bin/netconvert",
        "/usr/bin/netconvert",
    ):
        if path and Path(path).exists():
            return path
    raise FileNotFoundError(
        "netconvert not found. Install SUMO or add netconvert to PATH."
    )


def identify_main_edges(net_path: Path) -> List[Dict]:
    """Find main road edges in a segment network.

    Returns edges that are:
    - Not internal (don't start with ':')
    - Not pedestrian-only
    - Have vehicle lanes

    Returns:
        List of dicts with edge_id, length_m, lane_count, from_node, to_node
    """
    root = ET.parse(net_path).getroot()
    edges: List[Dict] = []

    for edge_el in root.findall("edge"):
        edge_id = edge_el.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge_el.get("function") in ("internal", "crossing", "walkingarea"):
            continue

        lanes = edge_el.findall("lane")
        vehicle_lanes = []
        for lane in lanes:
            allow = (lane.get("allow") or "").strip()
            disallow = (lane.get("disallow") or "").strip()
            if allow and all(c == "pedestrian" for c in allow.split()):
                continue
            vehicle_lanes.append(lane)

        if not vehicle_lanes:
            continue

        # Get edge length from first vehicle lane
        lane = vehicle_lanes[0]
        length = float(lane.get("length", 0) or 0)
        if length <= 0:
            shape = (lane.get("shape") or "").strip().split()
            coords = [tuple(map(float, p.split(","))) for p in shape if "," in p]
            if len(coords) >= 2:
                length = sum(
                    ((coords[i + 1][0] - coords[i][0]) ** 2 +
                     (coords[i + 1][1] - coords[i][1]) ** 2) ** 0.5
                    for i in range(len(coords) - 1)
                )

        edges.append({
            "edge_id": edge_id,
            "length_m": length,
            "lane_count": len(vehicle_lanes),
            "from_node": edge_el.get("from", ""),
            "to_node": edge_el.get("to", ""),
        })

    return edges


def find_paired_edges(edges: List[Dict]) -> List[Tuple[str, str]]:
    """Find pairs of edges that are reverse directions of the same road.

    SUMO convention: edge "123#0" and "-123#0" are the same road, opposite directions.

    Returns:
        List of (forward_edge_id, backward_edge_id) tuples
    """
    edge_ids = {e["edge_id"] for e in edges}
    pairs: List[Tuple[str, str]] = []
    seen: set = set()

    for edge in edges:
        eid = edge["edge_id"]
        if eid in seen:
            continue

        # Check for reverse direction
        if eid.startswith("-"):
            reverse_id = eid[1:]
        else:
            reverse_id = "-" + eid

        if reverse_id in edge_ids:
            # Use the non-negative as "forward"
            if eid.startswith("-"):
                pairs.append((reverse_id, eid))
            else:
                pairs.append((eid, reverse_id))
            seen.add(eid)
            seen.add(reverse_id)

    return pairs


def calculate_crosswalk_positions(
    edge_length_m: float,
    positions: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate crosswalk positions for a segment.

    Args:
        edge_length_m: Length of the edge in meters
        positions: List of position names to calculate (default: all)

    Returns:
        Dict mapping position name to distance from edge start (meters)
    """
    if positions is None:
        positions = ["near_start", "middle", "near_end"]

    result: Dict[str, float] = {}

    for pos_name in positions:
        if pos_name == "near_start":
            pos_m = max(POSITION_NEAR_START_M, MIN_POSITION_FROM_START_M)
        elif pos_name == "near_end":
            pos_m = edge_length_m - POSITION_NEAR_END_OFFSET_M
            pos_m = min(pos_m, edge_length_m - MIN_POSITION_FROM_END_M)
        elif pos_name == "middle":
            pos_m = edge_length_m / 2.0
        else:
            raise ValueError(f"Unknown position name: {pos_name}")

        # Clamp to valid range
        pos_m = max(MIN_POSITION_FROM_START_M, pos_m)
        pos_m = min(edge_length_m - MIN_POSITION_FROM_END_M, pos_m)

        if pos_m > MIN_POSITION_FROM_START_M:
            result[pos_name] = pos_m

    return result


def get_post_split_edge_ids(
    edge_ids: Tuple[str, ...],
    position_m: float,
) -> Tuple[str, ...]:
    """Get the edge IDs that will exist after the split.

    After splitting edge "X" at position P, SUMO creates:
    - "X" (before split, from start to P)
    - "X.P" (after split, from P to end)

    For backward edges ("-X") split at -P:
    - "-X" (before split, from start to P) 
    - "-X.-P" (after split, from P to end)
    """
    result = []
    pos_int = int(position_m)
    
    for edge_id in edge_ids:
        if edge_id.startswith("-"):
            # Backward edge: add original and .-(pos) suffix
            result.append(edge_id)
            result.append(f"{edge_id}.-{pos_int}")
        else:
            # Forward edge: add original and .(pos) suffix
            result.append(edge_id)
            result.append(f"{edge_id}.{pos_int}")
    
    return tuple(result)


def generate_split_xml(
    edge_ids: Tuple[str, ...],
    position_m: float,
    node_id: str,
    sidewalk_width_m: float = DEFAULT_SIDEWALK_WIDTH_M,
) -> str:
    """Generate PlainXML edges file with split elements.

    For bidirectional roads, both edges must be split at the same node.
    The backward edge uses negative position (from the end).
    """
    root = ET.Element("edges")
    pos_int = int(position_m)

    for i, edge_id in enumerate(edge_ids):
        edge_el = ET.SubElement(root, "edge", id=edge_id)

        # For backward edges (starting with '-'), split from the end
        if edge_id.startswith("-"):
            split_pos = f"-{position_m:.1f}"
            id_after = f"{edge_id}.-{pos_int}"
        else:
            split_pos = f"{position_m:.1f}"
            id_after = f"{edge_id}.{pos_int}"

        # Add split element with explicit idAfter to ensure correct naming
        split_el = ET.SubElement(
            edge_el, "split",
            pos=split_pos,
            id=node_id,
            idAfter=id_after,
        )

        # Add sidewalk width attribute to the edge
        edge_el.set("sidewalkWidth", f"{sidewalk_width_m:.1f}")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def generate_node_xml(
    node_id: str,
    node_type: str = "priority",
) -> str:
    """Generate PlainXML nodes file to set junction type."""
    root = ET.Element("nodes")
    node_el = ET.SubElement(root, "node", id=node_id, type=node_type)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def generate_crossing_xml(
    node_id: str,
    edge_ids: Tuple[str, ...],
    position_m: float,
    priority: bool = True,
    width_m: float = DEFAULT_CROSSWALK_WIDTH_M,
) -> str:
    """Generate PlainXML connections file with crossing and through-connections.

    After the split, the crossing must reference ALL edges passing through
    the node (both incoming and outgoing segments).
    We also need to add through-connections to make the junction a proper
    priority junction instead of a dead-end.
    """
    root = ET.Element("connections")
    pos_int = int(position_m)

    # Add through-connections for each edge pair
    # This ensures vehicles can pass through the junction
    for edge_id in edge_ids:
        if edge_id.startswith("-"):
            id_after = f"{edge_id}.-{pos_int}"
        else:
            id_after = f"{edge_id}.{pos_int}"
        
        # Connect incoming edge to outgoing edge
        conn_el = ET.SubElement(
            root,
            "connection",
            **{"from": edge_id, "to": id_after}
        )

    # Get all edge IDs that will exist after the split for crossing definition
    post_split_edges = get_post_split_edge_ids(edge_ids, position_m)
    edges_str = " ".join(post_split_edges)
    
    crossing_el = ET.SubElement(
        root,
        "crossing",
        node=node_id,
        edges=edges_str,
        priority="true" if priority else "false",
        width=f"{width_m:.1f}",
    )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def inject_crosswalk(
    injection: CrosswalkInjection,
    output_net: Path,
) -> CrosswalkInjectionResult:
    """Inject a crosswalk into a segment network.

    Creates a new network file with a pedestrian crossing infrastructure at the
    specified position. Uses a two-step approach:
    1. Add sidewalks and disallow pedestrians on vehicle lanes
    2. Split edges and add connections for both pedestrian and vehicle lanes

    Note: The SUMO crossing element may not be created if the road geometry
    is too narrow (single carriageway). The infrastructure (junction, sidewalks,
    walkingareas) will still be valid for MetaDrive pedestrian simulation.
    """
    if not injection.source_net.is_file():
        return CrosswalkInjectionResult(
            success=False,
            error=f"Source network not found: {injection.source_net}",
        )

    output_net.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="crosswalk_inject_") as tmpdir:
        tmp = Path(tmpdir)
        pos_int = int(injection.crosswalk_position_m)

        # ===== STEP 1: Add sidewalks to all edges =====
        sidewalk_xml = generate_sidewalk_xml(
            injection.edge_ids,
            injection.sidewalk_width_m,
        )
        sidewalk_file = tmp / "sidewalk.edg.xml"
        sidewalk_file.write_text(sidewalk_xml, encoding="utf-8")

        step1_out = tmp / "step1.net.xml"
        cmd1 = [
            _find_netconvert(),
            "--sumo-net-file", str(injection.source_net),
            "--edge-files", str(sidewalk_file),
            "--walkingareas",
            "--no-turnarounds", "true",
            "-o", str(step1_out),
        ]

        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0 or not step1_out.is_file():
            return CrosswalkInjectionResult(
                success=False,
                error=f"Step 1 (sidewalks) failed: {result1.stderr or result1.stdout}",
            )

        # ===== STEP 2: Split edges and add connections + crossing =====
        # Generate split XML (without sidewalk width - already added)
        split_xml_content = _generate_split_only_xml(
            injection.edge_ids,
            injection.crosswalk_position_m,
            injection.crosswalk_node_id,
        )
        split_file = tmp / "split.edg.xml"
        split_file.write_text(split_xml_content, encoding="utf-8")

        # Generate connections for BOTH lanes (0=pedestrian, 1=vehicle)
        conn_xml = generate_multilane_connection_xml(
            injection.edge_ids,
            injection.crosswalk_position_m,
            injection.crosswalk_node_id,
            injection.crosswalk_width_m,
            injection.priority,
        )
        conn_file = tmp / "conn.con.xml"
        conn_file.write_text(conn_xml, encoding="utf-8")

        step2_out = tmp / "step2.net.xml"
        cmd2 = [
            _find_netconvert(),
            "--sumo-net-file", str(step1_out),
            "--edge-files", str(split_file),
            "--connection-files", str(conn_file),
            "--walkingareas",
            "--no-turnarounds", "true",
            "-o", str(step2_out),
        ]

        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0 or not step2_out.is_file():
            return CrosswalkInjectionResult(
                success=False,
                error=f"Step 2 (split) failed: {result2.stderr or result2.stdout}",
            )

        # Validate the output - check for junction existence (crossing is optional)
        validation = validate_crosswalk_net(step2_out, injection.crosswalk_node_id)
        
        # Check if the junction exists (not just crossing)
        if not _junction_exists(step2_out, injection.crosswalk_node_id):
            return CrosswalkInjectionResult(
                success=False,
                error=f"Junction {injection.crosswalk_node_id} not created",
            )

        # Copy to final destination
        shutil.copy2(step2_out, output_net)

    return CrosswalkInjectionResult(
        success=True,
        output_net=output_net,
        crosswalk_node_id=injection.crosswalk_node_id,
        crosswalk_edge_id=validation.get("crossing_edge_id"),  # May be None
        crossed_edge_ids=injection.edge_ids,
    )


def generate_sidewalk_xml(
    edge_ids: Tuple[str, ...],
    sidewalk_width_m: float = DEFAULT_SIDEWALK_WIDTH_M,
) -> str:
    """Generate XML to add sidewalks to edges and disallow pedestrians on vehicle lanes."""
    root = ET.Element("edges")

    for edge_id in edge_ids:
        edge_el = ET.SubElement(
            root, "edge",
            id=edge_id,
            disallow="pedestrian",
            sidewalkWidth=f"{sidewalk_width_m:.1f}",
        )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _generate_split_only_xml(
    edge_ids: Tuple[str, ...],
    position_m: float,
    node_id: str,
) -> str:
    """Generate XML to split edges at specified position (no sidewalk width)."""
    root = ET.Element("edges")
    pos_int = int(position_m)

    for edge_id in edge_ids:
        edge_el = ET.SubElement(root, "edge", id=edge_id)

        if edge_id.startswith("-"):
            split_pos = f"-{position_m:.1f}"
            id_after = f"{edge_id}.-{pos_int}"
        else:
            split_pos = f"{position_m:.1f}"
            id_after = f"{edge_id}.{pos_int}"

        ET.SubElement(edge_el, "split", pos=split_pos, id=node_id, idAfter=id_after)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def generate_multilane_connection_xml(
    edge_ids: Tuple[str, ...],
    position_m: float,
    node_id: str,
    crossing_width_m: float = DEFAULT_CROSSWALK_WIDTH_M,
    priority: bool = True,
) -> str:
    """Generate connections for all lanes (pedestrian + vehicle) and crossing."""
    root = ET.Element("connections")
    pos_int = int(position_m)

    # Add connections for both lanes (0=pedestrian sidewalk, 1=vehicle)
    for edge_id in edge_ids:
        if edge_id.startswith("-"):
            id_after = f"{edge_id}.-{pos_int}"
        else:
            id_after = f"{edge_id}.{pos_int}"

        # Lane 0 (pedestrian sidewalk)
        ET.SubElement(root, "connection", **{
            "from": edge_id, "to": id_after,
            "fromLane": "0", "toLane": "0"
        })
        # Lane 1 (vehicle)
        ET.SubElement(root, "connection", **{
            "from": edge_id, "to": id_after,
            "fromLane": "1", "toLane": "1"
        })

    # Add crossing (may be discarded by netconvert if topology invalid)
    post_split_edges = get_post_split_edge_ids(edge_ids, position_m)
    edges_str = " ".join(post_split_edges)
    ET.SubElement(
        root, "crossing",
        node=node_id,
        edges=edges_str,
        priority="true" if priority else "false",
        width=f"{crossing_width_m:.1f}",
    )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _junction_exists(net_path: Path, junction_id: str) -> bool:
    """Check if a junction exists in the network."""
    root = ET.parse(net_path).getroot()
    for junction in root.findall("junction"):
        if junction.get("id") == junction_id:
            return True
    return False


def validate_crosswalk_net(
    net_path: Path,
    expected_node_id: Optional[str] = None,
) -> Dict:
    """Validate that a network has proper crosswalk infrastructure.

    Returns:
        Dict with validation results:
        - has_crossing: bool
        - crossing_count: int
        - crossing_edge_id: str or None
        - has_walkingareas: bool
        - walkingarea_count: int
    """
    root = ET.parse(net_path).getroot()

    crossings = []
    walkingareas = []

    for edge in root.findall("edge"):
        func = edge.get("function")
        if func == "crossing":
            crossings.append(edge.get("id"))
        elif func == "walkingarea":
            walkingareas.append(edge.get("id"))

    crossing_edge_id = None
    if crossings:
        if expected_node_id:
            # Find crossing that matches expected node
            for cid in crossings:
                if expected_node_id in cid:
                    crossing_edge_id = cid
                    break
        if not crossing_edge_id:
            crossing_edge_id = crossings[0]

    return {
        "has_crossing": len(crossings) > 0,
        "crossing_count": len(crossings),
        "crossing_edge_id": crossing_edge_id,
        "crossing_edge_ids": crossings,
        "has_walkingareas": len(walkingareas) > 0,
        "walkingarea_count": len(walkingareas),
    }


def count_net_crossings(net_path: Path) -> int:
    """Count the number of crossing edges in a network."""
    if not net_path.is_file():
        return 0
    root = ET.parse(net_path).getroot()
    return sum(
        1 for edge in root.findall("edge")
        if edge.get("function") == "crossing"
    )
