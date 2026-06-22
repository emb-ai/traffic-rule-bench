"""Orient zone-entry signs onto the directed edge that points INTO the zone.

Residential-zone sign 5.21 marks ENTERING a courtyard from a bigger road. The
scene machinery places the sign on a *directed* SUMO edge and then:
  - braking-spawn puts the ego `d_required` UPSTREAM (predecessors) of the sign,
  - the destination is `hops` edges DOWNSTREAM (successors) of the sign,
so the route is spawn -> sign -> destination. For the route to read as
"big road -> sign -> courtyard", the sign edge must be oriented so its UPSTREAM
side reaches the big road and its DOWNSTREAM side goes into the courtyard.

OSM way direction is arbitrary, so for ~half the scenes the sign edge is oriented
the wrong way (big road is DOWNSTREAM) and the ego drives OUT of the courtyard.
This module detects that via a both-directions BFS to the nearest big road and,
when the orientation is wrong, flips the sign onto the reverse directed edge
(courtyard streets are two-way), remapping the longitudinal offset.

Pure stdlib (xml.etree + collections) so it can be imported by both the async
extraction pipeline and a lightweight standalone re-orientation pass without
pulling in aiohttp/pyproj/shapely.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from typing import Optional

# Sign codes whose route must be oriented "big road -> sign -> courtyard".
# Parameterized so 5.31 (zone speed limit, same entry semantics) can be added.
ORIENT_INTO_ZONE_CODES = {"5.21"}

# Big-road highway types (edge `type` attribute = "highway.<class>"). A side of
# the sign edge that reaches one of these is the "entrance" the ego comes from.
BIG_ROAD_TYPES = {
    "primary", "secondary", "tertiary", "trunk", "unclassified", "living_street",
    "primary_link", "secondary_link", "tertiary_link", "trunk_link",
}

# Non-car edge types — ignored when walking the drivable graph.
_NON_CAR_TYPES = {
    "footway", "steps", "path", "cycleway", "pedestrian", "bridleway",
    "track", "corridor", "elevator", "escalator", "bus_guideway",
}

# A zone-entry sign (5.21) must NOT sit on a through-road; if it does, re-snap it
# onto the nearest courtyard/residential edge (mirrors the extraction's
# SIGN_HIGHWAY_FILTER allowed set).
THROUGH_ROAD_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
RESIDENTIAL_TARGET_TYPES = {"living_street", "service", "residential", "unclassified", "road"}

# A speed-limit sign (3.24) belongs on a real street, NOT a `service` alley/
# dead-end. When it lands on `service`, re-snap to the nearest real road.
RESNAP_TARGET_ROAD_TYPES = THROUGH_ROAD_TYPES | {"residential", "unclassified", "road"}


def _hwy_class(edge_type: Optional[str]) -> Optional[str]:
    """'highway.living_street|psv' -> 'living_street'; None-safe."""
    if not edge_type:
        return None
    t = edge_type.split("|", 1)[0]
    if t.startswith("highway."):
        t = t[len("highway."):]
    return t or None


def _base_edge(eid: str) -> str:
    """Edge id without direction sign and #segment: '-794#0' -> '794'."""
    s = eid[1:] if eid.startswith("-") else eid
    return s.split("#", 1)[0]


def _is_reverse_pair(a: str, b: str) -> bool:
    """True iff a and b are the two directions of the same way segment."""
    return _base_edge(a) == _base_edge(b) and a.startswith("-") != b.startswith("-")


def _reverse_edge_id(eid: str) -> str:
    return eid[1:] if eid.startswith("-") else "-" + eid


def parse_net_graph(net_path: str):
    """Parse a SUMO .net.xml into (etype, length, succ, pred, fromnode, tonode).

    Only real (non-internal) edges. succ/pred are directed adjacency over real
    edges from <connection> elements (internal ':' edges skipped).
    """
    root = ET.parse(net_path).getroot()
    etype: dict = {}
    length: dict = {}
    fromnode: dict = {}
    tonode: dict = {}
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        eid = e.get("id")
        etype[eid] = e.get("type")
        fromnode[eid] = e.get("from")
        tonode[eid] = e.get("to")
        ll = 0.0
        for lane in e.findall("lane"):
            try:
                ll = max(ll, float(lane.get("length", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        length[eid] = ll
    succ: dict = {}
    pred: dict = {}
    for c in root.findall("connection"):
        a, b = c.get("from"), c.get("to")
        if not a or not b or a.startswith(":") or b.startswith(":"):
            continue
        succ.setdefault(a, set()).add(b)
        pred.setdefault(b, set()).add(a)
    return etype, length, succ, pred, fromnode, tonode


def _bfs_hops_to_big_road(start, adj, etype, max_hops):
    """Min #hops from `start` to any big-road edge over directed adjacency `adj`
    (succ -> forward, pred -> backward), or None if none within `max_hops`.

    Skips internal/non-car edges and the reverse-of-current edge at each step so
    a two-way street can't "reach" the big road by U-turning onto the opposite
    carriageway (mirrors the enumerator's `_is_reverse` guard).
    """
    seen = {start}
    q = deque([(start, 0)])
    while q:
        n, d = q.popleft()
        if d > 0 and _hwy_class(etype.get(n)) in BIG_ROAD_TYPES:
            return d
        if d >= max_hops:
            continue
        for m in adj.get(n, ()):
            if m in seen or m.startswith(":"):
                continue
            if _hwy_class(etype.get(m)) in _NON_CAR_TYPES:
                continue
            if _is_reverse_pair(n, m):
                continue
            seen.add(m)
            q.append((m, d + 1))
    return None


def _resolve_reverse_edge(eid, etype, fromnode, tonode):
    """Reverse directed edge id (swapped from/to nodes) or None if one-way.

    Tries the literal '-E'/'E' first (the common case), then falls back to any
    edge of the same base way running between the same two junctions in the
    opposite direction (handles asymmetric #segment splits).
    """
    rev = _reverse_edge_id(eid)
    if (rev in etype
            and fromnode.get(rev) == tonode.get(eid)
            and tonode.get(rev) == fromnode.get(eid)):
        return rev
    base = _base_edge(eid)
    for cand in etype:
        if cand == eid:
            continue
        if (_base_edge(cand) == base
                and fromnode.get(cand) == tonode.get(eid)
                and tonode.get(cand) == fromnode.get(eid)):
            return cand
    return None


def orient_sign_edge(net_path, road_id, distance_from_start, sign_code,
                     codes=ORIENT_INTO_ZONE_CODES, max_hops=6):
    """Ensure a zone-entry sign sits on the directed edge pointing INTO the zone.

    Returns (new_road_id, new_distance_from_start, info). For codes not in
    `codes`, or when reorientation isn't warranted/possible, returns the inputs
    unchanged; `info["decision"]` explains why. `info` also carries dist_fwd /
    dist_bwd (hops to the nearest big road each way) for reporting.

    Decision:
      - big road FORWARD-closer  -> flip onto the reverse edge (remap offset),
      - big road BACKWARD-closer -> keep (already correct),
      - tie (both finite, equal) -> keep, flagged "ambiguous",
      - neither side near a big road -> keep, "no_big_road",
      - no reverse edge (one-way) -> keep, "no_reverse_edge".
    """
    info = {
        "decision": None,
        "dist_fwd": None,
        "dist_bwd": None,
        "from_road_id": road_id,
        "from_distance": float(distance_from_start),
    }
    if str(sign_code) not in codes:
        info["decision"] = "skip_code"
        return road_id, distance_from_start, info

    try:
        etype, length, succ, pred, fromnode, tonode = parse_net_graph(net_path)
    except Exception as e:  # malformed/missing net -> leave scene untouched
        info["decision"] = "parse_error"
        info["error"] = str(e)
        return road_id, distance_from_start, info

    if road_id not in etype:
        info["decision"] = "edge_not_found"
        return road_id, distance_from_start, info

    df = _bfs_hops_to_big_road(road_id, succ, etype, max_hops)  # big road ahead
    db = _bfs_hops_to_big_road(road_id, pred, etype, max_hops)  # big road behind
    info["dist_fwd"], info["dist_bwd"] = df, db

    if df is None and db is None:
        info["decision"] = "no_big_road"
        return road_id, distance_from_start, info

    # Wrong orientation: big road is DOWNSTREAM (ego would drive out) -> flip.
    if df is not None and (db is None or df < db):
        rev = _resolve_reverse_edge(road_id, etype, fromnode, tonode)
        if rev is None:
            info["decision"] = "no_reverse_edge"
            return road_id, distance_from_start, info
        L_e = length.get(road_id, 0.0) or 0.0
        L_r = length.get(rev, L_e) or L_e
        s = float(distance_from_start)
        frac = (s / L_e) if L_e > 0 else 0.0
        s_new = L_r * (1.0 - frac)
        s_new = max(0.1, min(s_new, max(0.1, L_r - 0.1)))
        info["decision"] = "flip"
        info["to_road_id"] = rev
        info["to_distance"] = s_new
        return rev, s_new, info

    # Already correct (big road behind), or a tie we leave alone.
    info["decision"] = "ambiguous" if (df is not None and db is not None and df == db) else "keep"
    return road_id, distance_from_start, info


def _edge_shapes(net_path):
    """{edge_id: [(x,y), ...]} lane-0 polyline for real (non-internal) edges."""
    root = ET.parse(net_path).getroot()
    shapes = {}
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        lane = e.find("lane")
        if lane is not None and lane.get("shape"):
            try:
                shapes[e.get("id")] = [tuple(map(float, p.split(",")))
                                       for p in lane.get("shape").split()]
            except ValueError:
                pass
    return shapes


def _pos_along(shape, s):
    """Point at arc-length `s` along a polyline."""
    import math
    acc = 0.0
    for i in range(len(shape) - 1):
        (x0, y0), (x1, y1) = shape[i], shape[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= s:
            t = (s - acc) / seg if seg > 0 else 0.0
            return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        acc += seg
    return shape[-1]


def _project_point(p, shape):
    """(min distance from point `p` to the polyline, arc-length offset of the
    projection)."""
    import math
    px, py = p
    best_d, best_off, acc = float("inf"), 0.0, 0.0
    for i in range(len(shape) - 1):
        ax, ay = shape[i]
        bx, by = shape[i + 1]
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        cx, cy = ax + t * dx, ay + t * dy
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d, best_off = d, acc + t * seg
        acc += seg
    return best_d, best_off


def resnap_to_residential(net_path, road_id, distance_from_start, sign_code,
                          codes=ORIENT_INTO_ZONE_CODES, max_dist=40.0):
    """Move a zone-entry sign off a through-road onto the nearest courtyard/
    residential edge (the road a 5.21 sign actually belongs on).

    Returns (new_road_id, new_distance_from_start, info). No-op (inputs unchanged)
    for non-listed codes, signs already on a residential edge, or when no
    residential edge is within `max_dist` metres; `info["decision"]` explains.
    Run BEFORE orient_sign_edge (which then orients the residential edge inbound).
    """
    info = {"decision": None, "from_road_id": road_id}
    if str(sign_code) not in codes:
        info["decision"] = "skip_code"
        return road_id, distance_from_start, info
    try:
        etype = parse_net_graph(net_path)[0]
        shapes = _edge_shapes(net_path)
    except Exception as e:
        info["decision"] = "parse_error"
        info["error"] = str(e)
        return road_id, distance_from_start, info
    if _hwy_class(etype.get(road_id)) not in THROUGH_ROAD_TYPES:
        info["decision"] = "already_residential"
        return road_id, distance_from_start, info
    if road_id not in shapes:
        info["decision"] = "edge_not_found"
        return road_id, distance_from_start, info
    sign_pos = _pos_along(shapes[road_id], float(distance_from_start))
    best = None  # (dist, edge_id, offset)
    for eid, shp in shapes.items():
        if _hwy_class(etype.get(eid)) not in RESIDENTIAL_TARGET_TYPES or len(shp) < 2:
            continue
        d, off = _project_point(sign_pos, shp)
        if best is None or d < best[0]:
            best = (d, eid, off)
    if best is None or best[0] > max_dist:
        info["decision"] = "no_residential_nearby"
        info["nearest_m"] = round(best[0], 1) if best else None
        return road_id, distance_from_start, info
    info["decision"] = "resnap"
    info["to_road_id"] = best[1]
    info["to_distance"] = round(best[2], 3)
    info["dist_m"] = round(best[0], 1)
    return best[1], max(0.1, best[2]), info


def resnap_off_service(net_path, road_id, distance_from_start, max_dist=40.0):
    """Move a speed-limit sign (3.24) off a `service` road onto the nearest real
    street (through-road / residential / unclassified). No orientation change.

    Returns (new_road_id, new_distance_from_start, info). No-op when the sign is
    not on a `service` edge or no real road is within `max_dist`.
    """
    info = {"decision": None, "from_road_id": road_id}
    try:
        etype = parse_net_graph(net_path)[0]
        shapes = _edge_shapes(net_path)
    except Exception as e:
        info["decision"] = "parse_error"
        info["error"] = str(e)
        return road_id, distance_from_start, info
    if _hwy_class(etype.get(road_id)) != "service":
        info["decision"] = "not_service"
        return road_id, distance_from_start, info
    if road_id not in shapes:
        info["decision"] = "edge_not_found"
        return road_id, distance_from_start, info
    sign_pos = _pos_along(shapes[road_id], float(distance_from_start))
    best = None  # (dist, edge_id, offset)
    for eid, shp in shapes.items():
        if _hwy_class(etype.get(eid)) not in RESNAP_TARGET_ROAD_TYPES or len(shp) < 2:
            continue
        d, off = _project_point(sign_pos, shp)
        if best is None or d < best[0]:
            best = (d, eid, off)
    if best is None or best[0] > max_dist:
        info["decision"] = "no_road_nearby"
        info["nearest_m"] = round(best[0], 1) if best else None
        return road_id, distance_from_start, info
    info["decision"] = "resnap"
    info["to_road_id"] = best[1]
    info["to_distance"] = round(best[2], 3)
    info["dist_m"] = round(best[0], 1)
    return best[1], max(0.1, best[2]), info
