#!/usr/bin/env python3
"""Review and correctly place 4.2.1/4.2.2/4.2.3 signs on SUMO scenes.

A 4.2.x sign defines the OBSTACLE LANE: cones are placed on it, the ego must
change lanes to the prescribed side. The original extraction pipeline snapped
the sign to the nearest highway-way ignoring the sign type, so the lane is
not recorded and the edge itself is often unusable (single-lane / no passenger).

SUMO convention: lane 0 is the rightmost; physically further right = SMALLER index.
  4.2.1 (detour on the right)  -> obstacle lane must have a neighbor with a smaller index;
  4.2.2 (detour on the left)   -> neighbor with a larger index;
  4.2.3 (either side)  -> any neighbor.

The tool writes to meta.json (see _resolve_detour_lane_and_s in envs/sumo_env.py):
  sign_lane_index, sign_s, detour_placed_by, detour_tool_version,
  excluded/excluded_reason, resnapped/road_id_orig/distance_from_start_orig.
On re-snap, road_id/distance_from_start are OVERWRITTEN (legacy consumers
keep working), originals are preserved in *_orig.

Interactive (default):
  click a lane — move the sign;  n/p — next/previous scene;
  a — auto-suggest;  x — toggle excluded;  w — save meta.json;
  r — reload from disk;  q — quit.

Batch:
  python3 detour_scene_editor.py --scenes-root ../../scenes \
      --codes 4.2.1,4.2.2,4.2.3 --batch --report report_detour.json
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import sumolib
from sumolib import geomhelper

DETOUR_CODES = ("4.2.1", "4.2.2", "4.2.3")
TOOL_VERSION = 1
SIGN_TO_OBSTACLE = 3.5          # detour_sign.py: cones 3.5 m past the sign
ZONE_BEFORE = 30.0              # violation zone starts 30 m before the cones
EDGE_TAIL_MARGIN = 12.0         # do not place the sign within 12 m of the edge end
SHORT_RUNWAY_S = 40.0           # sign_s below this — little run-up before the zone
CLICK_TOLERANCE_M = 4.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def polyline_length(shape):
    return sum(math.dist(shape[i], shape[i + 1]) for i in range(len(shape) - 1))


def point_at(shape, s):
    """Point and heading (rad) on the polyline at longitudinal position s."""
    if len(shape) < 2:
        x, y = shape[0]
        return x, y, 0.0
    acc = 0.0
    for i in range(len(shape) - 1):
        (x0, y0), (x1, y1) = shape[i], shape[i + 1]
        seg = math.dist((x0, y0), (x1, y1))
        if seg <= 0:
            continue
        if acc + seg >= s or i == len(shape) - 2:
            t = min(max((s - acc) / seg, 0.0), 1.0)
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0), math.atan2(y1 - y0, x1 - x0)
        acc += seg
    x, y = shape[-1]
    return x, y, 0.0


def project_on_edge(edge, x, y):
    """(s, dist): longitudinal position of the point's projection onto the edge axis, and distance.

    s is rescaled from the polyline length to edge.getLength() (they may
    differ slightly due to straightening)."""
    shape = edge.getShape()
    off = geomhelper.polygonOffsetWithMinimumDistanceToPoint((x, y), shape)
    dist = geomhelper.distancePointToPolygon((x, y), shape)
    plen = polyline_length(shape)
    s = off / plen * edge.getLength() if plen > 0 else 0.0
    return max(0.0, min(s, edge.getLength())), dist


def heading_diff_deg(a, b):
    d = math.degrees(abs(a - b)) % 360.0
    return min(d, 360.0 - d)


def lane_polygon(shape, width):
    """Lane polygon: centerline offset perpendicularly by ±width/2.

    Per-segment approximation (no proper joints) — good enough for review."""
    left, right = [], []
    for i in range(len(shape) - 1):
        (x0, y0), (x1, y1) = shape[i], shape[i + 1]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        if norm <= 0:
            continue
        nx, ny = -dy / norm * width / 2, dx / norm * width / 2
        left += [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny)]
        right += [(x0 - nx, y0 - ny), (x1 - nx, y1 - ny)]
    return left + right[::-1]


# ---------------------------------------------------------------------------
# Feasibility core (pure functions — used by both batch and interactive modes)
# ---------------------------------------------------------------------------

def drivable_lane_indices(edge):
    return {l.getIndex() for l in edge.getLanes() if l.allows("passenger")}


def valid_obstacle_lane_indices(edge, code):
    """Lane indices where the obstacle may be placed for the given code:
    among passenger lanes there must be a neighbor on the prescribed side."""
    drivable = drivable_lane_indices(edge)
    n = len(edge.getLanes())
    out = []
    for i in sorted(drivable):
        has_right = any(j in drivable for j in range(0, i))
        has_left = any(j in drivable for j in range(i + 1, n))
        if code == "4.2.1" and has_right:
            out.append(i)
        elif code == "4.2.2" and has_left:
            out.append(i)
        elif code == "4.2.3" and (has_right or has_left):
            out.append(i)
    return out


def allowed_target_lane_indices(edge, code, obstacle_idx):
    """Lanes onto which detour from the obstacle lane is allowed."""
    drivable = drivable_lane_indices(edge)
    n = len(edge.getLanes())
    out = set()
    if code in ("4.2.1", "4.2.3"):
        out |= {j for j in range(0, obstacle_idx) if j in drivable}
    if code in ("4.2.2", "4.2.3"):
        out |= {j for j in range(obstacle_idx + 1, n) if j in drivable}
    return out


def preferred_obstacle_lane(edge, code):
    valid = valid_obstacle_lane_indices(edge, code)
    if not valid:
        return None
    if code in ("4.2.1", "4.2.2"):
        # 4.2.1: smallest valid (usually 1 — detour into the rightmost lane);
        # 4.2.2: 0 if valid (obstacle on the right, detour to the left), else min.
        return valid[0]
    # 4.2.3: middle lane if present; else smallest i >= 1; else first.
    n = len(edge.getLanes())
    if n >= 3 and (n // 2) in valid:
        return n // 2
    ge1 = [i for i in valid if i >= 1]
    return ge1[0] if ge1 else valid[0]


def clamp_sign_s(s_proj, edge_len, target_sign_s):
    """Sign no closer than EDGE_TAIL_MARGIN to the edge end; floor is target_sign_s
    from the start (room for the lane change: zone_start = sign_s + 3.5 - 30)."""
    upper = max(0.5, edge_len - EDGE_TAIL_MARGIN)
    lower = min(float(target_sign_s), upper)
    return max(min(s_proj, upper), lower)


def _base_way(edge_id):
    return edge_id.lstrip("-").split("#")[0]


def _same_direction(edge_id_a, edge_id_b):
    return (_base_way(edge_id_a) == _base_way(edge_id_b)
            and edge_id_a.startswith("-") == edge_id_b.startswith("-"))


# ---------------------------------------------------------------------------
# Auto-suggest
# ---------------------------------------------------------------------------

@dataclass
class Params:
    radius: float = 75.0
    min_edge_len: float = 45.0
    target_sign_s: float = 60.0


def suggest_placement(net, meta, code, params):
    """Pick (road_id, sign_lane_index, sign_s) or a reason for exclusion."""
    x, y = net.convertLonLat2XY(float(meta["longitude"]), float(meta["latitude"]))
    orig_id = str(meta.get("road_id", ""))
    orig_edge = net.getEdge(orig_id) if net.hasEdge(orig_id) else None
    orig_heading = None
    if orig_edge is not None:
        s0 = min(float(meta.get("distance_from_start", 0.0) or 0.0),
                 orig_edge.getLength())
        _, _, orig_heading = point_at(orig_edge.getShape(), s0)

    near = net.getNeighboringEdges(x, y, params.radius)
    normal = [(e, d) for e, d in near if not e.getFunction()]
    with_passenger = [(e, d) for e, d in normal if drivable_lane_indices(e)]
    if not with_passenger:
        return {"excluded_reason": "no_passenger_edge_in_radius"}
    feasible = [(e, d) for e, d in with_passenger
                if valid_obstacle_lane_indices(e, code)]
    if not feasible:
        return {"excluded_reason": "no_multilane_edge_in_radius"}
    long_enough = [(e, d) for e, d in feasible
                   if e.getLength() >= params.min_edge_len]
    if not long_enough:
        return {"excluded_reason": "edge_too_short"}

    def rank(item):
        e, d = item
        eid = e.getID()
        if eid == orig_id:
            cls = 0
        elif orig_id and _same_direction(eid, orig_id):
            cls = 1
        elif orig_heading is not None:
            s_proj, _ = project_on_edge(e, x, y)
            _, _, h = point_at(e.getShape(), s_proj)
            cls = 2 if heading_diff_deg(h, orig_heading) <= 60.0 else 3
        else:
            cls = 2
        # Edges with no continuation (clipped by the OSM fragment boundary) go last:
        # after the detour the route must continue, otherwise the ego hits the
        # edge end and the scene is rejected as undrivable.
        dead_end = not any(not o.getFunction() and drivable_lane_indices(o)
                           for o in e.getOutgoing())
        return (dead_end, cls, d)

    edge, _ = min(long_enough, key=rank)
    s_proj, _ = project_on_edge(edge, x, y)
    sign_s = clamp_sign_s(s_proj, edge.getLength(), params.target_sign_s)
    return {
        "road_id": edge.getID(),
        "sign_lane_index": preferred_obstacle_lane(edge, code),
        "sign_s": round(sign_s, 2),
        "resnapped": edge.getID() != orig_id,
        "short_runway": sign_s < SHORT_RUNWAY_S,
    }


def apply_placement(meta, placement):
    """Write the result into meta (dict is modified in place)."""
    if "excluded_reason" in placement:
        meta["excluded"] = True
        meta["excluded_reason"] = placement["excluded_reason"]
        meta["detour_placed_by"] = "auto"
        meta["detour_tool_version"] = TOOL_VERSION
        return
    if placement["resnapped"] and str(meta.get("road_id")) != placement["road_id"]:
        meta.setdefault("road_id_orig", meta.get("road_id"))
        meta.setdefault("distance_from_start_orig", meta.get("distance_from_start"))
        meta["resnapped"] = True
    meta["road_id"] = placement["road_id"]
    meta["distance_from_start"] = placement["sign_s"]
    meta["sign_lane_index"] = int(placement["sign_lane_index"])
    meta["sign_s"] = placement["sign_s"]
    meta["detour_placed_by"] = placement.get("placed_by", "auto")
    meta["detour_tool_version"] = TOOL_VERSION
    meta.pop("excluded", None)
    meta.pop("excluded_reason", None)


# ---------------------------------------------------------------------------
# Scenes on disk
# ---------------------------------------------------------------------------

def iter_scenes(scenes_root, codes, only_scene=None, only_unplaced=False):
    for code in codes:
        code_dir = os.path.join(scenes_root, code)
        if not os.path.isdir(code_dir):
            print(f"[warn] no directory {code_dir}", file=sys.stderr)
            continue
        for name in sorted(os.listdir(code_dir)):
            scene_dir = os.path.join(code_dir, name)
            meta_path = os.path.join(scene_dir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            if only_scene and name != only_scene and name != f"sign_{only_scene}":
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if only_unplaced and (meta.get("detour_placed_by")
                                  or meta.get("excluded")):
                continue
            yield code, scene_dir, meta


def save_meta(scene_dir, meta):
    path = os.path.join(scene_dir, "meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def net_path_for(scene_dir, meta):
    return os.path.join(scene_dir, meta["net_file"])


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def run_batch(args, params):
    codes = args.codes
    report = {c: {"total": 0, "kept_edge": 0, "resnapped": 0,
                  "short_runway_warn": 0, "excluded": {}} for c in codes}
    for code, scene_dir, meta in iter_scenes(
            args.scenes_root, codes, args.scene, args.only_unplaced):
        rep = report[code]
        rep["total"] += 1
        try:
            net = sumolib.net.readNet(net_path_for(scene_dir, meta))
            placement = suggest_placement(net, meta, code, params)
        except Exception as e:
            placement = {"excluded_reason": f"error:{type(e).__name__}"}
        apply_placement(meta, placement)
        if "excluded_reason" in placement:
            reason = placement["excluded_reason"]
            rep["excluded"][reason] = rep["excluded"].get(reason, 0) + 1
        else:
            rep["resnapped" if placement["resnapped"] else "kept_edge"] += 1
            if placement.get("short_runway"):
                rep["short_runway_warn"] += 1
        if not args.dry_run:
            save_meta(scene_dir, meta)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"report: {args.report}")
    if args.dry_run:
        print("[dry-run] meta.json files not modified")
    return report


# ---------------------------------------------------------------------------
# Interactive
# ---------------------------------------------------------------------------

class InteractiveEditor:
    def __init__(self, scenes, params):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.scenes = scenes            # [(code, scene_dir, meta)]
        self.params = params
        self.idx = 0
        self.dirty = False
        self.status = ""
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # -- current scene state -----------------------------------------------
    @property
    def current(self):
        return self.scenes[self.idx]

    def load_net(self):
        code, scene_dir, meta = self.current
        return sumolib.net.readNet(net_path_for(scene_dir, meta))

    # -- drawing ------------------------------------------------------------
    def redraw(self):
        from matplotlib.patches import Polygon as MplPolygon, Circle
        code, scene_dir, meta = self.current
        net = self.net = self.load_net()
        ax = self.ax
        ax.clear()

        gps_x, gps_y = net.convertLonLat2XY(
            float(meta["longitude"]), float(meta["latitude"]))
        sign_edge_id = str(meta.get("road_id", ""))
        sign_edge = net.getEdge(sign_edge_id) if net.hasEdge(sign_edge_id) else None
        obstacle_idx = meta.get("sign_lane_index")
        targets = (allowed_target_lane_indices(sign_edge, code, int(obstacle_idx))
                   if sign_edge is not None and obstacle_idx is not None else set())
        feasible_now = (
            sign_edge is not None and obstacle_idx is not None
            and int(obstacle_idx) in valid_obstacle_lane_indices(sign_edge, code))

        for edge in net.getEdges():
            if edge.getFunction():
                continue
            is_sign_edge = sign_edge is not None and edge.getID() == sign_edge_id
            for lane in edge.getLanes():
                drivable = lane.allows("passenger")
                color, alpha, z = "0.75", 0.5, 1
                if not drivable:
                    color, alpha = "0.9", 0.35
                if is_sign_edge and drivable:
                    color, alpha, z = "#a8c8f0", 0.6, 2
                if is_sign_edge and lane.getIndex() in targets:
                    color, alpha, z = "#2b6cd4", 0.55, 3   # allowed detour lane
                if is_sign_edge and obstacle_idx is not None \
                        and lane.getIndex() == int(obstacle_idx):
                    color = "#2ca02c" if feasible_now else "#d62728"
                    alpha, z = 0.75, 4
                poly = lane_polygon(lane.getShape(), lane.getWidth())
                if len(poly) >= 3:
                    ax.add_patch(MplPolygon(
                        poly, closed=True, facecolor=color, alpha=alpha,
                        edgecolor="0.4", linewidth=0.3, zorder=z))
            # direction arrow along the edge axis
            shape = edge.getShape()
            mid = polyline_length(shape) / 2
            x0, y0, h = point_at(shape, max(0.0, mid - 2))
            ax.annotate(
                "", xy=(x0 + 4 * math.cos(h), y0 + 4 * math.sin(h)),
                xytext=(x0, y0), zorder=6,
                arrowprops=dict(arrowstyle="->", color="0.25", lw=0.8))

        ax.plot(gps_x, gps_y, "+", color="magenta", ms=14, mew=2.5, zorder=8,
                label="sign GPS")
        ax.add_patch(Circle((gps_x, gps_y), self.params.radius, fill=False,
                            linestyle="--", edgecolor="orange", zorder=7))
        if sign_edge is not None:
            sign_s = float(meta.get("sign_s",
                                    meta.get("distance_from_start", 0.0)) or 0.0)
            lane_for_marker = None
            if obstacle_idx is not None:
                lanes = sign_edge.getLanes()
                if 0 <= int(obstacle_idx) < len(lanes):
                    lane_for_marker = lanes[int(obstacle_idx)]
            shape = (lane_for_marker.getShape() if lane_for_marker is not None
                     else sign_edge.getShape())
            frac = sign_s / max(sign_edge.getLength(), 1e-6)
            sx, sy, _ = point_at(shape, frac * polyline_length(shape))
            ax.plot(sx, sy, "*", color="black", ms=16, zorder=9, label=f"sign {code}")

        ax.set_xlim(gps_x - self.params.radius * 1.6, gps_x + self.params.radius * 1.6)
        ax.set_ylim(gps_y - self.params.radius * 1.6, gps_y + self.params.radius * 1.6)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

        feas_txt = ("feasible" if feasible_now else "INFEASIBLE")
        excl = " [EXCLUDED: %s]" % meta.get("excluded_reason") \
            if meta.get("excluded") else ""
        dirty = " *unsaved*" if self.dirty else ""
        ax.set_title(
            f"[{self.idx + 1}/{len(self.scenes)}] {code} / "
            f"{os.path.basename(scene_dir)}  edge={sign_edge_id} "
            f"lane={obstacle_idx} s={meta.get('sign_s')} "
            f"({feas_txt}, {meta.get('detour_placed_by', '—')}){excl}{dirty}\n"
            f"{self.status}  |  click: obstacle lane; a=auto x=excl "
            f"w=save r=reload n/p=scene q=quit",
            fontsize=9)
        self.fig.canvas.draw_idle()

    # -- events -------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode:
            return  # zoom/pan active
        code, scene_dir, meta = self.current
        x, y = float(event.xdata), float(event.ydata)
        best = None  # (dist, edge, lane)
        for edge, _ in self.net.getNeighboringEdges(x, y, 20.0):
            if edge.getFunction():
                continue
            for lane in edge.getLanes():
                if not lane.allows("passenger"):
                    continue
                d = geomhelper.distancePointToPolygon((x, y), lane.getShape())
                d = max(0.0, d - lane.getWidth() / 2)
                if best is None or d < best[0]:
                    best = (d, edge, lane)
        if best is None or best[0] > CLICK_TOLERANCE_M:
            self.status = "no drivable lane near the click"
            self.redraw()
            return
        _, edge, lane = best
        s_proj, _ = project_on_edge(edge, x, y)
        sign_s = round(clamp_sign_s(s_proj, edge.getLength(),
                                    min(self.params.target_sign_s, s_proj)), 2)
        apply_placement(meta, {
            "road_id": edge.getID(),
            "sign_lane_index": lane.getIndex(),
            "sign_s": sign_s,
            "resnapped": edge.getID() != str(meta.get("road_id_orig",
                                                      meta.get("road_id"))),
            "placed_by": "manual",
        })
        ok = lane.getIndex() in valid_obstacle_lane_indices(edge, code)
        self.status = ("placed manually"
                       if ok else "WARNING: lane unsuitable for " + code)
        self.dirty = True
        self.redraw()

    def on_key(self, event):
        code, scene_dir, meta = self.current
        if event.key == "n":
            self.idx = (self.idx + 1) % len(self.scenes)
            self.dirty, self.status = False, ""
            self.redraw()
        elif event.key == "p":
            self.idx = (self.idx - 1) % len(self.scenes)
            self.dirty, self.status = False, ""
            self.redraw()
        elif event.key == "a":
            placement = suggest_placement(self.net, meta, code, self.params)
            apply_placement(meta, placement)
            self.status = ("auto: excluded %s" % placement["excluded_reason"]
                           if "excluded_reason" in placement else "auto-suggested")
            self.dirty = True
            self.redraw()
        elif event.key == "x":
            if meta.get("excluded"):
                meta.pop("excluded", None)
                meta.pop("excluded_reason", None)
            else:
                meta["excluded"] = True
                meta["excluded_reason"] = "manual_exclude"
            self.dirty = True
            self.redraw()
        elif event.key == "w":
            save_meta(scene_dir, meta)
            self.dirty = False
            self.status = "saved"
            self.redraw()
        elif event.key == "r":
            with open(os.path.join(scene_dir, "meta.json"), encoding="utf-8") as f:
                self.scenes[self.idx] = (code, scene_dir, json.load(f))
            self.dirty = False
            self.status = "reloaded from disk"
            self.redraw()
        elif event.key == "q":
            self.plt.close(self.fig)

    def run(self):
        self.redraw()
        self.plt.show()


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenes-root", required=True)
    p.add_argument("--codes", default="4.2.1,4.2.2,4.2.3",
                   type=lambda s: [c.strip() for c in s.split(",") if c.strip()])
    p.add_argument("--scene", default=None,
                   help="single scene: sign_100019 or 100019")
    p.add_argument("--batch", action="store_true", help="auto placement without GUI")
    p.add_argument("--report", default=None, help="batch-mode JSON report path")
    p.add_argument("--radius", type=float, default=75.0)
    p.add_argument("--min-edge-len", type=float, default=45.0)
    p.add_argument("--target-sign-s", type=float, default=60.0)
    p.add_argument("--only-unplaced", action="store_true",
                   help="skip scenes already processed by this tool")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for code in args.codes:
        if code not in DETOUR_CODES:
            p.error(f"code {code} is not a detour code (expected {DETOUR_CODES})")

    params = Params(radius=args.radius, min_edge_len=args.min_edge_len,
                    target_sign_s=args.target_sign_s)

    if args.batch:
        import matplotlib
        matplotlib.use("Agg")
        run_batch(args, params)
        return

    scenes = list(iter_scenes(args.scenes_root, args.codes,
                              args.scene, args.only_unplaced))
    if not scenes:
        print("no scenes match the given filters", file=sys.stderr)
        sys.exit(1)
    InteractiveEditor(scenes, params).run()


if __name__ == "__main__":
    main()
