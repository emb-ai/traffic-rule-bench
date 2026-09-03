#!/usr/bin/env python3
"""Recompute nuplan_statistics from the nuPlan v1.1 mini split (sqlite .db + .gpkg maps).

Every statistic is one function (see the STATISTICS section -- the formula is in
its docstring); process_db() only stitches their results together for one log.

Output (what NuPlanSampler and the benchmark consume):
    speeds.csv, acc_pos.csv, acc_neg.csv, following.csv, routes.csv,
    densities.csv, lane_changes.csv, ego_routes.csv,
    metadrive_config.json, statistics_report.json

Run:
  python3 extract_nuplan_statistics.py \
      --data-root $NUPLAN_ROOT/nuplan-v1.1/splits/mini --maps-root $NUPLAN_ROOT/maps \
      --out $OUT --workers 32
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Method thresholds (every number of the method is gathered here)
# ---------------------------------------------------------------------------
MOVING_SPEED_MIN = 0.5   # m/s: "in motion" (also the speed-frame cutoff)
MOVING_DISP_MIN = 2.0    # m: least displacement of a moving track over its life
MIN_TRACK_FRAMES = 20    # >=1 s at 20 Hz of track life for dynamics
SPEED_MAX_VALID = 40.0   # m/s: above this the annotation glitched, drop the frame
ACC_ABS_MAX = 8.0        # m/s2: |a| above this is dropped, NOT clipped
ACC_DEADBAND = 0.05      # m/s2: |a| below this is noise around zero, not written
SMOOTH_WIN = 5           # frames (0.25 s): speed smoothing before differentiating
FOLLOW_STRIDE = 2        # every 2nd frame (10 Hz) for following pairs
FOLLOW_MAX_GAP = 120.0   # m: longest longitudinal search for a leader
FOLLOW_LAT_MAX = 1.9     # m: |lateral| of the leader in the follower frame (~half a lane)
FOLLOW_HEADING_MAX = np.pi / 4   # 45 deg: heading mismatch within a pair
EGO_LENGTH = 5.0         # m: ego length (Chrysler Pacifica) for bumper-to-bumper
# m: the counting radius around the ego. 150 m is the benchmark's own scale --
# a speed scene caps the route at 150 m -- so the statistic describes the
# traffic a policy actually meets over one episode, not a city block.
DENSITY_RADIUS = 150.0
LANE_STRIDE = 5          # every 5th frame (4 Hz) for lane membership
LANE_BRIDGE_S = 1.0      # s: longest gap a transition is bridged across
LANE_MERGE_S = 3.0       # s: events of one track closer than this are one event
STATIC_RADIUS = 20.0     # m: a static obstacle "near the ego" (accident_prob)
STATIC_CATS = {"traffic_cone", "barrier", "czone_sign", "generic_object",
               "genericobject"}


def classify_size(length: float, width: float) -> str:
    """Thresholds EXACTLY as in nuplan_sampler.classify_size."""
    if length < 4.0:
        return "s"
    if length < 4.8:
        return "s" if width < 1.8 else "m"
    if length < 5.2:
        return "m"
    return "l" if length < 6.0 else "xl"


# ===========================================================================
# MAPS: lane polygons out of map.gpkg (no shapely/geopandas -- they are broken
# in the server environment). A GPKG is sqlite; its geometry is WGS84 while the
# coordinates in the .db are UTM (ego_pose.epsg), so project the polygons here.
# ===========================================================================
def wgs84_to_utm(lon_deg, lat_deg, epsg: int):
    """WGS84 -> UTM, Snyder's formulas (USGS), accurate to well under 1 m."""
    zone = epsg % 100
    north = (epsg // 100) % 10 == 6            # 326xx is the northern hemisphere
    a, f = 6378137.0, 1.0 / 298.257223563
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    k0 = 0.9996
    lon0 = np.radians(zone * 6 - 183)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = ep2 * np.cos(lat) ** 2
    A = np.cos(lat) * (lon - lon0)
    M = a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * lat)
             + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * lat)
             - (35 * e2**3 / 3072) * np.sin(6 * lat))
    x = k0 * N * (A + (1 - T + C) * A**3 / 6
                  + (5 - 18 * T + T**2 + 72 * C - 58 * ep2) * A**5 / 120) + 500000.0
    y = k0 * (M + N * np.tan(lat) * (A**2 / 2 + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
              + (61 - 58 * T + T**2 + 600 * C - 330 * ep2) * A**6 / 720))
    return x, (y if north else y + 10000000.0)


def gpkg_polygon_rings(blob: bytes) -> list[np.ndarray]:
    """GPKG blob -> a list of exterior rings (Nx2, lon/lat). Handles Polygon,
    MultiPolygon, Z/M dimensions and both byte orders."""
    off = 0
    if blob[:2] == b"GP":                      # GeoPackage header
        env = (blob[3] >> 1) & 0x07
        off = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    rings: list[np.ndarray] = []

    def read_geom(off: int) -> int:
        bo = "<" if blob[off] == 1 else ">"
        gtype = struct.unpack_from(bo + "I", blob, off + 1)[0]
        off += 5
        has_z = bool(gtype & 0x80000000) or (gtype % 10000) // 1000 in (1, 3)
        has_m = bool(gtype & 0x40000000) or (gtype % 10000) // 1000 in (2, 3)
        ndim = 2 + has_z + has_m
        base = gtype & 0xFF
        if base == 6:                          # MultiPolygon -> recurse
            (n,) = struct.unpack_from(bo + "I", blob, off)
            off += 4
            for _ in range(n):
                off = read_geom(off)
            return off
        if base != 3:
            raise ValueError(f"unsupported WKB type {gtype}")
        (nrings,) = struct.unpack_from(bo + "I", blob, off)
        off += 4
        for ri in range(nrings):
            (npts,) = struct.unpack_from(bo + "I", blob, off)
            off += 4
            pts = np.frombuffer(blob, np.dtype(bo + "f8"), npts * ndim, off)
            off += npts * ndim * 8
            if ri == 0:                        # the exterior ring only
                rings.append(pts.reshape(npts, ndim)[:, :2].copy())
        return off

    read_geom(off)
    return rings


class CityLanes:
    """Polygons of the lanes_polygons layer plus a grid index.

    assign(x, y) -> (lane_fid, lane_group_fid) per point; -1 where the point is in
    no lane (junctions and car parks are not in this layer, which is expected).
    """

    def __init__(self, gpkg_path: Path, epsg: int, grid_cell: float = 50.0):
        from matplotlib.path import Path as MplPath
        con = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT lane_fid, lane_group_fid, geom FROM lanes_polygons").fetchall()
        con.close()
        self.paths, self.lane_ids, self.group_ids, boxes = [], [], [], []
        for lane_fid, group_fid, blob in rows:
            if blob is None:
                continue
            for ring in gpkg_polygon_rings(blob):
                ux, uy = wgs84_to_utm(ring[:, 0], ring[:, 1], epsg)
                self.paths.append(MplPath(np.column_stack([ux, uy]), closed=True))
                self.lane_ids.append(int(lane_fid))
                self.group_ids.append(int(group_fid) if group_fid is not None else -1)
                boxes.append((ux.min(), uy.min(), ux.max(), uy.max()))
        # lanes per road: distinct lane_fid within one lane_group. A lane may
        # contribute several rings, so count the set, not the rows.
        _by_group: dict[int, set] = {}
        for lane_fid, group_fid in zip(self.lane_ids, self.group_ids):
            _by_group.setdefault(group_fid, set()).add(lane_fid)
        self.lanes_in_group = {g: len(s) for g, s in _by_group.items()}

        # grid index: a 50 m cell -> the polygons whose bbox touches it
        self.cell = grid_cell
        self.grid: dict[tuple[int, int], list[int]] = {}
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for cx in range(int(x0 // grid_cell), int(x1 // grid_cell) + 1):
                for cy in range(int(y0 // grid_cell), int(y1 // grid_cell) + 1):
                    self.grid.setdefault((cx, cy), []).append(i)

    def assign(self, xs, ys):
        lane = np.full(len(xs), -1, dtype=np.int64)
        group = np.full(len(xs), -1, dtype=np.int64)
        for j, (x, y) in enumerate(zip(xs, ys)):
            for i in self.grid.get((int(x // self.cell), int(y // self.cell)), ()):
                if self.paths[i].contains_point((x, y)):
                    lane[j], group[j] = self.lane_ids[i], self.group_ids[i]
                    break
        return lane, group


_MAPS_CACHE: dict = {}


def get_city_lanes(maps_root, map_name, epsg) -> CityLanes | None:
    if not maps_root or not epsg:
        return None
    key = (map_name, epsg)
    if key not in _MAPS_CACHE:
        hits = sorted((Path(maps_root) / map_name).rglob("map.gpkg"))
        if not hits:
            sys.stderr.write(f"[warn] no map.gpkg for {map_name}\n")
            _MAPS_CACHE[key] = None
        else:
            _MAPS_CACHE[key] = CityLanes(hits[-1], epsg)
    return _MAPS_CACHE[key]


# ===========================================================================
# READING A LOG: one .db -> flat arrays
# ===========================================================================
def read_log(db_path: str) -> dict:
    """Reads out of a .db: frames (20 Hz, time and ego pose), car boxes by track,
    positions of static obstacles, the map name and the EPSG."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    map_name = cur.execute("SELECT map_version FROM log LIMIT 1").fetchone()[0]
    epsg = int(cur.execute("SELECT epsg FROM ego_pose LIMIT 1").fetchone()[0])
    cats = dict(cur.execute("SELECT token, name FROM category").fetchall())

    frames = cur.execute(
        "SELECT lp.token, lp.timestamp, ep.x, ep.y, lp.scene_token "
        "FROM lidar_pc lp JOIN ego_pose ep ON lp.ego_pose_token = ep.token "
        "ORDER BY lp.timestamp").fetchall()
    tok2idx = {f[0]: i for i, f in enumerate(frames)}

    boxes = cur.execute(
        "SELECT lb.lidar_pc_token, lb.track_token, lb.x, lb.y, lb.yaw, "
        "lb.vx, lb.vy, lb.width, lb.length, t.category_token "
        "FROM lidar_box lb JOIN track t ON lb.track_token = t.token").fetchall()
    con.close()

    veh_rows, static_rows = [], []
    for pc_tok, tr_tok, x, y, yaw, vx, vy, w, l, cat_tok in boxes:
        fi = tok2idx.get(pc_tok)
        if fi is None:
            continue
        name = cats.get(cat_tok, "")
        if name == "vehicle":
            veh_rows.append((fi, tr_tok.hex(), x, y, yaw,
                             np.nan if vx is None else vx,
                             np.nan if vy is None else vy,
                             np.nan if w is None else w,
                             np.nan if l is None else l))
        elif name in STATIC_CATS:
            static_rows.append((fi, x, y))

    df = pd.DataFrame(veh_rows, columns=["fi", "tk", "x", "y", "yaw",
                                         "vx", "vy", "w", "l"])
    df["speed"] = np.hypot(df["vx"], df["vy"])
    return {
        "log": Path(db_path).stem,
        "map_name": map_name, "epsg": epsg,
        "t": np.array([f[1] for f in frames], dtype=np.int64) / 1e6,   # seconds
        "ts_us": np.array([f[1] for f in frames], dtype=np.int64),
        "ego_xy": np.array([[f[2], f[3]] for f in frames]),
        "scene": [f[4].hex() for f in frames],
        "veh": df.sort_values(["tk", "fi"], kind="stable").reset_index(drop=True),
        "static": np.array(static_rows, dtype=float).reshape(-1, 3),
    }


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) < win:
        return x
    return np.convolve(np.pad(x, win // 2, mode="edge"),
                       np.ones(win) / win, mode="valid")[: len(x)]


def split_tracks(log: dict) -> list[dict]:
    """Cuts boxes into tracks. A track "moves" when p95(speed) >= 0.5 m/s AND its
    total displacement >= 2 m (which rejects parked cars and annotation jitter)."""
    tracks = []
    for tk, g in log["veh"].groupby("tk", sort=False):
        fi = g["fi"].to_numpy()
        xs, ys = g["x"].to_numpy(), g["y"].to_numpy()
        sp = g["speed"].to_numpy()
        sp = np.where(np.isfinite(sp) & (sp <= SPEED_MAX_VALID), sp, np.nan)
        disp = float(np.nansum(np.hypot(np.diff(xs), np.diff(ys))))
        valid = sp[np.isfinite(sp)]
        moving = (len(valid) > 0
                  and np.percentile(valid, 95) >= MOVING_SPEED_MIN
                  and disp >= MOVING_DISP_MIN)
        tracks.append({
            "tk": tk, "fi": fi, "x": xs, "y": ys, "t": log["t"][fi],
            "speed": sp, "disp": disp, "moving": moving,
            "med_l": float(np.nanmedian(g["l"])),
            "med_w": float(np.nanmedian(g["w"])),
        })
    return tracks


# ===========================================================================
# STATISTICS -- one function per output file
# ===========================================================================
def stat_speeds(tr: dict) -> list[float]:
    """speeds.csv: per-frame speeds of a moving track, >= 0.5 m/s."""
    sp = tr["speed"]
    return sp[np.isfinite(sp) & (sp >= MOVING_SPEED_MIN)].tolist()


def stat_accels(tr: dict) -> tuple[list[float], list[float], int]:
    """acc_pos/acc_neg.csv: a = d(speed smoothed over 0.25 s)/dt on frames in
    motion; |a| > 8 is dropped, |a| < 0.05 is noise around zero.
    Returns (accelerations, decelerations>0, how many were dropped by |a|>8)."""
    ok = np.isfinite(tr["speed"])
    if ok.sum() < MIN_TRACK_FRAMES:
        return [], [], 0
    t, sp = tr["t"][ok], tr["speed"][ok]
    a = np.gradient(moving_average(sp, SMOOTH_WIN), t)[sp >= MOVING_SPEED_MIN]
    dropped = int((np.abs(a) > ACC_ABS_MAX).sum())
    a = a[np.abs(a) <= ACC_ABS_MAX]
    return (a[a >= ACC_DEADBAND].tolist(),
            (-a[a <= -ACC_DEADBAND]).tolist(), dropped)


def stat_route(tr: dict) -> dict | None:
    """routes.csv: one row per moving track that lived >= 1 s.
    NOTE: distance/duration are the track's path INSIDE the annotation window
    (~50-80 m around the ego), NOT the real trip length; initial_speed is the
    speed on entering the window -- that is what spawn_velocity samples."""
    duration = float(tr["t"][-1] - tr["t"][0])
    if duration < 1.0:
        return None
    finite = tr["speed"][np.isfinite(tr["speed"])]
    return {
        "track_token": tr["tk"], "category": "vehicle",
        "distance": tr["disp"], "duration": duration,
        "avg_speed": tr["disp"] / duration,
        "initial_speed": float(finite[0]) if len(finite) else np.nan,
        "size_class": classify_size(tr["med_l"], tr["med_w"])
        if np.isfinite(tr["med_l"]) else "m",
        "length": tr["med_l"], "width": tr["med_w"],
    }


def stat_lane_changes(tr: dict, lanes: CityLanes, cnt: Counter) -> list[dict]:
    """lane_changes.csv: 1 row = 1 event. Track positions (4 Hz) are assigned to
    map lanes; an event is a change of lane_fid WITHIN one lane_group (adjacent
    lanes of the same stretch of road). Transitions between groups (driving
    forward along the road, junctions) are not events; events closer than 3 s
    merge into one."""
    sub = slice(0, len(tr["fi"]), LANE_STRIDE)
    xs, ys, ts = tr["x"][sub], tr["y"][sub], tr["t"][sub]
    lane, group = lanes.assign(xs, ys)
    cnt["lane_samples_assigned"] += int((lane >= 0).sum())
    cnt["lane_samples_total"] += len(lane)

    events = []
    prev = None                                # index of the last sample with a lane
    for j in np.flatnonzero(lane >= 0):
        if prev is not None and lane[j] != lane[prev] \
                and ts[j] - ts[prev] <= LANE_BRIDGE_S:
            if group[j] == group[prev] and group[j] >= 0:
                dx, dy = xs[j] - xs[prev], ys[j] - ys[prev]
                heading = np.arctan2(dy, dx)
                lat = abs(-np.sin(heading) * dx + np.cos(heading) * dy)
                events.append((0.5 * (ts[j] + ts[prev]), float(lat)))
                cnt["lane_events_raw"] += 1
            else:
                cnt["lane_group_transitions"] += 1
        prev = j

    merged, rows = -np.inf, []
    for ev_t, ev_lat in events:
        if ev_t - merged < LANE_MERGE_S:
            continue
        merged = ev_t
        rows.append({"track_token": tr["tk"], "timestamp": int(ev_t * 1e6),
                     "lateral_shift": round(ev_lat, 4)})
    return rows


def stat_ego_routes(log: dict) -> list[dict]:
    """ego_routes.csv: the EGO route length, per nuPlan scene (~20 s) and over the
    whole log. The only route length not cut off by the annotation window.
    Steps across time gaps > 1 s do not count towards the path."""
    t, xy = log["t"], log["ego_xy"]
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    step = np.where(np.diff(t) <= 1.0, step, 0.0)
    v = np.concatenate([[0.0], step / np.maximum(np.diff(t), 1e-3)])

    rows = []
    for sc, idx in pd.Series(range(len(t))).groupby(np.asarray(log["scene"])):
        ii = idx.to_numpy()
        dur = float(t[ii[-1]] - t[ii[0]])
        if len(ii) < 2 or dur <= 0:
            continue
        dist = float(step[ii[0]:ii[-1]].sum())
        rows.append({"log": log["log"], "scene_token": sc, "kind": "scene",
                     "distance": dist, "duration": dur,
                     "avg_speed": dist / dur, "initial_speed": float(v[ii[0]])})
    rows.append({"log": log["log"], "scene_token": "", "kind": "log",
                 "distance": float(step.sum()),
                 "duration": float(t[-1] - t[0]),
                 "avg_speed": float(step.sum() / max(t[-1] - t[0], 1e-3)),
                 "initial_speed": float(v[0])})
    return rows


def ego_heading(log: dict) -> np.ndarray:
    """Ego heading from trajectory steps; while parked, the last known one."""
    xy = log["ego_xy"]
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    yaw = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    yaw = np.concatenate([yaw[:1] if len(yaw) else [0.0], yaw])
    mask = np.concatenate([[False], step >= 0.05])       # a step under 5 cm is standing
    return pd.Series(np.where(mask, yaw, np.nan)).ffill().bfill().fillna(0.0).to_numpy()


def stats_per_frame(log: dict, tracks: list[dict], cnt: Counter, lanes=None):
    """densities.csv and following.csv in one pass over the frames.

    density (per frame): everything is counted inside DENSITY_RADIUS of the ego and
      nowhere else -- a count over the whole annotation window describes the city,
      not the road the ego drives. ego_lanes is the number of lanes on the ego's
      own road (distinct lane_fid in its lane_group), 0 where the ego is in a
      junction or a car park, which the lane layer does not cover.
      count_moving_per_lane is the quantity the simulator has to reproduce:
      MetaDrive applies traffic_density to each lane separately, so a per-frame
      total is not comparable to it, and a per-lane one is.

    following (every 2nd frame): for each MOVING follower the others are put into
      its own frame of reference (by the follower yaw); the leader is the nearest
      with 0.5 < dx <= 120 m, |dy| <= 1.9 m, |dyaw| <= 45 deg;
      distance = dx - (follower_length + leader_length)/2, bumper to bumper.
      The ego is a leader candidate (otherwise the distance of a car behind it is
      overstated) but is never written as a follower (the population is tracks, as
    """
    moving = {tr["tk"]: tr["moving"] for tr in tracks}
    df = log["veh"]
    order = np.argsort(df["fi"].to_numpy(), kind="stable")
    fi = df["fi"].to_numpy()[order]
    X, Y = df["x"].to_numpy()[order], df["y"].to_numpy()[order]
    YAW, SP = df["yaw"].to_numpy()[order], df["speed"].to_numpy()[order]
    LEN = df["l"].to_numpy()[order]
    MOV = df["tk"].map(moving).to_numpy()[order].astype(bool)
    eyaw = ego_heading(log)
    ego_v = np.concatenate([[0.0], np.hypot(np.diff(log["ego_xy"][:, 0]),
                                            np.diff(log["ego_xy"][:, 1]))
                            / np.maximum(np.diff(log["t"]), 1e-3)])

    n_frames = len(log["t"])
    bounds = np.searchsorted(fi, np.arange(n_frames + 1))

    # Lanes on the ego's road, per frame. One assign() for the whole log.
    ego_lanes = np.zeros(n_frames, dtype=np.int64)
    if lanes is not None:
        _, groups = lanes.assign(log["ego_xy"][:, 0], log["ego_xy"][:, 1])
        for f, g in enumerate(groups):
            ego_lanes[f] = lanes.lanes_in_group.get(int(g), 0)
        cnt["density_frames_with_lanes"] += int((ego_lanes > 0).sum())
        cnt["density_frames_no_lane"] += int((ego_lanes == 0).sum())

    density_rows, gaps = [], []
    for f in range(n_frames):
        lo, hi = bounds[f], bounds[f + 1]
        ex, ey = log["ego_xy"][f]
        r = np.hypot(X[lo:hi] - ex, Y[lo:hi] - ey)
        mv, inr = MOV[lo:hi], r <= DENSITY_RADIUS
        n_in = int(inr.sum())
        n_mv = int((mv & inr).sum())
        nl = int(ego_lanes[f])
        density_rows.append((int(log["ts_us"][f]), nl, n_in, n_mv,
                             (n_mv / nl) if nl > 0 else float("nan")))

        if f % FOLLOW_STRIDE or hi == lo:
            continue
        # candidates = cars of the frame plus the ego (ego last, never a follower)
        px = np.append(X[lo:hi], ex)
        py = np.append(Y[lo:hi], ey)
        pyaw = np.append(YAW[lo:hi], eyaw[f])
        psp = np.append(SP[lo:hi], ego_v[f])
        plen = np.where(np.isfinite(np.append(LEN[lo:hi], EGO_LENGTH)),
                        np.append(LEN[lo:hi], EGO_LENGTH), 4.5)
        for i in range(hi - lo):
            if not mv[i] or not np.isfinite(psp[i]) or psp[i] < MOVING_SPEED_MIN:
                continue
            c, s = np.cos(pyaw[i]), np.sin(pyaw[i])
            dx = c * (px - px[i]) + s * (py - py[i])          # ahead of the follower
            dy = -s * (px - px[i]) + c * (py - py[i])         # left of the follower
            dyaw = np.abs((pyaw - pyaw[i] + np.pi) % (2 * np.pi) - np.pi)
            cand = ((dx > 0.5) & (dx <= FOLLOW_MAX_GAP)
                    & (np.abs(dy) <= FOLLOW_LAT_MAX) & (dyaw <= FOLLOW_HEADING_MAX))
            cand[i] = False
            if not cand.any():
                continue
            j = np.flatnonzero(cand)[np.argmin(dx[cand])]
            gap = float(dx[j] - 0.5 * (plen[i] + plen[j]))
            if gap > 0.1:
                gaps.append(gap)

    # accident_prob: frames with a static obstacle closer than 20 m to the ego
    st = log["static"]
    if len(st):
        d = np.hypot(st[:, 1] - log["ego_xy"][st[:, 0].astype(int), 0],
                     st[:, 2] - log["ego_xy"][st[:, 0].astype(int), 1])
        cnt["frames_with_static"] += len(set(st[d <= STATIC_RADIUS, 0].astype(int).tolist()))
    cnt["frames_total"] += n_frames
    return density_rows, gaps


# ===========================================================================
# Orchestrator: one .db -> every statistic
# ===========================================================================
def process_db(args_tuple):
    db_path, maps_root = args_tuple
    out = {"speeds": [], "acc_pos": [], "acc_neg": [], "following": [],
           "routes": [], "density_rows": [], "lane_events": [], "ego_routes": [],
           "counters": Counter()}
    try:
        log = read_log(db_path)
        cnt = out["counters"]
        lanes = get_city_lanes(maps_root, log["map_name"], log["epsg"])
        tracks = split_tracks(log)

        out["ego_routes"] = stat_ego_routes(log)
        for tr in tracks:
            if not tr["moving"] or len(tr["fi"]) < MIN_TRACK_FRAMES:
                continue
            cnt["moving_tracks"] += 1
            out["speeds"].extend(stat_speeds(tr))
            pos, neg, dropped = stat_accels(tr)
            out["acc_pos"].extend(pos)
            out["acc_neg"].extend(neg)
            cnt["acc_dropped_gt8"] += dropped
            route = stat_route(tr)
            if route:
                out["routes"].append(route)
            if lanes is not None:
                out["lane_events"].extend(stat_lane_changes(tr, lanes, cnt))

        out["density_rows"], out["following"] = stats_per_frame(log, tracks, cnt, lanes)
        cnt["dbs_ok"] += 1
    except Exception as e:
        out["counters"]["dbs_failed"] += 1
        sys.stderr.write(f"[error] {db_path}: {type(e).__name__}: {e}\n")
    return out


# ===========================================================================
# Aggregation and writing
# ===========================================================================
def downsample(arr, max_rows, rng):
    if max_rows and len(arr) > max_rows:
        idx = np.sort(rng.choice(len(arr), size=max_rows, replace=False))
        return [arr[i] for i in idx]
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--maps-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-dbs", type=int, default=0)
    ap.add_argument("--max-speed-rows", type=int, default=2_000_000)
    ap.add_argument("--max-acc-rows", type=int, default=1_000_000)
    ap.add_argument("--max-follow-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    dbs = sorted(Path(args.data_root).glob("*.db")) \
        or sorted(Path(args.data_root).rglob("*.db"))
    if args.limit_dbs:
        dbs = dbs[: args.limit_dbs]
    if not dbs:
        sys.exit(f"no .db files under {args.data_root}")
    print(f"processing {len(dbs)} db files, {args.workers} workers "
          f"(maps: {args.maps_root or 'OFF'})")

    t0 = time.time()
    agg = {"speeds": [], "acc_pos": [], "acc_neg": [], "following": [],
           "routes": [], "density_rows": [], "lane_events": [], "ego_routes": [],
           "counters": Counter()}
    with Pool(min(args.workers, len(dbs))) as pool:
        tasks = [(str(p), args.maps_root) for p in dbs]
        for i, res in enumerate(pool.imap_unordered(process_db, tasks), 1):
            for k, v in res.items():
                agg[k].update(v) if k == "counters" else agg[k].extend(v)
            if i % 10 == 0 or i == len(dbs):
                print(f"  [{i}/{len(dbs)}] speeds={len(agg['speeds'])} "
                      f"lane_events={len(agg['lane_events'])} ({time.time()-t0:.0f}s)")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    speeds = downsample(agg["speeds"], args.max_speed_rows, rng)
    acc_pos = downsample(agg["acc_pos"], args.max_acc_rows, rng)
    acc_neg = downsample(agg["acc_neg"], args.max_acc_rows, rng)
    following = downsample(agg["following"], args.max_follow_rows, rng)

    pd.DataFrame({"speed": speeds}).to_csv(outdir / "speeds.csv", index=False)
    pd.DataFrame({"acceleration": acc_pos}).to_csv(outdir / "acc_pos.csv", index=False)
    pd.DataFrame({"deceleration": acc_neg}).to_csv(outdir / "acc_neg.csv", index=False)
    pd.DataFrame({"following_distance": following}).to_csv(
        outdir / "following.csv", index=False)
    routes = pd.DataFrame(agg["routes"])
    routes.to_csv(outdir / "routes.csv", index=False)
    dens = pd.DataFrame(agg["density_rows"], columns=[
        "timestamp", "ego_lanes", "count_r150", "count_moving_r150",
        "count_moving_r150_per_lane"])
    dens.to_csv(outdir / "densities.csv", index=False)
    lc = pd.DataFrame(agg["lane_events"],
                      columns=["track_token", "timestamp", "lateral_shift"])
    lc.to_csv(outdir / "lane_changes.csv", index=False)
    ego = pd.DataFrame(agg["ego_routes"])
    ego.to_csv(outdir / "ego_routes.csv", index=False)

    write_reports(outdir, args, dbs, agg, speeds, acc_pos, acc_neg, following,
                  routes, dens, lc, ego)
    print(f"\ndone in {time.time()-t0:.0f}s -> {outdir}")


def write_reports(outdir, args, dbs, agg, speeds, acc_pos, acc_neg, following,
                  routes, dens, lc, ego):
    sp, ap_, an = map(np.asarray, (speeds, acc_pos, acc_neg))
    fo = np.asarray(following)
    ego_scene = ego[ego["kind"] == "scene"]
    ego_log = ego[ego["kind"] == "log"]
    total_km = routes["distance"].sum() / 1000.0
    lane_rate = len(lc) / total_km if total_km > 0 else float("nan")
    cnt = agg["counters"]
    accident_prob = cnt["frames_with_static"] / max(1, cnt["frames_total"])

    def pct(a, q):
        # NaN-tolerant: the per-lane density column is NaN wherever the ego sat in
        # a junction, and np.percentile would poison the whole row.
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, q)) if len(a) else float("nan")

    def block(a):
        return {"mean": float(np.nanmean(a)), "median": pct(a, 50),
                "std": float(np.nanstd(a))}

    config = {
        "INITIAL_SPEED": {"mean": float(routes["initial_speed"].mean())},
        "NORMAL_SPEED": {**block(sp), "percentiles": {
            str(q): pct(sp, q) for q in (5, 50, 85, 95)}},
        "MAX_SPEED": {"percentile_95": pct(sp, 95), "percentile_99": pct(sp, 99)},
        "CREEP_SPEED": {"percentile_5": pct(sp, 5)},
        "ACC_FACTOR": block(ap_),
        "DEACC_FACTOR": block(an),
        "DISTANCE_WANTED": block(fo),
        "LANE_CHANGE_FREQ": {"per_km": lane_rate},
        "horizon": {"min": float(routes["duration"].min()),
                    "mean": float(routes["duration"].mean()),
                    "median": float(routes["duration"].median())},
        # Per-lane moving traffic inside DENSITY_RADIUS: the quantity the
        # simulator is calibrated to reproduce. Kept as the full distribution,
        # not a tier, so a scene can sample its own value.
        "traffic_density_per_lane": {
            "mean": float(np.nanmean(dens["count_moving_r150_per_lane"])),
            "percentiles": {str(q): pct(dens["count_moving_r150_per_lane"], q)
                            for q in (5, 10, 25, 50, 75, 90, 95)},
        },
        "accident_prob": accident_prob,
        "size_prob": {k: float(v) for k, v in
                      routes["size_class"].value_counts(normalize=True).items()},
        "route_length": {"min": float(routes["distance"].min()),
                         "mean": float(routes["distance"].mean()),
                         "median": float(routes["distance"].median()),
                         "std": float(routes["distance"].std())},
        "ego_route_length": {
            "per_scene": {**block(ego_scene["distance"].to_numpy()),
                          "p5": pct(ego_scene["distance"], 5),
                          "p95": pct(ego_scene["distance"], 95),
                          "n": int(len(ego_scene))},
            "per_log": {"mean": float(ego_log["distance"].mean()),
                        "median": float(ego_log["distance"].median()),
                        "min": float(ego_log["distance"].min()),
                        "max": float(ego_log["distance"].max()),
                        "n": int(len(ego_log))},
        },
        "_provenance": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": str(args.data_root), "n_dbs": len(dbs),
            "maps_root": args.maps_root, "counters": dict(cnt),
            "definitions": {
                "speeds": "per-frame speeds of moving car tracks, >=0.5 m/s, <=40",
                "acc": "gradient of the 0.25 s smoothed speed; |a|<=8; deadband 0.05",
                "following": "bumper-to-bumper to the nearest leader (cars + ego): "
                             "dx<=120, |dy|<=1.9, dyaw<=45deg; followers are tracks only",
                "densities.count_r150": "cars within 150 m of the ego -- one episode's worth of road",
                "densities.count_moving_r150": "of those, the moving ones",
                "densities.ego_lanes": "lanes on the ego's own road (lane_group), 0 in junctions",
                "densities.count_moving_r150_per_lane": "count_moving_r150 / ego_lanes -- "
                                                        "the quantity traffic_density is calibrated against",
                "lane_changes": "1 row = 1 event: a lane_fid change within one lane_group",
                "accident_prob": "share of frames with a static obstacle closer than 20 m to the ego",
                "routes.distance": "track path INSIDE THE ANNOTATION WINDOW (~50-80 m), cut by observation",
                "ego_routes": "ego route: kind=scene (~20 s) and kind=log -- not cut",
                "size_class": "median track dimensions, thresholds as in nuplan_sampler",
            },
        },
    }
    (outdir / "metadrive_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False))

    report = {
        "speed_stats": {**block(sp), "percentiles": {
            f"{q}%": pct(sp, q) for q in (5, 25, 50, 75, 85, 95, 99)}},
        "acceleration_stats": block(ap_),
        "deceleration_stats": block(an),
        "following_distance_stats": block(fo),
        "route_stats": {"distance_mean": float(routes["distance"].mean()),
                        "distance_median": float(routes["distance"].median()),
                        "duration_mean": float(routes["duration"].mean()),
                        "duration_median": float(routes["duration"].median())},
        "traffic_density": {
            col: {"mean": float(dens[col].mean()),
                  "p25/50/75": [pct(dens[col], q) for q in (25, 50, 75)]}
            for col in ("ego_lanes", "count_r150", "count_moving_r150",
                        "count_moving_r150_per_lane")},
        "lane_changes": {"events": int(len(lc)), "total_km": total_km,
                         "rate_per_km": lane_rate},
        "ego_route_stats": {
            "per_scene_distance": {"mean": float(ego_scene["distance"].mean()),
                                   "median": float(ego_scene["distance"].median()),
                                   "p5": pct(ego_scene["distance"], 5),
                                   "p95": pct(ego_scene["distance"], 95)},
            "per_scene_duration": {"median": float(ego_scene["duration"].median())},
            "per_log_distance": {"mean": float(ego_log["distance"].mean()),
                                 "median": float(ego_log["distance"].median())},
            "per_scene_avg_speed": {"median": float(ego_scene["avg_speed"].median())},
        },
    }
    (outdir / "statistics_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
