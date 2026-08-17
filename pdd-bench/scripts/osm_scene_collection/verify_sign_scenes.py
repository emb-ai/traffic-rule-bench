#!/usr/bin/env python3
"""Verify that speed-limit and detour scenes carry a usable manoeuvre.

``verify_scene.py`` answers "does the net load"; this answers "can the sign's
rule actually be obeyed here", which is the question that decides whether a
scene is worth dumping and training on. Both are needed: a scene can parse
perfectly and still be impossible.

Detour (4.2.1 / 4.2.2 / 4.2.3) — the meta written by ``detour_scene_editor``
is re-derived from the net instead of trusted:

  * ``sign_lane_index`` must still be a drivable lane of ``road_id``
  * the obstacle lane must have somewhere to merge INTO, on the side the code
    demands: 4.2.1 passes on the right (needs a drivable lane at a LOWER SUMO
    index), 4.2.2 on the left (HIGHER index), 4.2.3 either. SUMO numbers lanes
    from the right, so the index arithmetic is the whole check — an off-by-one
    convention error turns every scene into an impossible manoeuvre while the
    net still parses.
  * ``sign_s`` must sit inside the edge, with room for the obstacle ahead

Speed limit (3.24 / 5.31) — the scene stores no limit at all; the enforced
value is assigned later by ``sumo_catalog``. What is checkable here is the
road speed that becomes the model's ``speed_limit`` input, and whether the
catalog would keep the scene at all (it drops roads above 80 km/h).

Usage:
  python verify_sign_scenes.py scenes/4.2.1 scenes/4.2.2 scenes/4.2.3
  python verify_sign_scenes.py scenes/3.24 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import sumolib

DETOUR_CODES = ("4.2.1", "4.2.2", "4.2.3")
SPEED_CODES = ("3.24", "5.31")
# sumo_catalog drops these: motorway geometry is unsuitable for a 20-40 limit.
CATALOG_MAX_ROAD_KMH = 80
# The obstacle sits ahead of the sign; without this much edge left the cones
# spawn past the end and the scene has a sign but nothing to drive around.
MIN_TAIL_AFTER_SIGN_M = 10.0


def drivable_lane_indices(edge) -> set[int]:
    out = set()
    for lane in edge.getLanes():
        try:
            if lane.allows("passenger"):
                out.add(lane.getIndex())
        except Exception:
            out.add(lane.getIndex())
    return out


def merge_side_exists(drivable: set[int], idx: int, code: str) -> bool:
    """Is there a drivable lane on the side this code sends the ego to?"""
    has_right = any(j < idx for j in drivable)
    has_left = any(j > idx for j in drivable)
    if code == "4.2.1":
        return has_right
    if code == "4.2.2":
        return has_left
    return has_right or has_left


def check_detour(meta: dict, edge, code: str) -> list[str]:
    problems = []
    drivable = drivable_lane_indices(edge)
    idx = meta.get("sign_lane_index")

    if idx is None:
        problems.append("no sign_lane_index (scene predates the detour editor)")
        return problems
    idx = int(idx)
    if idx not in drivable:
        problems.append(
            f"sign_lane_index={idx} is not a drivable lane of {edge.getID()} "
            f"(drivable={sorted(drivable)})")
        return problems
    if not merge_side_exists(drivable, idx, code):
        side = {"4.2.1": "right", "4.2.2": "left"}.get(code, "either side")
        problems.append(
            f"{code} must pass on the {side}, but lane {idx} of "
            f"{edge.getID()} has no drivable lane there (drivable={sorted(drivable)})")

    s = meta.get("sign_s")
    if s is not None:
        s = float(s)
        length = float(edge.getLength())
        if not (0.0 <= s <= length):
            problems.append(f"sign_s={s:.1f} outside edge length {length:.1f}")
        elif length - s < MIN_TAIL_AFTER_SIGN_M:
            problems.append(
                f"only {length - s:.1f} m of edge after the sign "
                f"(< {MIN_TAIL_AFTER_SIGN_M:.0f} m): no room for the obstacle")
    return problems


def scene_dirs(paths) -> list[Path]:
    out = []
    for raw in paths:
        p = Path(raw)
        if (p / "meta.json").is_file():
            out.append(p)
        else:
            out.extend(sorted(d for d in p.glob("sign_*") if (d / "meta.json").is_file()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="scene dir(s) or a code dir")
    ap.add_argument("--verbose", action="store_true", help="print every scene, not just failures")
    args = ap.parse_args()

    dirs = scene_dirs(args.paths)
    if not dirs:
        print("no scenes found", file=sys.stderr)
        return 2

    n_ok = n_excluded = 0
    failures: list[tuple[Path, list[str]]] = []
    road_kmh = Counter()
    over_limit = 0
    lane_hist: dict[str, Counter] = {}

    for d in dirs:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        code = str(meta.get("sign_type", "")).strip()

        if meta.get("excluded"):
            n_excluded += 1
            if not meta.get("excluded_reason"):
                failures.append((d, ["excluded without excluded_reason"]))
            continue

        net_file = d / meta.get("net_file", f"{d.name}.net.xml")
        if not net_file.is_file():
            failures.append((d, [f"missing net file {net_file.name}"]))
            continue
        try:
            net = sumolib.net.readNet(str(net_file))
        except Exception as e:  # noqa: BLE001
            failures.append((d, [f"net will not load: {e}"]))
            continue

        road_id = meta.get("road_id")
        if not net.hasEdge(road_id):
            failures.append((d, [f"road_id {road_id!r} absent from the net"]))
            continue
        edge = net.getEdge(road_id)

        problems: list[str] = []
        if code in DETOUR_CODES:
            problems = check_detour(meta, edge, code)
            lane_hist.setdefault(code, Counter())[meta.get("sign_lane_index")] += 1
        elif code in SPEED_CODES:
            kmh = round(float(edge.getSpeed()) * 3.6)
            road_kmh[kmh] += 1
            if kmh > CATALOG_MAX_ROAD_KMH:
                over_limit += 1

        if problems:
            failures.append((d, problems))
        else:
            n_ok += 1
            if args.verbose:
                print(f"OK   {d}")

    for d, problems in failures:
        print(f"FAIL {d}")
        for p in problems:
            print(f"       {p}")

    print()
    print(f"{n_ok}/{len(dirs)} scenes usable"
          f"  ({n_excluded} excluded by the editor, {len(failures)} failed)")

    for code, hist in sorted(lane_hist.items()):
        print(f"  {code}: obstacle lane index {dict(sorted(hist.items()))}")

    if road_kmh:
        print(f"  road speed (km/h) -> scenes: {dict(sorted(road_kmh.items()))}")
        print(f"  above the catalog's {CATALOG_MAX_ROAD_KMH} km/h cut: {over_limit}"
              f" scene(s) would be dropped before training")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
