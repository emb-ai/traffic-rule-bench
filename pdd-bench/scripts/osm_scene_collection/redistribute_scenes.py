#!/usr/bin/env python3
"""Redistribute collected scenes between sign-code directories to EQUALIZE counts.

The scene geometry (.net.xml) is sign-agnostic; a scene's sign code is just its
parent directory name (`<code>/sign_<id>/`), and `scene_id = sumo_<code>_<id>`
is derived from that at enumeration time. So a scene can be "moved" to another
code simply by moving its folder and rewriting `meta.json["sign_type"]`.

User-requested flows (donors -> recipients):
    3.24 -> 4.6     (max-speed road repurposed as a min-speed scene)
    3.24 -> 5.31    (max-speed road repurposed as a zone-speed-limit scene)
    5.21 -> 5.31    (residential-zone scene repurposed as a zone-speed-limit)

Goal: each of {3.24, 5.21, 5.31, 4.6} ends with N scenes, N = floor(total/4)
(total = sum of the four counts), conserving the total (pure MOVE, no copies).
The `total mod 4` remainder is parked in 5.31 deterministically.

Geometric constraint: 4.6 needs the road to allow the minimum speed, so 3.24->4.6
donors are filtered to road OSM speed >= --min-road-kmh (default 20). The catalog
then clamps the enforced minimum up to the 20 km/h floor (see sumo_catalog.py).
5.31 has no road-type requirement (it is a standalone forward zone, not resnapped
like 5.21), so 3.24->5.31 and 5.21->5.31 need no placement fix.

Dry-run by default; pass --apply to actually move folders. Deterministic donor
selection (sorted by sign_id) so a re-run reproduces the same split.

Usage:
    # preview the split
    python redistribute_scenes.py --root scenes_balanced
    # apply
    python redistribute_scenes.py --root scenes_balanced --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, NamedTuple, Optional

# Sign codes in the equalized set and the donation flows.
SP = "3.24"   # max speed limit  (donor -> 4.6, 5.31)
RZ = "5.21"   # residential zone (donor -> 5.31)
ZL = "5.31"   # zone speed limit (recipient; also absorbs the remainder)
MS = "4.6"    # minimum speed    (recipient)
EQUALIZED = [SP, RZ, ZL, MS]


class Scene(NamedTuple):
    code: str
    sign_id: int
    dir: Path           # <root>/<code>/sign_<id>
    meta_path: Path
    net_abs: Path
    road_id: str


def _edge_speed_kmh(net_path: Path, road_id: str) -> float:
    """Max lane speed (km/h) of `road_id` in a SUMO .net.xml, or 0.0.

    Self-contained copy of sumo_catalog.edge_speed_mps (×3.6) so this script
    has no dependency on the per_sign_bench package.
    """
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return 0.0
    for edge in root.findall("edge"):
        if edge.get("id") == road_id:
            speeds = [float(l.get("speed")) for l in edge.findall("lane") if l.get("speed")]
            if speeds:
                return max(speeds) * 3.6
            s = edge.get("speed")
            return (float(s) * 3.6) if s else 0.0
    return 0.0


def enumerate_code(root: Path, code: str) -> List[Scene]:
    """All scenes under <root>/<code>/sign_*/ with a readable meta.json, by sign_id."""
    code_dir = root / code
    out: List[Scene] = []
    if not code_dir.is_dir():
        return out
    for scene_dir in sorted(code_dir.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("sign_"):
            continue
        meta_path = scene_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        net_file = meta.get("net_file")
        if not net_file:
            continue
        try:
            sign_id = int(meta.get("sign_id", scene_dir.name.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
        out.append(Scene(
            code=code,
            sign_id=sign_id,
            dir=scene_dir,
            meta_path=meta_path,
            net_abs=scene_dir / net_file,
            road_id=str(meta.get("road_id", "")),
        ))
    out.sort(key=lambda s: s.sign_id)
    return out


def _counts_line(root: Path) -> str:
    parts = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n = sum(1 for s in d.iterdir() if s.is_dir() and s.name.startswith("sign_"))
        parts.append(f"{d.name}={n}")
    return "  ".join(parts)


def move_scene(scene: Scene, new_code: str, root: Path, apply: bool) -> bool:
    """Move scene dir into <root>/<new_code>/ and set meta sign_type. Returns done."""
    dst_dir = root / new_code / scene.dir.name
    if dst_dir.exists():
        print(f"  COLLISION skip: {new_code}/{scene.dir.name} already exists "
              f"(from {scene.code})", file=sys.stderr)
        return False
    if not apply:
        return True
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(scene.dir), str(dst_dir))
    # keep meta.json consistent with the new directory/code
    meta_path = dst_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
        meta["sign_type"] = new_code
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"  WARN: moved {dst_dir} but failed to rewrite sign_type: {exc}",
              file=sys.stderr)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Scenes root to mutate IN PLACE (use a COPY, e.g. scenes_balanced).")
    ap.add_argument("--target", default="auto",
                    help="Equal count N per code: 'auto' = floor(total/4), or an integer.")
    ap.add_argument("--min-road-kmh", type=float, default=20.0,
                    help="3.24->4.6 donors must have road OSM speed >= this (default 20).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Unused for selection (deterministic by sign_id); kept for parity.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually move folders (default: dry-run preview).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(2)

    sp = enumerate_code(root, SP)
    rz = enumerate_code(root, RZ)
    zl = enumerate_code(root, ZL)
    ms = enumerate_code(root, MS)
    a, b, c, d = len(sp), len(rz), len(zl), len(ms)
    total = a + b + c + d

    print(f"root: {root}")
    print(f"counts (all codes): {_counts_line(root)}")
    print(f"equalized four: {SP}={a}  {RZ}={b}  {ZL}={c}  {MS}={d}  total={total}")

    if total == 0:
        print("ERROR: no scenes in the four equalized codes.", file=sys.stderr)
        sys.exit(2)

    if args.target == "auto":
        N = total // 4
    else:
        try:
            N = int(args.target)
        except ValueError:
            print(f"ERROR: --target must be 'auto' or an integer, got {args.target!r}",
                  file=sys.stderr)
            sys.exit(2)
    rem = total - 4 * N            # parked in 5.31; for 'auto' rem in 0..3

    # Targets: 3.24=N, 5.21=N, 4.6=N, 5.31=N+rem.
    n_sp = n_rz = n_ms = N
    n_zl = N + rem
    x_46 = n_ms - d               # 3.24 -> 4.6
    y_531 = b - n_rz              # 5.21 -> 5.31
    x_531a = (n_zl - c) - y_531   # 3.24 -> 5.31

    print(f"\ntarget N={N} (5.31 absorbs remainder -> {n_zl})")
    print(f"planned moves:  3.24->4.6 x_46={x_46}   "
          f"5.21->5.31 y_531={y_531}   3.24->5.31 x_531a={x_531a}")

    # Feasibility (pure move, conserve total).
    problems = []
    if x_46 < 0:
        problems.append(f"4.6 already has {d} > N={N}; can't shrink by moving here.")
    if y_531 < 0:
        problems.append(f"5.21 has {b} < N={N}; it can't be a 5.31 donor (would need to receive).")
    if x_531a < 0:
        problems.append(f"3.24->5.31 negative ({x_531a}); 5.21 alone over-fills 5.31.")
    sp_final = a - x_46 - x_531a
    if sp_final != n_sp:
        problems.append(f"3.24 would end at {sp_final}, not N={n_sp} "
                        f"(only N=floor(total/4) is exactly feasible by pure move).")

    # 4.6 donor pool: 3.24 scenes whose road is fast enough.
    ms_eligible = [s for s in sp if _edge_speed_kmh(s.net_abs, s.road_id) >= args.min_road_kmh]
    if x_46 > len(ms_eligible):
        problems.append(f"need {x_46} 3.24->4.6 donors with road>= {args.min_road_kmh:g} km/h, "
                        f"only {len(ms_eligible)} eligible.")

    if problems:
        print("\nINFEASIBLE for an exact equal split via pure move:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("Pick a smaller --target N (<= floor(total/4)) or adjust flows.",
              file=sys.stderr)
        sys.exit(1)

    # Deterministic donor selection.
    ms_donors = ms_eligible[:x_46]
    used_ms_ids = {s.sign_id for s in ms_donors}
    sp_rest = [s for s in sp if s.sign_id not in used_ms_ids]
    zl_donors_sp = sp_rest[:x_531a]
    zl_donors_rz = rz[:y_531]

    print(f"\nselected donors: 4.6<-3.24 {len(ms_donors)}   "
          f"5.31<-3.24 {len(zl_donors_sp)}   5.31<-5.21 {len(zl_donors_rz)}")
    for label, donors in (("3.24->4.6", ms_donors[:3]),
                          ("3.24->5.31", zl_donors_sp[:3]),
                          ("5.21->5.31", zl_donors_rz[:3])):
        ids = ", ".join(f"sign_{s.sign_id}" for s in donors)
        if ids:
            print(f"  e.g. {label}: {ids} ...")

    if not args.apply:
        print(f"\nDRY-RUN. expected after: {SP}={n_sp}  {RZ}={n_rz}  "
              f"{ZL}={n_zl}  {MS}={n_ms}")
        print("re-run with --apply to move.")
        return

    moved = 0
    for s in ms_donors:
        moved += move_scene(s, MS, root, apply=True)
    for s in zl_donors_sp:
        moved += move_scene(s, ZL, root, apply=True)
    for s in zl_donors_rz:
        moved += move_scene(s, ZL, root, apply=True)

    print(f"\nMOVED {moved} scenes.")
    print(f"counts after: {_counts_line(root)}")


if __name__ == "__main__":
    main()
