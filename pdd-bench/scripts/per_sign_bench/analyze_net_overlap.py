#!/usr/bin/env python3
"""Geometric overlap between scene sets, measured on the SUMO nets themselves.

Catalog fields only tell which road the SIGN stands on. A scene is a cut-out
fragment holding dozens of edges, so two scenes on different sign roads can
still share an intersection or a whole block. This script measures that, on two
levels:

  fragment  every non-internal edge of the scene net
  route     only edges the ego actually drives: BFS over <connection> from the
            sign edge (road_id) to the destination edge (destination_lane_id)

Edge ids are OSM-derived ("-399521721#2"), so they are comparable across
scenes; they are normalised to a bare way id (sign and "#segment" dropped) so
that opposite directions and segment splits of one road collapse together.

Reported per ordered pair of scene sets (train->test, speed->detour, ...):
  * shared roads (OSM ways) and how many scenes of each side stand on them
  * how many right-hand scenes share at least one road with the left-hand side,
    split by whether the road is shared with the SAME sign class or another one
  * how many roads a scene shares, as a distribution

  python3 analyze_net_overlap.py \\
      --set speed:<train80.jsonl>:<test20.jsonl>:<scenes_root> \\
      --set detour:<train80.jsonl>:<test20.jsonl>:<scenes_root> \\
      --out-dir net_overlap
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_WAY_RE = re.compile(r"^-?([^#]+)")
BUCKETS = ((1, 1), (2, 5), (6, 20), (21, 10**9))


def way_of(edge_id: str) -> str | None:
    """'-399521721#2' -> '399521721'. Internal edges (':j0_1') are dropped."""
    if not edge_id or edge_id.startswith(":"):
        return None
    m = _WAY_RE.match(edge_id)
    return m.group(1) if m else None


def edge_of_lane(lane_id: str) -> str:
    """'lane_-123260010#2_0' -> '-123260010#2'."""
    lane = str(lane_id)
    if lane.startswith("lane_"):
        lane = lane[len("lane_"):]
    return lane.rsplit("_", 1)[0] if "_" in lane else lane


def parse_net(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Return (all edge ids, connection graph from->to). Streaming parse: the
    nets are small individually but there are thousands of them."""
    edges: set[str] = set()
    graph: dict[str, set[str]] = collections.defaultdict(set)
    for _, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag == "edge":
            eid = elem.get("id")
            if eid and elem.get("function") != "internal":
                edges.add(eid)
            elem.clear()
        elif elem.tag == "connection":
            src, dst = elem.get("from"), elem.get("to")
            if src and dst:
                graph[src].add(dst)
            elem.clear()
    return edges, graph


def route_edges(graph: dict[str, set[str]], start: str, goal: str,
                max_depth: int = 60) -> list[str]:
    """Shortest edge path start->goal over connections; [] if unreachable."""
    if start == goal:
        return [start]
    prev: dict[str, str] = {start: ""}
    frontier = [start]
    for _ in range(max_depth):
        nxt = []
        for e in frontier:
            for succ in graph.get(e, ()):
                if succ in prev:
                    continue
                prev[succ] = e
                if succ == goal:
                    path, cur = [goal], goal
                    while prev[cur]:
                        cur = prev[cur]
                        path.append(cur)
                    return path[::-1]
                nxt.append(succ)
        if not nxt:
            break
        frontier = nxt
    return []


def load_scenes(catalog: Path, scenes_root: Path, name: str) -> list[dict]:
    """One entry per unique net (scene variants share the map)."""
    by_net: dict[str, dict] = {}
    with catalog.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            np_ = row.get("net_path")
            if np_:
                by_net.setdefault(np_, {"row": row, "n_variants": 0})
                by_net[np_]["n_variants"] += 1

    out, missing, unrouted = [], 0, 0
    for np_, rec in by_net.items():
        net_file = scenes_root / np_
        if not net_file.is_file():
            missing += 1
            continue
        edges, graph = parse_net(net_file)
        frag = {w for w in (way_of(e) for e in edges) if w}
        row = rec["row"]
        start, dest = row.get("road_id"), row.get("destination_lane_id")
        path = route_edges(graph, str(start), edge_of_lane(dest)) if start and dest else []
        if not path:
            unrouted += 1
        route = {w for w in (way_of(e) for e in path) if w} or ({way_of(str(start))} - {None})
        out.append({"net_path": np_, "scene_id": row.get("scene_id"),
                    "sign_code": str(row.get("sign_code")),
                    "n_variants": rec["n_variants"],
                    "fragment": frag, "route": route})
    print(f"[{name}] nets={len(by_net)} loaded={len(out)} missing={missing} "
          f"no-route={unrouted}", file=sys.stderr)
    return out


def compare(left: list[dict], right: list[dict], level: str) -> dict:
    """Which roads of the right-hand set also occur on the left-hand side."""
    index: dict[str, set[int]] = collections.defaultdict(set)
    for i, s in enumerate(left):
        for w in s[level]:
            index[w].add(i)

    ways_left = set(index)
    ways_right: set[str] = set()
    for s in right:
        ways_right |= s[level]
    shared_ways = ways_left & ways_right

    n_nets = len(right) or 1
    n_scenes = sum(s["n_variants"] for s in right) or 1
    touched = {"any": [0, 0], "same_class": [0, 0], "other_class_only": [0, 0]}
    hist = {b: [0, 0] for b in BUCKETS}
    per_class: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    worst = []

    for s in right:
        hit_ways = s[level] & shared_ways
        code = s["sign_code"]
        stat = per_class[code]
        stat[0] += 1
        stat[1] += s["n_variants"]
        if not hit_ways:
            continue
        same = any(left[i]["sign_code"] == code
                   for w in hit_ways for i in index[w])
        for key in ("any", "same_class" if same else "other_class_only"):
            touched[key][0] += 1
            touched[key][1] += s["n_variants"]
        stat[2] += 1
        stat[3] += s["n_variants"]
        for lo, hi in BUCKETS:
            if lo <= len(hit_ways) <= hi:
                hist[(lo, hi)][0] += 1
                hist[(lo, hi)][1] += s["n_variants"]
                break
        partners = sorted({left[i]["scene_id"] for w in hit_ways for i in index[w]})
        worst.append({"scene": s["scene_id"], "sign": code,
                      "shared_roads": len(hit_ways),
                      "roads": sorted(hit_ways)[:5],
                      "train_scenes": partners[:3],
                      "train_signs": sorted({left[i]["sign_code"]
                                             for w in hit_ways for i in index[w]})})

    worst.sort(key=lambda d: -d["shared_roads"])
    return {
        "level": level, "n_left": len(left), "n_right": len(right),
        "ways_left": len(ways_left), "ways_right": len(ways_right),
        "shared_ways": len(shared_ways),
        "touched": {k: {"nets": v[0], "nets_pct": 100.0 * v[0] / n_nets,
                        "scenes": v[1], "scenes_pct": 100.0 * v[1] / n_scenes}
                    for k, v in touched.items()},
        "hist": {f"{lo}-{hi if hi < 10**9 else '+'}": {"nets": v[0], "scenes": v[1]}
                 for (lo, hi), v in hist.items()},
        "per_class": {c: {"nets": v[0], "scenes": v[1], "nets_touched": v[2],
                          "scenes_touched": v[3],
                          "nets_pct": 100.0 * v[2] / (v[0] or 1),
                          "scenes_pct": 100.0 * v[3] / (v[1] or 1)}
                      for c, v in sorted(per_class.items())},
        "worst": worst[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", required=True,
                    metavar="NAME:TRAIN:TEST:SCENES_ROOT")
    ap.add_argument("--out-dir", type=Path, default=Path("net_overlap"))
    args = ap.parse_args()

    parts: dict[str, list[dict]] = {}
    for spec in args.sets:
        name, train, test, root = spec.split(":", 3)
        parts[f"{name}/train"] = load_scenes(Path(train), Path(root), f"{name}/train")
        parts[f"{name}/test"] = load_scenes(Path(test), Path(root), f"{name}/test")

    names = list(parts)
    comparisons = []
    for a in names:
        for b in names:
            if a == b:
                continue
            set_a, split_a = a.split("/")
            set_b, split_b = b.split("/")
            same_set_train_test = set_a == set_b and split_a == "train" and split_b == "test"
            cross_set = set_a != set_b and split_b == "test"
            if not (same_set_train_test or cross_set):
                continue
            for level in ("fragment", "route"):
                r = compare(parts[a], parts[b], level)
                r["left_name"], r["right_name"] = a, b
                comparisons.append(r)

    md = ["# Road overlap between scene sets (net level)", "",
          "`fragment` = every edge of the cut-out net; `route` = edges the ego drives",
          "(BFS sign edge -> destination edge). Edge ids normalised to OSM way,",
          "so both directions and segment splits of a road count as one.", "",
          "`A -> B` reads: scenes of B standing on a road that also occurs in A.", ""]
    for r in comparisons:
        t = r["touched"]
        md += [f"## {r['left_name']} -> {r['right_name']} ({r['level']})", "",
               f"nets: {r['n_left']} -> {r['n_right']} | "
               f"roads: {r['ways_left']} vs {r['ways_right']}, "
               f"**shared {r['shared_ways']}**", "",
               "| Shared road with | Nets | % nets | Scenes | % scenes |",
               "|---|---:|---:|---:|---:|"]
        for key, label in (("any", "any train scene"),
                           ("same_class", "same sign class"),
                           ("other_class_only", "other class only")):
            d = t[key]
            md.append(f"| {label} | {d['nets']} | {d['nets_pct']:.1f}% | "
                      f"{d['scenes']} | {d['scenes_pct']:.1f}% |")

        md += ["", "Roads shared per scene:", "",
               "| Roads | Nets | Scenes |", "|---|---:|---:|"]
        for label, d in r["hist"].items():
            md.append(f"| {label} | {d['nets']} | {d['scenes']} |")

        md += ["", "By sign class:", "",
               "| Sign | Nets | Touched | % | Scenes | Touched | % |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for code, d in r["per_class"].items():
            md.append(f"| {code} | {d['nets']} | {d['nets_touched']} | {d['nets_pct']:.1f}% | "
                      f"{d['scenes']} | {d['scenes_touched']} | {d['scenes_pct']:.1f}% |")

        if r["worst"]:
            md += ["", "Most overlapping scenes:", "",
                   "| Scene | Sign | Shared roads | Example roads | Train signs |",
                   "|---|---|---:|---|---|"]
            for w in r["worst"]:
                md.append(f"| {w['scene']} | {w['sign']} | {w['shared_roads']} | "
                          f"{', '.join(w['roads'])} | {', '.join(w['train_signs'])} |")
        md.append("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "net_overlap.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (args.out_dir / "net_overlap.json").write_text(json.dumps(comparisons, indent=2),
                                                    encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {args.out_dir / 'net_overlap.md'}")


if __name__ == "__main__":
    main()
