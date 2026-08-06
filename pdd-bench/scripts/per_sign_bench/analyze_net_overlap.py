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


def _sign_key(code: str) -> tuple:
    """Order sign codes numerically: 2.3.1 < 2.4 < 3.24 < 4.2.1 < 5.7.1."""
    return tuple((0, int(p)) if p.isdigit() else (1, 0, p)
                 for p in str(code).split("."))


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
        "hist": {(f"{lo}+" if hi >= 10**9 else str(lo) if lo == hi else f"{lo}-{hi}"):
                 {"nets": v[0], "scenes": v[1]} for (lo, hi), v in hist.items()},
        "per_class": {c: {"nets": v[0], "scenes": v[1], "nets_touched": v[2],
                          "scenes_touched": v[3],
                          "nets_pct": 100.0 * v[2] / (v[0] or 1),
                          "scenes_pct": 100.0 * v[3] / (v[1] or 1)}
                      for c, v in sorted(per_class.items())},
        "worst": worst[:10],
    }


# --- charts -----------------------------------------------------------------
# Validated categorical slots (light surface): blue, orange. Bars carry direct
# value labels, so the palette's contrast relief rule is satisfied.
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, AXIS = "#e5e5e2", "#c9c9c4"
SERIES = {"fragment": "#2a78d6", "route": "#eb6834"}


def _axes(plt, title: str, ylabel: str, size=(9.0, 4.6)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, pad=14, loc="left")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    return fig, ax


def _grouped_bars(ax, labels, series: dict[str, list[float]], fmt="{:.1f}%"):
    n = len(series)
    # Narrow groups when there are few categories, else bars become slabs.
    group = 0.7 if len(labels) >= 4 else 0.45
    width = group / n
    for i, (name, values) in enumerate(series.items()):
        xs = [x - group / 2 + width * (i + 0.5) for x in range(len(labels))]
        ax.bar(xs, values, width * 0.92, label=name, color=SERIES.get(name, "#2a78d6"))
        for x, v in zip(xs, values):
            ax.annotate(fmt.format(v), (x, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    if n > 1:
        # Top-right of the title band: inside the axes it collides with tall bars.
        ax.legend(frameon=False, fontsize=9, loc="lower right", ncols=n,
                  bbox_to_anchor=(1, 1.0), labelcolor=INK_2, handlelength=1.2)


def write_summary_chart(comparisons: list[dict], out_dir: Path) -> list[str]:
    """The summary table as one chart: route-level overlap per pair, split into
    same-class and other-class-only. Fragment level is left out on purpose — in a
    dense grid it saturates and says nothing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[charts] matplotlib not available — skipping", file=sys.stderr)
        return []

    rows = [r for r in comparisons if r["level"] == "route"]
    if not rows:
        return []
    labels = [f"{r['left_name']}\n-> {r['right_name']}" for r in rows]
    other = [r["touched"]["other_class_only"]["scenes_pct"] for r in rows]
    same = [r["touched"]["any"]["scenes_pct"] - o for r, o in zip(rows, other)]

    fig, ax = _axes(plt, "Test scenes driving a road also driven in the other set",
                    "% of scenes", size=(max(8.0, 1.6 * len(rows)), 4.8))
    xs = range(len(rows))
    # One hue: the whole bar is the route measure, so orange stays "route"
    # everywhere in the report. The subset is separated by texture, not by a
    # second hue that would collide with fragment/route in the other charts.
    ax.bar(xs, same, 0.55, label="same sign class", color=SERIES["route"])
    # 2px surface gap between stacked segments.
    ax.bar(xs, other, 0.55, bottom=[s + 0.35 for s in same],
           label="other class only", color=SERIES["route"],
           hatch="///", edgecolor=SURFACE, linewidth=0)
    for x, (s, o) in enumerate(zip(same, other)):
        ax.annotate(f"{s + o:.1f}%", (x, s + o), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(1.0, max(s + o for s, o in zip(same, other))) * 1.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncols=2,
              bbox_to_anchor=(1, 1.0), labelcolor=INK_2, handlelength=1.2)
    fig.tight_layout()
    p = out_dir / "summary_route.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    written = [p.name]

    # Same measure per sign class. Only train->test pairs: a sign belongs to one
    # set, so every class gets exactly one bar and no series are needed.
    per_sign: dict[str, float] = {}
    for r in rows:
        left_set, left_split = r["left_name"].split("/")
        right_set, right_split = r["right_name"].split("/")
        if left_set != right_set or left_split != "train" or right_split != "test":
            continue
        for code, d in r["per_class"].items():
            per_sign[code] = d["scenes_pct"]
    if not per_sign:
        return written

    codes = sorted(per_sign, key=_sign_key)
    vals = [per_sign[c] for c in codes]
    fig, ax = _axes(plt, "Test scenes on a shared route road, by sign",
                    "% of scenes", size=(max(8.0, 1.1 * len(codes)), 4.4))
    ax.bar(range(len(codes)), vals, 0.55, color=SERIES["route"])
    for x, v in enumerate(vals):
        ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=9)
    ax.set_ylim(0, max(1.0, max(vals)) * 1.25)
    fig.tight_layout()
    p = out_dir / "summary_by_sign.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    written.append(p.name)
    return written


def write_charts(comparisons: list[dict], out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("[charts] matplotlib not available — skipping", file=sys.stderr)
        return []

    by_pair: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in comparisons:
        by_pair[f"{r['left_name']} -> {r['right_name']}"][r["level"]] = r
    pairs = list(by_pair)
    levels = ["fragment", "route"]
    written = []

    fig, ax = _axes(plt, "Scenes standing on a road seen in the other set",
                    "% of scenes")
    _grouped_bars(ax, pairs,
                  {lv: [by_pair[p].get(lv, {}).get("touched", {}).get("any", {})
                        .get("scenes_pct", 0.0) for p in pairs] for lv in levels})
    ax.set_ylim(0, 105)
    fig.tight_layout()
    p = out_dir / "overlap_any.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    written.append(p.name)

    for pair, per_level in by_pair.items():
        ref = per_level.get("fragment") or next(iter(per_level.values()))
        codes = list(ref["per_class"])
        if not codes:
            continue
        fig, ax = _axes(plt, f"By sign class — {pair}", "% of scenes")
        _grouped_bars(ax, codes,
                      {lv: [per_level.get(lv, {}).get("per_class", {})
                            .get(c, {}).get("scenes_pct", 0.0) for c in codes]
                       for lv in levels})
        ax.set_ylim(0, 105)
        fig.tight_layout()
        p = out_dir / f"overlap_by_class_{pair.replace(' -> ', '_to_').replace('/', '-')}.png"
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        written.append(p.name)

        buckets = list(ref["hist"])
        fig, ax = _axes(plt, f"How many roads a scene shares — {pair}", "scenes")
        _grouped_bars(ax, buckets,
                      {lv: [per_level.get(lv, {}).get("hist", {}).get(b, {})
                            .get("scenes", 0) for b in buckets] for lv in levels},
                      fmt="{:.0f}")
        ax.set_xlabel("shared roads per scene", color=INK_2, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        p = out_dir / f"overlap_hist_{pair.replace(' -> ', '_to_').replace('/', '-')}.png"
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        written.append(p.name)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", required=True,
                    metavar="NAME:TRAIN:TEST:SCENES_ROOT")
    ap.add_argument("--out-dir", type=Path, default=Path("net_overlap"))
    ap.add_argument("--no-charts", action="store_true", help="skip PNG charts")
    ap.add_argument("--brief", action="store_true",
                    help="summary table only — no per-pair breakdowns, no charts")
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

    by_pair: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in comparisons:
        by_pair[f"{r['left_name']} -> {r['right_name']}"][r["level"]] = r

    md = ["# Road overlap between scene sets (net level)", "",
          "`fragment` = every edge of the cut-out net; `route` = edges the ego drives",
          "(BFS sign edge -> destination edge). Edge ids normalised to OSM way,",
          "so both directions and segment splits of a road count as one.", "",
          "`A -> B` reads: scenes of B standing on a road that also occurs in A.", "",
          "## Summary", "",
          "| Pair | Shared roads (frag / route) | Scenes on a shared road (frag / route) "
          "| Route-shared with other class only |",
          "|---|---|---|---|"]
    for p, lv in by_pair.items():
        f_, r_ = lv.get("fragment", {}), lv.get("route", {})
        ft = f_.get("touched", {}).get("any", {})
        rt = r_.get("touched", {}).get("any", {})
        other = r_.get("touched", {}).get("other_class_only", {})
        md.append(f"| {p} | {f_.get('shared_ways', 0)} / {r_.get('shared_ways', 0)} | "
                  f"{ft.get('scenes_pct', 0):.1f}% / {rt.get('scenes_pct', 0):.1f}% | "
                  f"{other.get('scenes_pct', 0):.1f}% |")
    md.append("")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_chart = [] if args.no_charts else write_summary_chart(comparisons, args.out_dir)
    if summary_chart:
        md += [f"![{c}]({c})" for c in summary_chart] + [""]

    if args.brief:
        (args.out_dir / "net_overlap.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        (args.out_dir / "net_overlap.json").write_text(json.dumps(comparisons, indent=2),
                                                        encoding="utf-8")
        print("\n".join(md))
        print(f"\nWrote {args.out_dir / 'net_overlap.md'}"
              + (f" (+{len(summary_chart)} chart)" if summary_chart else ""))
        return

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
    charts = [] if args.no_charts else write_charts(comparisons, args.out_dir)
    if charts:
        md += ["## Charts", ""] + [f"![{c}]({c})" for c in charts] + [""]

    (args.out_dir / "net_overlap.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (args.out_dir / "net_overlap.json").write_text(json.dumps(comparisons, indent=2),
                                                    encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {args.out_dir / 'net_overlap.md'}"
          + (f" (+{len(charts)} charts)" if charts else ""))


if __name__ == "__main__":
    main()
