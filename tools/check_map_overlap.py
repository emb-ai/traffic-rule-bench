#!/usr/bin/env python
"""Geometric / road-graph overlap between scene maps of one split.

The place-id audit (``analysis overlap``) only asks whether two scenes are keyed
to the same ``junction_id`` / ``osm_way_id``. Two crops with different keys can
still show the same block: a segment crop that runs into the junction of a
junction crop, two junction crops 60 m apart, a dual-path crop that contains a
detour segment. This script measures that on the nets themselves.

All official maps are cut from one ``moscow.net.xml`` with the same
``netOffset`` / projection, so ``<location convBoundary>`` is one shared x/y
frame and edge / junction ids are OSM ids comparable across scenes.

Overlap level of a scene pair (strongest wins):

  same_place  identical place id (junction:<id> / way:<id>)
  core        the *target* of one scene lies inside the other map: its sign
              road (osm way) or its junction node is part of the other net
  fragment    the two nets share at least one road (osm way) or junction node
  bbox        bounding boxes intersect but the road graphs share nothing
  none        disjoint

Pairs are formed inside one split (default train), separately for
"within one sign" and "across two signs".

  python tools/check_map_overlap.py                      # train, all signs
  python tools/check_map_overlap.py --split test --out reports/map_overlap_test
  python tools/check_map_overlap.py --signs yield,stop,main --fail-on same_place

Exit status is 1 when a within-sign pair reaches ``--fail-on`` (default
``core``): two train scenes of one sign should never test the same road.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = REPO_ROOT / "data" / "scenes"
LEVELS = ("none", "bbox", "fragment", "core", "same_place")
LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}
NEAR_DUP_IOU = 0.9  # two crops of practically the same area
_WAY_RE = re.compile(r"^-?([^#]+)")

# Optional taxonomy roll-up (behavioral family / semantic group per sign folder).
try:  # pragma: no cover - depends on repo import path
    sys.path.insert(0, str(REPO_ROOT))
    from traffic_bench.scene_collection.analysis.overlap.catalog import (  # type: ignore
        SIGN_FAMILY,
        SIGN_SEMANTIC,
    )
except Exception:  # noqa: BLE001
    SIGN_FAMILY, SIGN_SEMANTIC = {}, {}


# --------------------------------------------------------------------------- parsing


def way_of(edge_id: str) -> Optional[str]:
    """'-399521721#2' -> '399521721'; internal (':j_0') and crosswalk edges dropped."""
    if not edge_id or edge_id.startswith(":") or "cw_node" in edge_id:
        return None
    m = _WAY_RE.match(edge_id)
    return m.group(1) if m else None


def _as_list(value) -> List[str]:
    """meta.json stores lists either natively or as their Python repr."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    s = str(value).strip()
    if s.startswith("["):
        try:
            return [str(v) for v in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            return []
    return [s] if s else []


def _place_id(meta: dict, scene_id: str) -> str:
    way = meta.get("osm_way_id") or meta.get("way_id")
    jid = meta.get("junction_id")
    kind = str(meta.get("scene_kind") or meta.get("crop_kind") or "")
    if scene_id.startswith("seg_") or kind.startswith("segment"):
        if way:
            return f"way:{way}"
        m = re.match(r"^seg_(\d+)", scene_id)
        if m:
            return f"way:{m.group(1)}"
    if jid:
        return f"junction:{jid}"
    if way:
        return f"way:{way}"
    return f"scene:{scene_id}"


def _core(meta: dict, scene_id: str) -> Tuple[Set[str], Set[str]]:
    """(core ways, core junction nodes): the one thing the scene tests.

    segment     the sign road (``osm_way_id``)
    dual_path   the decision junction + the road the ego arrives on (``road_id``)
    junction    the junction node
    roundabout  the ring nodes
    Arms of a junction and the destination road are deliberately *not* core:
    they are long streets shared by every neighbouring crop (that is ``fragment``).
    """
    ways: Set[str] = set()
    nodes: Set[str] = set()
    kind = str(meta.get("scene_kind") or meta.get("crop_kind") or "")
    if scene_id.startswith("seg_") or kind.startswith("segment"):
        if meta.get("osm_way_id"):
            ways.add(str(meta["osm_way_id"]))
        w = way_of(str(meta.get("road_id") or ""))
        if w:
            ways.add(w)
        return ways, nodes
    if meta.get("junction_id"):
        nodes.add(str(meta["junction_id"]))
    for n in _as_list(meta.get("ring_node_ids")):
        nodes.add(str(n))
    if kind == "dual_path" or scene_id.startswith("dual_"):
        w = way_of(str(meta.get("road_id") or ""))
        if w:
            ways.add(w)
    return ways, nodes


@dataclass
class Scene:
    sign: str
    scene_id: str
    split: str
    place: str
    bbox: Tuple[float, float, float, float]
    ways: FrozenSet[str]
    nodes: FrozenSet[str]
    core_ways: FrozenSet[str]
    core_nodes: FrozenSet[str]
    lat: float = 0.0
    lon: float = 0.0

    @property
    def uid(self) -> str:
        return f"{self.sign}/{self.scene_id}"

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def parse_scene(args: Tuple[str, str, str, str]) -> Optional[dict]:
    """Worker: read one scene dir → plain dict (picklable)."""
    sign, scene_id, split, scene_dir = args
    d = Path(scene_dir)
    net = d / "map.net.xml"
    if not net.is_file():
        return None
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    bbox = None
    ways: Set[str] = set()
    nodes: Set[str] = set()
    for _event, el in ET.iterparse(str(net), events=("start",)):
        tag = el.tag
        if tag == "location":
            cb = el.get("convBoundary", "")
            try:
                x0, y0, x1, y1 = (float(v) for v in cb.split(","))
                bbox = (x0, y0, x1, y1)
            except ValueError:
                bbox = None
        elif tag == "edge":
            w = way_of(el.get("id", ""))
            if w:
                ways.add(w)
        elif tag == "junction":
            jid = el.get("id", "")
            if jid and not jid.startswith(":") and not jid.startswith("cw_"):
                nodes.add(jid)
        el.clear()
    if bbox is None:
        return None
    core_ways, core_nodes = _core(meta, scene_id)
    return {
        "sign": sign,
        "scene_id": scene_id,
        "split": split,
        "place": _place_id(meta, scene_id),
        "bbox": bbox,
        "ways": sorted(ways),
        "nodes": sorted(nodes),
        "core_ways": sorted(core_ways),
        "core_nodes": sorted(core_nodes),
        "lat": float(meta.get("latitude") or 0.0),
        "lon": float(meta.get("longitude") or 0.0),
    }


def discover(scenes_root: Path, split: str, signs: Optional[Set[str]]) -> List[Tuple[str, str, str, str]]:
    jobs: List[Tuple[str, str, str, str]] = []
    seen: Set[Path] = set()
    for sign_dir in sorted(p for p in scenes_root.iterdir() if p.is_dir()):
        real = sign_dir.resolve()
        if real in seen:
            continue  # alias symlink (main_road -> main)
        seen.add(real)
        sign = sign_dir.name
        if signs and sign not in signs:
            continue
        pool_path = sign_dir / "moscow_pool.json"
        if pool_path.is_file():
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            recs = [
                (str(r["scene_id"]), str(r.get("split") or "unknown"))
                for r in pool.get("scenes") or []
                if r.get("scene_id")
            ]
        else:
            recs = [(p.name, "unknown") for p in sign_dir.iterdir() if p.is_dir()]
        for scene_id, half in recs:
            if split != "all" and half != split:
                continue
            jobs.append((sign, scene_id, half, str(sign_dir / scene_id)))
    return jobs


# --------------------------------------------------------------------------- pairs


def _bbox_pairs(scenes: List[Scene]) -> Set[Tuple[int, int]]:
    """Indices of scene pairs whose bounding boxes intersect (sweep on x)."""
    order = sorted(range(len(scenes)), key=lambda i: scenes[i].bbox[0])
    active: List[int] = []
    out: Set[Tuple[int, int]] = set()
    for i in order:
        x0, y0, x1, y1 = scenes[i].bbox
        active = [j for j in active if scenes[j].bbox[2] >= x0]
        for j in active:
            bx0, by0, bx1, by1 = scenes[j].bbox
            if by1 >= y0 and y1 >= by0:
                out.add((min(i, j), max(i, j)))
        active.append(i)
    return out


def _index_pairs(scenes: List[Scene], attr: str) -> Set[Tuple[int, int]]:
    """Pairs sharing at least one element of ``attr`` (ways / nodes)."""
    index: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(scenes):
        for key in getattr(s, attr):
            index[key].append(i)
    out: Set[Tuple[int, int]] = set()
    for members in index.values():
        if len(members) > 1:
            out.update(combinations(members, 2))
    return out


def _iou(a, b) -> Tuple[float, float]:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0, 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter, (inter / union if union > 0 else 0.0)


@dataclass
class PairResult:
    a: Scene
    b: Scene
    level: str
    shared_ways: int
    shared_nodes: int
    core_hit: str
    inter_m2: float
    iou: float
    dist_m: float

    @property
    def kind(self) -> str:
        return "within" if self.a.sign == self.b.sign else "across"


def evaluate_pairs(scenes: List[Scene]) -> List[PairResult]:
    cands = _bbox_pairs(scenes) | _index_pairs(scenes, "ways") | _index_pairs(scenes, "nodes")
    results: List[PairResult] = []
    for i, j in sorted(cands):
        a, b = scenes[i], scenes[j]
        shared_w = a.ways & b.ways
        shared_n = a.nodes & b.nodes
        inter, iou = _iou(a.bbox, b.bbox)
        core_bits = []
        if a.core_ways & b.ways or a.core_nodes & b.nodes:
            core_bits.append("a_in_b")
        if b.core_ways & a.ways or b.core_nodes & a.nodes:
            core_bits.append("b_in_a")
        if a.place == b.place:
            level = "same_place"
        elif core_bits:
            level = "core"
        elif shared_w or shared_n:
            level = "fragment"
        elif inter > 0:
            level = "bbox"
        else:
            continue
        (ax, ay), (bx, by) = a.center, b.center
        results.append(
            PairResult(
                a=a,
                b=b,
                level=level,
                shared_ways=len(shared_w),
                shared_nodes=len(shared_n),
                core_hit="+".join(core_bits),
                inter_m2=inter,
                iou=iou,
                dist_m=((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5,
            )
        )
    return results


# --------------------------------------------------------------------------- report


def _bucket(sign_a: str, sign_b: str) -> str:
    if sign_a == sign_b:
        return "within_sign"
    fa, fb = SIGN_FAMILY.get(sign_a), SIGN_FAMILY.get(sign_b)
    sa, sb = SIGN_SEMANTIC.get(sign_a), SIGN_SEMANTIC.get(sign_b)
    if fa is None or fb is None:
        return "unknown_family"
    if fa == fb:
        return "within_behavioral"
    if sa == sb:
        return "within_semantic_diff_family"
    return "across_semantic"


def _count_at_least(pairs: Iterable[PairResult], level: str) -> int:
    rank = LEVEL_RANK[level]
    return sum(1 for p in pairs if LEVEL_RANK[p.level] >= rank)


def build_summary(scenes: List[Scene], pairs: List[PairResult], split: str) -> dict:
    signs = sorted({s.sign for s in scenes})
    n_by_sign = Counter(s.sign for s in scenes)
    within = [p for p in pairs if p.kind == "within"]
    across = [p for p in pairs if p.kind == "across"]

    per_sign = {}
    for sign in signs:
        ps = [p for p in within if p.a.sign == sign]
        per_sign[sign] = {
            "scenes": n_by_sign[sign],
            "family": SIGN_FAMILY.get(sign, ""),
            **{lvl: sum(1 for p in ps if p.level == lvl) for lvl in LEVELS[1:]},
            "ge_fragment": _count_at_least(ps, "fragment"),
            "ge_core": _count_at_least(ps, "core"),
            "near_duplicate": sum(1 for p in ps if p.iou >= NEAR_DUP_IOU),
        }

    cross_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for p in across:
        key = tuple(sorted((p.a.sign, p.b.sign)))
        cross_counts[key][p.level] += 1
    cross_pairs = []
    for (sa, sb), c in cross_counts.items():
        cross_pairs.append(
            {
                "sign_a": sa,
                "sign_b": sb,
                "bucket": _bucket(sa, sb),
                **{lvl: c.get(lvl, 0) for lvl in LEVELS[1:]},
                "ge_fragment": sum(v for k, v in c.items() if LEVEL_RANK[k] >= LEVEL_RANK["fragment"]),
                "ge_core": sum(v for k, v in c.items() if LEVEL_RANK[k] >= LEVEL_RANK["core"]),
            }
        )
    cross_pairs.sort(key=lambda r: (-r["ge_core"], -r["ge_fragment"], r["sign_a"], r["sign_b"]))

    bucket_counts: Dict[str, Counter] = defaultdict(Counter)
    for p in across:
        bucket_counts[_bucket(p.a.sign, p.b.sign)][p.level] += 1

    # scenes whose target is inside many other train maps (any sign)
    degree: Counter = Counter()
    for p in pairs:
        if LEVEL_RANK[p.level] >= LEVEL_RANK["core"]:
            degree[p.a.uid] += 1
            degree[p.b.uid] += 1

    return {
        "split": split,
        "scenes": len(scenes),
        "signs": len(signs),
        "pairs_considered": len(pairs),
        "within": {
            lvl: sum(1 for p in within if p.level == lvl) for lvl in LEVELS[1:]
        },
        "across": {
            lvl: sum(1 for p in across if p.level == lvl) for lvl in LEVELS[1:]
        },
        "near_duplicate": {
            "within": sum(1 for p in within if p.iou >= NEAR_DUP_IOU),
            "across": sum(1 for p in across if p.iou >= NEAR_DUP_IOU),
            "iou_threshold": NEAR_DUP_IOU,
        },
        "across_by_bucket": {b: dict(c) for b, c in bucket_counts.items()},
        "per_sign": per_sign,
        "cross_sign_pairs": cross_pairs,
        "top_scene_degree": degree.most_common(25),
    }


def _md_table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def write_report(out_dir: Path, summary: dict, pairs: List[PairResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (out_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "kind", "level", "sign_a", "scene_a", "sign_b", "scene_b", "place_a", "place_b",
                "core_hit", "shared_ways", "shared_nodes", "bbox_inter_m2", "bbox_iou", "center_dist_m",
            ]
        )
        for p in sorted(pairs, key=lambda p: (-LEVEL_RANK[p.level], p.kind, p.a.uid, p.b.uid)):
            w.writerow(
                [
                    p.kind, p.level, p.a.sign, p.a.scene_id, p.b.sign, p.b.scene_id, p.a.place, p.b.place,
                    p.core_hit, p.shared_ways, p.shared_nodes, f"{p.inter_m2:.0f}", f"{p.iou:.3f}", f"{p.dist_m:.0f}",
                ]
            )

    s = summary
    lines = [
        f"# Map overlap inside the `{s['split']}` split",
        "",
        f"Scenes: **{s['scenes']}** across **{s['signs']}** signs; scene pairs with any contact: {s['pairs_considered']}.",
        "",
        "Levels: `same_place` = same junction / way key · `core` = the target road or junction of one scene is",
        "inside the other map · `fragment` = the two nets share a road or junction node · `bbox` = boxes touch,",
        "graphs disjoint.",
        "",
        "## Totals",
        "",
        _md_table(
            ["Pairs", "same_place", "core", "fragment", "bbox", f"near-duplicate (IoU ≥ {NEAR_DUP_IOU})"],
            [
                ["within one sign", *(s["within"][l] for l in ("same_place", "core", "fragment", "bbox")), s["near_duplicate"]["within"]],
                ["across two signs", *(s["across"][l] for l in ("same_place", "core", "fragment", "bbox")), s["near_duplicate"]["across"]],
            ],
        ),
        "",
        "## Across signs by policy bucket",
        "",
        _md_table(
            ["Bucket", "same_place", "core", "fragment", "bbox"],
            [
                [b, *(c.get(l, 0) for l in ("same_place", "core", "fragment", "bbox"))]
                for b, c in sorted(s["across_by_bucket"].items())
            ]
            or [["—", 0, 0, 0, 0]],
        ),
        "",
        "## Within one sign",
        "",
        _md_table(
            ["Sign", "Family", "Scenes", "same_place", "core", "fragment", "bbox", "near-dup"],
            [
                [f"`{sign}`", r["family"], r["scenes"], r["same_place"], r["core"], r["fragment"], r["bbox"], r["near_duplicate"]]
                for sign, r in sorted(s["per_sign"].items())
            ],
        ),
        "",
        "## Sign pairs (across), sorted by core overlaps",
        "",
        _md_table(
            ["Sign A", "Sign B", "Bucket", "same_place", "core", "fragment", "bbox"],
            [
                [f"`{r['sign_a']}`", f"`{r['sign_b']}`", r["bucket"], r["same_place"], r["core"], r["fragment"], r["bbox"]]
                for r in s["cross_sign_pairs"][:40]
            ]
            or [["—", "—", "—", 0, 0, 0, 0]],
        ),
        "",
        "## Scenes whose target sits inside the most other maps",
        "",
        _md_table(["Scene", "# maps (core level)"], [[f"`{u}`", n] for u, n in s["top_scene_degree"]] or [["—", 0]]),
        "",
        "Full pair list: `pairs.csv`; machine-readable: `summary.json`.",
        "",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES)
    ap.add_argument("--split", choices=("train", "test", "all"), default="train")
    ap.add_argument("--signs", default="", help="comma-separated sign folders (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="default: reports/map_overlap_<split>")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--fail-on", choices=LEVELS[1:], default="core", help="within-sign level that fails the run")
    args = ap.parse_args(argv)

    root = args.scenes_root.expanduser().resolve()
    out_dir = args.out or (REPO_ROOT / "reports" / f"map_overlap_{args.split}")
    signs = {s.strip() for s in args.signs.split(",") if s.strip()} or None

    jobs = discover(root, args.split, signs)
    if not jobs:
        print(f"[map-overlap] no scenes for split={args.split} under {root}", file=sys.stderr)
        return 2
    print(f"[map-overlap] parsing {len(jobs)} nets ({args.split}) with {args.jobs} workers …", flush=True)
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            raw = list(ex.map(parse_scene, jobs, chunksize=16))
    else:
        raw = [parse_scene(j) for j in jobs]
    scenes = [
        Scene(
            sign=r["sign"], scene_id=r["scene_id"], split=r["split"], place=r["place"], bbox=tuple(r["bbox"]),
            ways=frozenset(r["ways"]), nodes=frozenset(r["nodes"]),
            core_ways=frozenset(r["core_ways"]), core_nodes=frozenset(r["core_nodes"]), lat=r["lat"], lon=r["lon"],
        )
        for r in raw
        if r
    ]
    skipped = len(jobs) - len(scenes)
    print(f"[map-overlap] parsed {len(scenes)} scenes ({skipped} unreadable) in {time.time() - t0:.0f}s", flush=True)

    pairs = evaluate_pairs(scenes)
    summary = build_summary(scenes, pairs, args.split)
    report = write_report(out_dir, summary, pairs)

    w, a = summary["within"], summary["across"]
    print(
        f"[map-overlap] within-sign pairs: same_place={w['same_place']} core={w['core']} "
        f"fragment={w['fragment']} bbox={w['bbox']}"
    )
    print(
        f"[map-overlap] across-sign pairs: same_place={a['same_place']} core={a['core']} "
        f"fragment={a['fragment']} bbox={a['bbox']}"
    )
    for b, c in sorted(summary["across_by_bucket"].items()):
        print(f"[map-overlap]   {b:28s} " + " ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: -LEVEL_RANK[kv[0]])))
    nd = summary["near_duplicate"]
    print(f"[map-overlap] near-duplicate maps (IoU >= {NEAR_DUP_IOU}): within-sign={nd['within']} across-sign={nd['across']}")
    print(f"[map-overlap] report: {report}")

    bad = _count_at_least((p for p in pairs if p.kind == "within"), args.fail_on)
    if bad:
        print(f"[map-overlap] FAIL: {bad} within-sign pairs at level >= {args.fail_on}")
        return 1
    print(f"[map-overlap] OK: no within-sign pair at level >= {args.fail_on}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
