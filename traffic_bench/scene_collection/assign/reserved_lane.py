"""Reserved-lane scene set: bus_lane / bike_lane (5.14.1/2) and bus_lane_road / bike_lane_road (5.11.1/2).

    python -m traffic_bench.scene_collection reserved-lane select      # maps/index/segments.jsonl -> assignment
    python -m traffic_bench.scene_collection reserved-lane crop        # netconvert into maps/crops/segment/
    python -m traffic_bench.scene_collection reserved-lane materialize # data/scenes/<sign>/ + moscow_pool.json
    python -m traffic_bench.scene_collection reserved-lane report      # histograms / distances
    python -m traffic_bench.scene_collection reserved-lane all

Selection rules (all deterministic under ``--seed``):

* candidates: multi-lane segments (>= 2 vehicle lanes, lane 0 drivable, long enough
  for the nominal geometry), straight or curved; 5.11.x need a true one-way
  street (``collect.segments.oneway``), 5.14.x a street with an opposite
  carriageway nearby;
* split: the global by-``osm_way_id`` split (``split_segments_by_type``, seed 42),
  so a way is train or test consistently across the whole benchmark;
* quotas per (sign, split): 80 / 20 maps, equal thirds of 2 / 3 / 4+ lanes,
  straight and curved alternating while the pool has curved maps;
* places: ``--reuse-policy unique`` (default) admits only ways / junctions that no
  official sign uses in either split; ``detour`` allows same-split detour places,
  ``any`` any same-split place; other-split places are never admitted;
* geometry: crop windows pairwise disjoint (10 m clearance) with every map of the
  family and every other-split official segment; map centres >= ``--min-center-m``
  apart inside the family and >= ``--min-official-m`` from other-split official
  maps; ``geo_cell`` caps (1 per sign, 3 per family);
* relaxation when a bucket runs dry: geo caps doubled -> geo caps off -> distances
  halved -> neighbouring lane bucket -> (5.14 only) one-way streets. Every pick
  records the level that admitted it; shortfalls are reported, never hidden.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from traffic_bench.scene_collection.collect.segments.crop import CROP_MARGIN_M, _crop_one, compute_crop_bbox
from traffic_bench.scene_collection.collect.segments.oneway import scan_true_oneway
from traffic_bench.scene_collection.collect.segments.select import split_segments_by_type
from traffic_bench.scene_collection.paths import (
    DATA_SCENES,
    INDEX,
    MOSCOW_NET,
    REPO_ROOT,
    SEGMENT_CROPS,
    SEGMENTS_INDEX,
)
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import save_moscow_pool


@dataclass(frozen=True)
class SignSpec:
    sign: str
    pdd_code: str
    kind: str      # "oneway" | "twoway"
    flow: str      # "opposite" | "same"
    lane_user: str # "bus" | "bicycle"


SIGNS: Tuple[SignSpec, ...] = (
    SignSpec("bus_lane_road", "5.11.1", "oneway", "opposite", "bus"),
    SignSpec("bike_lane_road", "5.11.2", "oneway", "opposite", "bicycle"),
    SignSpec("bus_lane", "5.14.1", "twoway", "same", "bus"),
    SignSpec("bike_lane", "5.14.2", "twoway", "same", "bicycle"),
)
SIGN_BY_NAME = {s.sign: s for s in SIGNS}
BUCKETS = ("2", "3", "4plus")
SUBTYPES = ("curved", "straight")
SPLITS = ("test", "train")
DEFAULT_MIN_LENGTH_M = 160.0      # 10 + approach 60 + zone 60 + tail 10 + 15 m plate slide
DEFAULT_ASSIGNMENT = INDEX / "reserved_lane_assignment_v3.json"
DEFAULT_ONEWAY_CACHE = INDEX / "oneway_flags.json"
DEFAULT_CATALOG = REPO_ROOT / "data" / "metadata" / "catalog.jsonl"
DEFAULT_ALLOCATIONS = REPO_ROOT / "data" / "metadata" / "sign_allocations.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "restricted_lane_scenes_v3"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "data" / "scenes_archive"
BOX_CLEARANCE_M = 10.0
DETOUR_PREFIX = "4.2"

# Relaxation ladder: (geo cap per sign, geo cap per family, centre factor).
LEVELS: Tuple[Tuple[Optional[int], Optional[int], float], ...] = (
    (1, 3, 1.0),
    (2, 6, 1.0),
    (None, None, 1.0),
    (None, None, 0.5),
)
LEVEL_NAMES = ("strict", "geo_cap_x2", "no_geo_cap", "half_distance", "borrow_bucket", "oneway_fallback")


# --------------------------------------------------------------------------- utils

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def boxes_overlap(a: Sequence[float], b: Sequence[float], clearance: float = BOX_CLEARANCE_M) -> bool:
    return not (a[2] + clearance < b[0] or b[2] + clearance < a[0] or a[3] + clearance < b[1] or b[3] + clearance < a[1])


def rl_bucket(lane_count: int) -> str:
    n = int(lane_count or 0)
    return "2" if n <= 2 else ("3" if n == 3 else "4plus")


def split_quotas(n_total: int) -> Dict[str, int]:
    base, rem = divmod(int(n_total), len(BUCKETS))
    return {b: base + (1 if i < rem else 0) for i, b in enumerate(BUCKETS)}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- candidates

@dataclass
class Candidate:
    row: Dict[str, Any]
    scene_id: str
    edge_id: str
    way: str
    junction: str
    lane_count: int
    bucket: str
    segment_type: str
    geo_cell: str
    length_m: float
    lat: float
    lon: float
    box: Tuple[float, float, float, float]
    split: Optional[str] = None
    oneway: Optional[bool] = None


def candidate_rows(rows: Iterable[Dict[str, Any]], *, min_length_m: float) -> List[Candidate]:
    out: List[Candidate] = []
    for r in rows:
        lanes = r.get("vehicle_lane_indices") or []
        if int(r.get("lane_count") or 0) < 2 or not lanes or int(lanes[0]) != 0:
            continue
        if float(r.get("length_m") or 0.0) < float(min_length_m):
            continue
        if str(r.get("segment_type") or "") not in ("straight", "curved"):
            continue
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        box = compute_crop_bbox(tuple(r["start_xy"]), tuple(r["end_xy"]), r.get("window_shape"), margin_m=CROP_MARGIN_M)
        out.append(
            Candidate(
                row=r,
                scene_id=str(r["scene_id"]),
                edge_id=str(r["edge_id"]),
                way=str(r.get("osm_way_id") or "").strip(),
                junction=str(r.get("junction_id") or "").strip(),
                lane_count=int(r.get("lane_count") or 0),
                bucket=rl_bucket(int(r.get("lane_count") or 0)),
                segment_type=str(r.get("segment_type")),
                geo_cell=str(r.get("geo_cell") or ""),
                length_m=float(r.get("length_m") or 0.0),
                lat=float(r["latitude"]),
                lon=float(r["longitude"]),
                box=tuple(float(x) for x in box),  # type: ignore[arg-type]
            )
        )
    return out


def way_split_map(rows: Sequence[Dict[str, Any]], *, test_frac: float, seed: int) -> Dict[str, str]:
    """Reproduce the global segment split (fed the FULL index, as the allocator is)."""
    train_w, test_w = split_segments_by_type(rows, test_frac=test_frac, seed=seed)
    out: Dict[str, str] = {}
    for d, name in ((train_w, "train"), (test_w, "test")):
        for ways in d.values():
            for w in ways:
                out[str(w)] = name
    return out


# ------------------------------------------------------------- official maps

@dataclass
class OfficialMaps:
    owners: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {"train": {}, "test": {}})
    points: List[Tuple[str, float, float, Optional[Tuple[float, float, float, float]], str, str]] = field(default_factory=list)
    n_catalog: int = 0

    def owner_codes(self, place: str, split: str) -> Optional[List[str]]:
        return self.owners.get(split, {}).get(place)


def load_official(
    *, catalog_path: Path, allocations_path: Path, scenes_root: Path, verbose: bool = True
) -> OfficialMaps:
    off = OfficialMaps()
    if allocations_path.is_file():
        alloc = json.loads(allocations_path.read_text(encoding="utf-8"))
        for split in ("train", "test"):
            for place, codes in (alloc.get("place_registry", {}).get(split) or {}).items():
                off.owners[split][str(place)] = [str(c) for c in (codes or [])]
    if catalog_path.is_file():
        rows = _load_jsonl(catalog_path)
        off.n_catalog = len(rows)
        for r in rows:
            split = str(r.get("split") or "")
            if split not in ("train", "test"):
                continue
            sign_id = str(r.get("sign_id") or "")
            scene_id = str(r.get("scene_id") or "")
            lat, lon = r.get("latitude"), r.get("longitude")
            box = None
            meta_path = scenes_root / sign_id / scene_id / "meta.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
                way = str(meta.get("osm_way_id") or "").strip()
                jid = str(meta.get("junction_id") or "").strip()
                if way:
                    off.owners[split].setdefault(f"way:{way}", []).append(str(r.get("pdd_code") or ""))
                if jid:
                    off.owners[split].setdefault(f"junction:{jid}", []).append(str(r.get("pdd_code") or ""))
                if meta.get("crop_bbox"):
                    box = tuple(float(x) for x in meta["crop_bbox"])  # type: ignore[assignment]
                lat = lat if lat is not None else meta.get("latitude")
                lon = lon if lon is not None else meta.get("longitude")
            if lat is None or lon is None:
                continue
            off.points.append((split, float(lat), float(lon), box, sign_id, scene_id))
    if verbose:
        print(
            f"[official] catalog rows {off.n_catalog}; places train {len(off.owners['train'])} "
            f"test {len(off.owners['test'])}; located maps {len(off.points)}"
        )
    return off


# ---------------------------------------------------------------- selection

@dataclass
class Pick:
    cand: Candidate
    sign: str
    split: str
    level: str
    bucket_target: str


class Selector:
    def __init__(
        self,
        *,
        official: OfficialMaps,
        reuse_policy: str,
        min_center_m: float,
        min_official_m: float,
        seed: int,
    ) -> None:
        self.official = official
        self.reuse_policy = reuse_policy
        self.min_center_m = float(min_center_m)
        self.min_official_m = float(min_official_m)
        self.seed = int(seed)
        self.picks: List[Pick] = []
        self.used_ways: Set[str] = set()
        self.used_junctions: Set[str] = set()
        self.cell_sign: Dict[Tuple[str, str], int] = collections.Counter()
        self.cell_family: Dict[str, int] = collections.Counter()
        self.rejections: Dict[str, int] = collections.Counter()
        self._official_by_split: Dict[str, List[Tuple[float, float, Optional[Tuple[float, float, float, float]]]]] = {
            sp: [(lat, lon, box) for osplit, lat, lon, box, _s, _i in official.points if osplit == sp]
            for sp in ("train", "test")
        }

    # ---- constraints
    def _place_ok(self, place: str, split: str) -> bool:
        other = "train" if split == "test" else "test"
        if self.official.owner_codes(place, other) is not None:
            return False
        same = self.official.owner_codes(place, split)
        if same is None:
            return True
        if self.reuse_policy == "any":
            return True
        if self.reuse_policy == "detour":
            return all(c.startswith(DETOUR_PREFIX) for c in same)
        return False

    def admissible(self, c: Candidate, *, sign: str, split: str, level: int) -> Tuple[bool, str]:
        cap_sign, cap_family, dist_factor = LEVELS[level]
        if not c.way or c.way in self.used_ways:
            return False, "way_used"
        if c.junction and c.junction in self.used_junctions:
            return False, "junction_used"
        if not self._place_ok(f"way:{c.way}", split):
            return False, "official_way"
        if c.junction and not self._place_ok(f"junction:{c.junction}", split):
            return False, "official_junction"
        if cap_sign is not None and self.cell_sign[(sign, c.geo_cell)] >= cap_sign:
            return False, "geo_cap_sign"
        if cap_family is not None and self.cell_family[c.geo_cell] >= cap_family:
            return False, "geo_cap_family"
        min_center = self.min_center_m * dist_factor
        min_official = self.min_official_m * dist_factor
        for p in self.picks:
            if boxes_overlap(c.box, p.cand.box):
                return False, "box_family"
            if haversine_m(c.lat, c.lon, p.cand.lat, p.cand.lon) < min_center:
                return False, "near_family"
        other = "train" if split == "test" else "test"
        for lat, lon, box in self._official_by_split.get(other, ()):
            if abs(lat - c.lat) > 0.02 or abs(lon - c.lon) > 0.04:
                continue  # > ~2 km away in both axes: cannot violate either rule
            if haversine_m(c.lat, c.lon, lat, lon) < min_official:
                return False, "near_official_other_split"
            if box is not None and boxes_overlap(c.box, box):
                return False, "box_official_other_split"
        return True, ""

    def _take(self, c: Candidate, *, sign: str, split: str, level_name: str, bucket_target: str) -> None:
        self.picks.append(Pick(c, sign, split, level_name, bucket_target))
        self.used_ways.add(c.way)
        if c.junction:
            self.used_junctions.add(c.junction)
        self.cell_sign[(sign, c.geo_cell)] += 1
        self.cell_family[c.geo_cell] += 1

    def _rng(self, *tags) -> random.Random:
        return random.Random("|".join(str(t) for t in (self.seed, *tags)))

    # ---- one (sign, split)
    def select_for(
        self, spec: SignSpec, split: str, pool: List[Candidate], fallback_pool: List[Candidate], n_total: int
    ) -> Tuple[List[Pick], Dict[str, int]]:
        quotas = split_quotas(n_total)
        taken: List[Pick] = []
        sub: Dict[Tuple[str, str], List[Candidate]] = {}
        for b in BUCKETS:
            for st in SUBTYPES:
                lst = [c for c in pool if c.bucket == b and c.segment_type == st]
                self._rng(spec.sign, split, b, st).shuffle(lst)
                sub[(b, st)] = lst

        def count(b: str, st: Optional[str] = None) -> int:
            return sum(1 for p in taken if p.bucket_target == b and (st is None or p.cand.segment_type == st))

        def try_fill(bucket_target: str, sources: List[Tuple[str, str]], level: int, level_name: str) -> None:
            while count(bucket_target) < quotas[bucket_target]:
                # alternate subtypes: curved whenever it is not ahead of straight
                order = sorted(sources, key=lambda k: (count(bucket_target, k[1]), SUBTYPES.index(k[1])))
                got = False
                for key in order:
                    lst = sub.get(key, [])
                    for i, c in enumerate(lst):
                        ok, why = self.admissible(c, sign=spec.sign, split=split, level=level)
                        if ok:
                            self._take(c, sign=spec.sign, split=split, level_name=level_name, bucket_target=bucket_target)
                            taken.append(self.picks[-1])
                            del lst[i]
                            got = True
                            break
                        self.rejections[why] += 1
                    if got:
                        break
                if not got:
                    return

        for level, name in enumerate(LEVEL_NAMES[: len(LEVELS)]):
            for b in BUCKETS:
                try_fill(b, [(b, st) for st in SUBTYPES], level, name)
            if all(count(b) >= quotas[b] for b in BUCKETS):
                break
        # borrow from neighbouring lane buckets (4plus -> 3 -> 2), strictest distances
        if any(count(b) < quotas[b] for b in BUCKETS):
            for b in BUCKETS:
                if count(b) >= quotas[b]:
                    continue
                neighbours = [x for x in BUCKETS if x != b]
                neighbours.sort(key=lambda x: abs(BUCKETS.index(x) - BUCKETS.index(b)))
                for nb in neighbours:
                    for level in range(len(LEVELS)):
                        try_fill(b, [(nb, st) for st in SUBTYPES], level, "borrow_bucket")
                        if count(b) >= quotas[b]:
                            break
                    if count(b) >= quotas[b]:
                        break
        # 5.14 only: one-way streets when the two-way pool is exhausted
        if fallback_pool and any(count(b) < quotas[b] for b in BUCKETS):
            for b in BUCKETS:
                for st in SUBTYPES:
                    lst = [c for c in fallback_pool if c.bucket == b and c.segment_type == st and c.way not in self.used_ways]
                    self._rng(spec.sign, split, "fallback", b, st).shuffle(lst)
                    sub[("fb_" + b, st)] = lst
            for b in BUCKETS:
                for level in range(len(LEVELS)):
                    try_fill(b, [("fb_" + b, st) for st in SUBTYPES], level, "oneway_fallback")
                    if count(b) >= quotas[b]:
                        break
        shortfall = {b: quotas[b] - count(b) for b in BUCKETS if count(b) < quotas[b]}
        return taken, shortfall


def run_select(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    rows = _load_jsonl(Path(args.index))
    print(f"[select] index rows {len(rows)} from {_rel(Path(args.index))}")
    cands = candidate_rows(rows, min_length_m=args.min_length_m)
    splits = way_split_map(rows, test_frac=args.test_frac, seed=args.split_seed)
    for c in cands:
        c.split = splits.get(c.way)
    cands = [c for c in cands if c.split in ("train", "test")]
    print(f"[select] multi-lane candidates {len(cands)} "
          f"(train {sum(c.split == 'train' for c in cands)} / test {sum(c.split == 'test' for c in cands)})")

    flags = scan_true_oneway(Path(args.net), [c.edge_id for c in cands], cache_path=Path(args.oneway_cache))
    for c in cands:
        c.oneway = flags.get(c.edge_id)
    by = collections.Counter((c.split, c.oneway) for c in cands)
    print(f"[select] by (split, one-way): {dict(sorted(by.items(), key=str))}")
    lanes = collections.Counter((c.oneway, c.bucket) for c in cands)
    print(f"[select] by (one-way, lane bucket): {dict(sorted(lanes.items(), key=str))}")

    official = load_official(
        catalog_path=Path(args.catalog), allocations_path=Path(args.allocations), scenes_root=Path(args.scenes_root)
    )
    sel = Selector(
        official=official,
        reuse_policy=args.reuse_policy,
        min_center_m=args.min_center_m,
        min_official_m=args.min_official_m,
        seed=args.seed,
    )
    n_test = int(round(args.per_sign * args.test_frac))
    n_train = int(args.per_sign) - n_test
    quota = {"test": n_test, "train": n_train}
    assignment: Dict[str, Any] = {
        "mode": "reserved_lane_v3",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "test_frac": args.test_frac,
        "per_sign": args.per_sign,
        "reuse_policy": args.reuse_policy,
        "min_center_m": args.min_center_m,
        "min_official_m": args.min_official_m,
        "min_length_m": args.min_length_m,
        "box_clearance_m": BOX_CLEARANCE_M,
        "crop_margin_m": CROP_MARGIN_M,
        "index": _rel(Path(args.index)),
        "signs": {},
        "shortfalls": {},
    }
    for spec in SIGNS:
        entry: Dict[str, Any] = {"pdd_code": spec.pdd_code, "kind": spec.kind, "flow": spec.flow, "lane_user": spec.lane_user}
        for split in SPLITS:
            if spec.kind == "oneway":
                pool = [c for c in cands if c.split == split and c.oneway is True]
                fallback: List[Candidate] = []
            else:
                pool = [c for c in cands if c.split == split and c.oneway is False]
                fallback = [c for c in cands if c.split == split and c.oneway is True]
            picks, short = sel.select_for(spec, split, pool, fallback, quota[split])
            entry[split] = [
                {
                    "scene_id": p.cand.scene_id,
                    "edge_id": p.cand.edge_id,
                    "osm_way_id": p.cand.way,
                    "junction_id": p.cand.junction or None,
                    "lane_count": p.cand.lane_count,
                    "rl_bucket": p.cand.bucket,
                    "bucket_target": p.bucket_target,
                    "segment_type": p.cand.segment_type,
                    "subtype": p.cand.row.get("subtype"),
                    "geo_cell": p.cand.geo_cell,
                    "length_m": p.cand.length_m,
                    "one_way_street": p.cand.oneway,
                    "latitude": p.cand.lat,
                    "longitude": p.cand.lon,
                    "bbox": list(p.cand.box),
                    "level": p.level,
                }
                for p in picks
            ]
            if short:
                assignment["shortfalls"][f"{spec.sign}/{split}"] = short
            lanes_c = collections.Counter(p.cand.bucket for p in picks)
            types_c = collections.Counter(p.cand.segment_type for p in picks)
            levels_c = collections.Counter(p.level for p in picks)
            print(
                f"  {spec.sign:15s} {split:5s} {len(picks):3d}/{quota[split]}  pool={len(pool):4d}  "
                f"lanes 2/3/4+={lanes_c['2']}/{lanes_c['3']}/{lanes_c['4plus']}  "
                f"straight/curved={types_c['straight']}/{types_c['curved']}  levels={dict(levels_c)}"
                + (f"  SHORT {short}" if short else "")
            )
        assignment["signs"][spec.sign] = entry
    assignment["rejections"] = dict(sel.rejections)
    cells = collections.Counter(p.cand.geo_cell for p in sel.picks)
    assignment["geo_cells_used"] = len(cells)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assignment, indent=1), encoding="utf-8")
    print(f"[select] {len(sel.picks)} maps, {len(cells)} geo cells, rejections {dict(sel.rejections)}")
    print(f"[select] wrote {_rel(out)} in {time.time() - t0:.0f}s")
    return assignment


# --------------------------------------------------------------------- crop

def _iter_picks(assignment: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for sign, entry in assignment["signs"].items():
        for split in ("train", "test"):
            for e in entry.get(split) or []:
                yield sign, split, e


def run_crop(args: argparse.Namespace) -> None:
    assignment = json.loads(Path(args.assignment).read_text(encoding="utf-8"))
    index = {str(r["scene_id"]): r for r in _load_jsonl(Path(args.index))}
    crops_root = Path(args.crops_root)
    crops_root.mkdir(parents=True, exist_ok=True)
    jobs_list = []
    skipped = 0
    for _sign, split, e in _iter_picks(assignment):
        sid = e["scene_id"]
        if (crops_root / sid / "map.net.xml").is_file() and (crops_root / sid / "meta.json").is_file() and not args.force:
            skipped += 1
            continue
        row = dict(index[sid])
        row["split"] = split
        jobs_list.append((row, str(args.net), str(crops_root), False, CROP_MARGIN_M))
    print(f"[crop] {len(jobs_list)} to crop, {skipped} already in {_rel(crops_root)}; jobs={args.jobs}")
    counts: Dict[str, int] = collections.Counter()
    fails: List[Tuple[str, str]] = []
    t0 = time.time()
    if jobs_list:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            for i, (status, sid, info) in enumerate(ex.map(_crop_one, jobs_list), 1):
                counts[status] += 1
                if status == "fail":
                    fails.append((sid, info))
                if i % 25 == 0 or i == len(jobs_list):
                    print(f"[crop] {i}/{len(jobs_list)} {dict(counts)} {time.time() - t0:.0f}s", flush=True)
    print(f"[crop] done {dict(counts)}")
    for sid, info in fails[:20]:
        print(f"  FAIL {sid}: {info}")
    if fails:
        raise SystemExit(f"[crop] {len(fails)} crop(s) failed; fix or re-select before materializing")


# -------------------------------------------------------------- materialize

def run_materialize(args: argparse.Namespace) -> None:
    assignment = json.loads(Path(args.assignment).read_text(encoding="utf-8"))
    scenes_root = Path(args.scenes_root)
    crops_root = Path(args.crops_root)
    archive_dir = Path(args.archive_dir)
    tag = time.strftime("%Y%m%d_%H%M%S")
    for sign, entry in assignment["signs"].items():
        spec = SIGN_BY_NAME[sign]
        dst_sign = scenes_root / sign
        if dst_sign.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            tgz = archive_dir / f"{sign}_{tag}.tgz"
            n_old = sum(1 for p in dst_sign.iterdir() if p.is_dir())
            subprocess.run(["tar", "-czhf", str(tgz), "-C", str(scenes_root), sign], check=True)
            shutil.rmtree(dst_sign)
            print(f"[materialize] archived previous {sign} ({n_old} maps) -> {_rel(tgz)}")
        dst_sign.mkdir(parents=True, exist_ok=True)
        scenes = []
        n_fail = 0
        for split in ("train", "test"):
            for e in entry.get(split) or []:
                sid = e["scene_id"]
                src = crops_root / sid
                if not (src / "map.net.xml").is_file() or not (src / "meta.json").is_file():
                    print(f"  MISSING crop {sid}")
                    n_fail += 1
                    continue
                dst = dst_sign / sid
                if dst.exists() or dst.is_symlink():
                    shutil.rmtree(dst) if dst.is_dir() and not dst.is_symlink() else dst.unlink()
                if args.mode == "symlink":
                    import os
                    dst.symlink_to(os.path.relpath(src.resolve(), start=dst_sign.resolve()))
                else:
                    shutil.copytree(src, dst)
                    meta_path = dst / "meta.json"
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta.update(
                        {
                            "split": split,
                            "one_way_street": e.get("one_way_street"),
                            "rl_bucket": e.get("rl_bucket"),
                            "reserved_lane_sign": spec.pdd_code,
                            "reserved_lane_flow": spec.flow,
                            "reserved_lane_user": spec.lane_user,
                            "selection_level": e.get("level"),
                        }
                    )
                    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                scenes.append(
                    {
                        "scene_id": sid,
                        "shape": e.get("segment_type"),
                        "crop_kind": "segment",
                        "slot": None,
                        "split": split,
                        "path": str(dst),
                        "moscow_path": f"{sign}/{sid}",
                        "one_way_street": e.get("one_way_street"),
                        "lane_count": e.get("lane_count"),
                        "rl_bucket": e.get("rl_bucket"),
                        "geo_cell": e.get("geo_cell"),
                        "subtype": e.get("subtype"),
                        "selection_level": e.get("level"),
                    }
                )
        pool = {
            "sign": spec.pdd_code,
            "split": "all",
            "mode": "reserved_lane_v3",
            "crop_kind": "segment",
            "flow": spec.flow,
            "lane_user": spec.lane_user,
            "reuse_policy": assignment.get("reuse_policy"),
            "selection": _rel(Path(args.assignment)),
            "n_ok": len(scenes),
            "n_fail": n_fail,
            "scenes": scenes,
        }
        save_moscow_pool(dst_sign, pool)
        print(f"[materialize] {sign}: {len(scenes)} maps ({sum(s['split'] == 'train' for s in scenes)} train / "
              f"{sum(s['split'] == 'test' for s in scenes)} test), {n_fail} missing -> {_rel(dst_sign)}")


# ------------------------------------------------------------------- report

def run_report(args: argparse.Namespace) -> None:
    scenes_root = Path(args.scenes_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    official = load_official(
        catalog_path=Path(args.catalog), allocations_path=Path(args.allocations), scenes_root=scenes_root, verbose=False
    )
    recs: List[Dict[str, Any]] = []
    for spec in SIGNS:
        pool_path = scenes_root / spec.sign / "moscow_pool.json"
        if not pool_path.is_file():
            print(f"[report] no pool for {spec.sign}")
            continue
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        for s in pool.get("scenes") or []:
            meta_path = scenes_root / spec.sign / s["scene_id"] / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            recs.append(
                {
                    "sign": spec.sign,
                    "pdd_code": spec.pdd_code,
                    "split": s.get("split"),
                    "scene_id": s["scene_id"],
                    "lane_count": int(meta.get("lane_count") or s.get("lane_count") or 0),
                    "rl_bucket": s.get("rl_bucket") or rl_bucket(int(meta.get("lane_count") or 0)),
                    "segment_type": meta.get("segment_type") or s.get("shape"),
                    "geo_cell": meta.get("geo_cell") or s.get("geo_cell"),
                    "length_m": float(meta.get("length_m") or 0.0),
                    "one_way_street": s.get("one_way_street"),
                    "osm_way_id": str(meta.get("osm_way_id") or ""),
                    "junction_id": str(meta.get("junction_id") or ""),
                    "latitude": meta.get("latitude"),
                    "longitude": meta.get("longitude"),
                    "crop_bbox": meta.get("crop_bbox"),
                    "level": s.get("selection_level"),
                }
            )
    # nearest neighbours
    for r in recs:
        best_f, best_o, box_min = None, None, None
        for q in recs:
            if q is r or r["latitude"] is None or q["latitude"] is None:
                continue
            d = haversine_m(r["latitude"], r["longitude"], q["latitude"], q["longitude"])
            best_f = d if best_f is None or d < best_f else best_f
            if r["crop_bbox"] and q["crop_bbox"]:
                a, b = r["crop_bbox"], q["crop_bbox"]
                gap = max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])
                box_min = gap if box_min is None or gap < box_min else box_min
        other = "train" if r["split"] == "test" else "test"
        for osplit, lat, lon, _box, _sign, _sid in official.points:
            if osplit != other or r["latitude"] is None:
                continue
            d = haversine_m(r["latitude"], r["longitude"], lat, lon)
            best_o = d if best_o is None or d < best_o else best_o
        r["nn_family_m"] = round(best_f, 1) if best_f is not None else None
        r["nn_official_other_split_m"] = round(best_o, 1) if best_o is not None else None
        r["min_box_gap_family_m"] = round(box_min, 1) if box_min is not None else None

    cols = ["sign", "pdd_code", "split", "scene_id", "lane_count", "rl_bucket", "segment_type", "geo_cell", "length_m",
            "one_way_street", "osm_way_id", "junction_id", "latitude", "longitude", "level",
            "nn_family_m", "nn_official_other_split_m", "min_box_gap_family_m"]
    with open(out_dir / "selection.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)

    lines = ["# Reserved-lane scene set v3", ""]
    lines.append(f"Maps: {len(recs)} across {len(set(r['sign'] for r in recs))} signs; "
                 f"unique ways {len(set(r['osm_way_id'] for r in recs))}, unique geo cells {len(set(r['geo_cell'] for r in recs))}.")
    ways = collections.Counter(r["osm_way_id"] for r in recs)
    dup_ways = [w for w, n in ways.items() if n > 1]
    tt_leak = set(r["osm_way_id"] for r in recs if r["split"] == "train") & set(r["osm_way_id"] for r in recs if r["split"] == "test")
    lines.append(f"Repeated ways inside the family: {len(dup_ways)}; train/test way overlap: {len(tt_leak)}.")
    lines.append("")
    lines.append("| sign | split | n | lanes 2/3/4+ | straight/curved | one-way | geo cells | median len m | nn family m (min/median) | nn official other split m (min) | min box gap m |")
    lines.append("|---|---|---:|---|---|---|---:|---:|---|---:|---:|")
    for spec in SIGNS:
        for split in ("train", "test"):
            rs = [r for r in recs if r["sign"] == spec.sign and r["split"] == split]
            if not rs:
                continue
            lb = collections.Counter(r["rl_bucket"] for r in rs)
            st = collections.Counter(r["segment_type"] for r in rs)
            ow = collections.Counter(r["one_way_street"] for r in rs)
            nn = sorted(r["nn_family_m"] for r in rs if r["nn_family_m"] is not None)
            no = sorted(r["nn_official_other_split_m"] for r in rs if r["nn_official_other_split_m"] is not None)
            bg = sorted(r["min_box_gap_family_m"] for r in rs if r["min_box_gap_family_m"] is not None)
            lens = sorted(r["length_m"] for r in rs)
            lines.append(
                f"| {spec.sign} | {split} | {len(rs)} | {lb['2']}/{lb['3']}/{lb['4plus']} | {st['straight']}/{st['curved']} | "
                f"{ow.get(True, 0)} | {len(set(r['geo_cell'] for r in rs))} | {lens[len(lens)//2]:.0f} | "
                f"{nn[0] if nn else '-'}/{nn[len(nn)//2] if nn else '-'} | {no[0] if no else '-'} | {bg[0] if bg else '-'} |"
            )
    levels = collections.Counter((r["sign"], r["level"]) for r in recs)
    lines.append("")
    lines.append("Selection levels: " + ", ".join(f"{s}:{l}={n}" for (s, l), n in sorted(levels.items(), key=str)))
    lines.append("")
    lines.append("Columns of `selection.csv`: nn_family_m = distance to the nearest other map of the family, "
                 "nn_official_other_split_m = distance to the nearest official map of the other split, "
                 "min_box_gap_family_m = smallest clearance between crop boxes inside the family.")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[report] wrote {_rel(out_dir / 'README.md')} and selection.csv")


# ---------------------------------------------------------------------- CLI

def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--index", type=Path, default=SEGMENTS_INDEX)
    ap.add_argument("--net", type=Path, default=MOSCOW_NET)
    ap.add_argument("--scenes-root", type=Path, default=DATA_SCENES)
    ap.add_argument("--crops-root", type=Path, default=SEGMENT_CROPS)
    ap.add_argument("--assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--allocations", type=Path, default=DEFAULT_ALLOCATIONS)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reserved-lane", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("select", help="pick the maps")
    _add_common(s)
    s.add_argument("--oneway-cache", type=Path, default=DEFAULT_ONEWAY_CACHE)
    s.add_argument("--out", type=Path, default=DEFAULT_ASSIGNMENT)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--split-seed", type=int, default=42)
    s.add_argument("--test-frac", type=float, default=0.2)
    s.add_argument("--per-sign", type=int, default=100)
    s.add_argument("--min-length-m", type=float, default=DEFAULT_MIN_LENGTH_M)
    s.add_argument("--reuse-policy", choices=("unique", "detour", "any"), default="unique")
    s.add_argument("--min-center-m", type=float, default=500.0)
    s.add_argument("--min-official-m", type=float, default=300.0)

    c = sp.add_parser("crop", help="netconvert the selected maps")
    _add_common(c)
    c.add_argument("--jobs", type=int, default=16)
    c.add_argument("--force", action="store_true")

    m = sp.add_parser("materialize", help="data/scenes/<sign>/ + moscow_pool.json")
    _add_common(m)
    m.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    m.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)

    r = sp.add_parser("report", help="histograms and distances of the materialized set")
    _add_common(r)
    r.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)

    a = sp.add_parser("all", help="select + crop + materialize + report")
    _add_common(a)
    a.add_argument("--oneway-cache", type=Path, default=DEFAULT_ONEWAY_CACHE)
    a.add_argument("--out", type=Path, default=DEFAULT_ASSIGNMENT)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--split-seed", type=int, default=42)
    a.add_argument("--test-frac", type=float, default=0.2)
    a.add_argument("--per-sign", type=int, default=100)
    a.add_argument("--min-length-m", type=float, default=DEFAULT_MIN_LENGTH_M)
    a.add_argument("--reuse-policy", choices=("unique", "detour", "any"), default="unique")
    a.add_argument("--min-center-m", type=float, default=500.0)
    a.add_argument("--min-official-m", type=float, default=300.0)
    a.add_argument("--jobs", type=int, default=16)
    a.add_argument("--force", action="store_true")
    a.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    a.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    a.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "select":
        run_select(args)
    elif args.cmd == "crop":
        run_crop(args)
    elif args.cmd == "materialize":
        run_materialize(args)
    elif args.cmd == "report":
        run_report(args)
    elif args.cmd == "all":
        run_select(args)
        args.assignment = args.out
        run_crop(args)
        run_materialize(args)
        args.out = args.report_dir
        run_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
