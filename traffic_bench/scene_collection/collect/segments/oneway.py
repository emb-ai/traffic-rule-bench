"""True one-way classification of SUMO edges for the reserved-lane family.

An edge is a *true one-way street* when no other edge runs anti-parallel to it
close by: every lane shape of the net is sampled into a coarse grid once, then
each candidate edge is walked every 10 m and an opposed heading (> 150 deg)
within 16 m counts as a hit; three hits disprove one-way. Results are cached
per edge id in a JSON file keyed by the scan parameters and the net file.

5.11.x (counter-flow bus / bicycle lane) needs ``True``; 5.14.x prefers
``False`` (a street with an opposite carriageway nearby).
"""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

DEFAULT_SAMPLE_M = 8.0
DEFAULT_WALK_M = 10.0
DEFAULT_GRID_M = 40.0
DEFAULT_RADIUS_M = 16.0
DEFAULT_OPPOSED_DEG = 150.0
DEFAULT_DISPROVE_HITS = 3

_EDGE_RE = re.compile(r'<edge id="([^"]+)"([^>]*)>(.*?)</edge>', re.S)
_SHAPE_RE = re.compile(r'shape="([^"]+)"')
_LANE_SHAPE_RE = re.compile(r'<lane [^>]*shape="([^"]+)"')


def load_edge_shapes(net_path: Path | str) -> Dict[str, List[Point]]:
    """Edge id -> polyline (edge shape, else the first lane shape); internal edges skipped."""
    text = Path(net_path).read_text(encoding="utf-8")
    shapes: Dict[str, List[Point]] = {}
    for m in _EDGE_RE.finditer(text):
        eid = m.group(1)
        if eid.startswith(":"):
            continue
        sh = _SHAPE_RE.search(m.group(2)) or _LANE_SHAPE_RE.search(m.group(3))
        if not sh:
            continue
        pts: List[Point] = []
        for tok in sh.group(1).split():
            parts = tok.split(",")
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        if len(pts) >= 2:
            shapes[eid] = pts
    return shapes


def _samples(pts: Sequence[Point], step: float) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 0.0:
            continue
        n = max(1, int(seg // step))
        h = math.atan2(y1 - y0, x1 - x0)
        for k in range(n):
            t = k / n
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), h))
    return out


class OneWayIndex:
    """Grid of heading samples over the whole net; answers ``true_oneway(edge)``."""

    def __init__(
        self,
        shapes: Dict[str, List[Point]],
        *,
        sample_m: float = DEFAULT_SAMPLE_M,
        grid_m: float = DEFAULT_GRID_M,
    ) -> None:
        self.shapes = shapes
        self.grid_m = float(grid_m)
        self.grid: Dict[Tuple[int, int], List[Tuple[float, float, float, str]]] = collections.defaultdict(list)
        for eid, pts in shapes.items():
            for x, y, h in _samples(pts, sample_m):
                self.grid[(int(x // self.grid_m), int(y // self.grid_m))].append((x, y, h, eid))

    def true_oneway(
        self,
        edge_id: str,
        *,
        walk_m: float = DEFAULT_WALK_M,
        radius_m: float = DEFAULT_RADIUS_M,
        opposed_deg: float = DEFAULT_OPPOSED_DEG,
        disprove_hits: int = DEFAULT_DISPROVE_HITS,
    ) -> Optional[bool]:
        pts = self.shapes.get(edge_id)
        if not pts:
            return None
        r2 = float(radius_m) ** 2
        opp = math.radians(opposed_deg)
        hits = 0
        for x, y, h in _samples(pts, walk_m):
            cx, cy = int(x // self.grid_m), int(y // self.grid_m)
            found = False
            for dx in (-1, 0, 1):
                if found:
                    break
                for dy in (-1, 0, 1):
                    if found:
                        break
                    for ex, ey, eh, oid in self.grid.get((cx + dx, cy + dy), ()):
                        if oid == edge_id:
                            continue
                        if (x - ex) ** 2 + (y - ey) ** 2 < r2:
                            d = abs((h - eh + math.pi) % (2 * math.pi) - math.pi)
                            if d > opp:
                                found = True
                                break
            if found:
                hits += 1
                if hits >= int(disprove_hits):
                    return False
        return True


def _cache_key(net_path: Path, **params) -> str:
    st = net_path.stat()
    return json.dumps({"net": net_path.name, "size": st.st_size, "mtime": int(st.st_mtime), **params}, sort_keys=True)


def scan_true_oneway(
    net_path: Path | str,
    edge_ids: Iterable[str],
    *,
    cache_path: Optional[Path | str] = None,
    sample_m: float = DEFAULT_SAMPLE_M,
    walk_m: float = DEFAULT_WALK_M,
    grid_m: float = DEFAULT_GRID_M,
    radius_m: float = DEFAULT_RADIUS_M,
    opposed_deg: float = DEFAULT_OPPOSED_DEG,
    disprove_hits: int = DEFAULT_DISPROVE_HITS,
    verbose: bool = True,
) -> Dict[str, Optional[bool]]:
    """One-way flag per edge id (True / False / None = no shape), cached."""
    net_path = Path(net_path)
    wanted = [str(e) for e in edge_ids]
    if not net_path.is_file():
        # No net on this machine (e.g. a laptop clone): serve whatever the cache
        # holds, unknown edges stay None.
        cached: Dict = {}
        if cache_path is not None and Path(cache_path).is_file():
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8")).get("flags") or {}
        if verbose:
            print(f"[oneway] net {net_path} missing; using cache only ({len(cached)} flags)", flush=True)
        return {e: cached.get(e) for e in wanted}
    key = _cache_key(
        net_path, sample_m=sample_m, walk_m=walk_m, grid_m=grid_m,
        radius_m=radius_m, opposed_deg=opposed_deg, disprove_hits=disprove_hits,
    )
    flags: Dict[str, Optional[bool]] = {}
    cache: Dict = {}
    if cache_path is not None and Path(cache_path).is_file():
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
        if cache.get("key") == key:
            flags = dict(cache.get("flags") or {})
    missing = [e for e in wanted if e not in flags]
    if missing:
        if verbose:
            print(f"[oneway] scanning {len(missing)} edge(s) (cached {len(wanted) - len(missing)}) ...", flush=True)
        shapes = load_edge_shapes(net_path)
        if verbose:
            print(f"[oneway] {len(shapes)} edges with shape", flush=True)
        index = OneWayIndex(shapes, sample_m=sample_m, grid_m=grid_m)
        for i, e in enumerate(missing, 1):
            flags[e] = index.true_oneway(
                e, walk_m=walk_m, radius_m=radius_m, opposed_deg=opposed_deg, disprove_hits=disprove_hits
            )
            if verbose and i % 1000 == 0:
                print(f"[oneway] {i}/{len(missing)}", flush=True)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps({"key": key, "flags": flags}, indent=0), encoding="utf-8")
    return {e: flags.get(e) for e in wanted}
