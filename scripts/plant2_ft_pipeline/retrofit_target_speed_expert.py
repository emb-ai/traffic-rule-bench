#!/usr/bin/env python3
"""Retrofit bogus constant target_speed=20 in non-speed-limit PlanT dumps.

For routes whose PDD sign is not a speed-limit catalog (no v_target_*), set:
  target_speed = min(max(speed, 0), 20)
  brake = (speed < 0.5)

Speed-limit signs (3.24 / 4.6 / 5.21 / 5.31) are left unchanged.

Also optionally purges matching keys from a PlanT diskcache (samples bake in
target_speed).

Examples:
  python retrofit_target_speed_expert.py \\
    --split /home/jovyan/.../plant2_l1_fv_experts_split_signs

  python retrofit_target_speed_expert.py \\
    --split .../plant2_l1_fv_experts_split_signs \\
    --purge-cache /tmp/plant2_ds_cache_spatial_aug
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from _paths import plan_t

PLAN_T = plan_t()
if str(PLAN_T) not in sys.path:
    sys.path.insert(0, str(PLAN_T))

from util.sign_id import (  # noqa: E402
    load_split_meta_route2sign,
    load_uid2sign,
    resolve_route_sign,
)

_MAX_TARGET_SPEED = 20.0
_BRAKE_SPEED_EPS = 0.5
SPEED_LIMIT_SIGNS = frozenset({"3.24", "4.6", "5.21", "5.31"})


def _sign_from_boxes(route_dir: Path) -> str | None:
    boxes_dir = route_dir / "boxes"
    if not boxes_dir.is_dir():
        return None
    files = sorted(boxes_dir.glob("*.json.gz"))
    if not files:
        return None
    mid = files[len(files) // 2]
    try:
        with gzip.open(mid, "rt", encoding="utf-8") as f:
            boxes = json.load(f)
    except Exception:
        return None
    if not isinstance(boxes, list):
        return None
    for b in boxes:
        if not isinstance(b, dict):
            continue
        if not b.get("affects_ego"):
            continue
        code = b.get("pdd_code") or b.get("sign_code") or b.get("class")
        if code:
            return str(code)
    return None


def resolve_sign(route_dir: Path, extra_map: dict[str, str], uid2sign: dict[str, str]) -> str | None:
    name = route_dir.name
    if name in extra_map:
        return extra_map[name]
    s = resolve_route_sign(name, uid2sign)
    if s:
        return s
    return _sign_from_boxes(route_dir)


def retrofit_file(path: Path) -> tuple[bool, str]:
    """Rewrite one measurements json.gz. Returns (changed, detail)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return False, f"read_error:{e}"

    speed = float(d.get("speed", 0.0) or 0.0)
    new_ts = min(max(speed, 0.0), _MAX_TARGET_SPEED)
    new_brake = speed < _BRAKE_SPEED_EPS
    old_ts = d.get("target_speed")
    old_brake = d.get("brake")
    has_ego = "ego_speed" in d and d["ego_speed"] == speed
    if old_ts == new_ts and bool(old_brake) == new_brake and has_ego:
        return False, "unchanged"

    d["target_speed"] = new_ts
    d["brake"] = new_brake
    d["ego_speed"] = speed  # always mirror expert speed (m/s)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False, f"write_error:{e}"
    return True, f"{old_ts}->{new_ts};brake {old_brake}->{new_brake}"


def _worker(route_dir_str: str) -> tuple[str, int, int, int]:
    """Process one route. Returns (name, n_files, n_changed, skipped_speed_limit)."""
    route_dir = Path(route_dir_str)
    # sign resolution done in parent; this worker only gets non-speed-limit routes
    meas = route_dir / "measurements"
    if not meas.is_dir():
        return route_dir.name, 0, 0, 0
    n_files = 0
    n_changed = 0
    for p in meas.glob("*.json.gz"):
        n_files += 1
        changed, _ = retrofit_file(p)
        if changed:
            n_changed += 1
    return route_dir.name, n_files, n_changed, 0


def collect_routes(split: Path) -> list[Path]:
    out: list[Path] = []
    for split_name in ("train", "val"):
        data = split / split_name / "data"
        if not data.is_dir():
            continue
        out.extend(sorted(p for p in data.iterdir() if p.is_dir()))
    return out


def _route_name_from_cache_key(ks: str) -> str | None:
    """.../data/<route>/boxes/0003.json.gz[_aug] → <route>."""
    if ks.endswith("_aug"):
        ks = ks[: -len("_aug")]
    try:
        p = Path(ks)
    except Exception:
        return None
    # .../<route>/boxes/<file>
    if p.parent.name == "boxes":
        return p.parent.parent.name
    return None


def purge_cache(cache_dir: Path, route_names: set[str], log_every: int = 50000) -> tuple[int, int]:
    from diskcache import Cache

    cache = Cache(str(cache_dir))
    n_seen = 0
    n_del = 0
    try:
        for key in cache.iterkeys():
            n_seen += 1
            if n_seen % log_every == 0:
                print(f"  cache scan {n_seen} keys, deleted {n_del}", flush=True)
            ks = key if isinstance(key, str) else str(key)
            route = _route_name_from_cache_key(ks)
            if route is None or route not in route_names:
                continue
            try:
                del cache[key]
                n_del += 1
            except KeyError:
                pass
    finally:
        cache.close()
    return n_seen, n_del


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        type=Path,
        default=Path(
            str(SHEPELEV / "plant2_l1_fv_experts_split_signs")
        ),
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--purge-cache",
        type=Path,
        default=None,
        help="diskcache dir to drop keys for retrofitted routes",
    )
    ap.add_argument(
        "--signs",
        default=None,
        help="Comma-separated sign codes to retrofit (default: all non-speed-limit)",
    )
    args = ap.parse_args()

    split: Path = args.split
    split_meta = split / "split_meta.json"
    extra = load_split_meta_route2sign(split_meta)
    uid2sign = load_uid2sign()
    only_signs = (
        {s.strip() for s in args.signs.split(",") if s.strip()} if args.signs else None
    )

    routes = collect_routes(split)
    print(f"split={split} routes={len(routes)} split_meta={split_meta.is_file()}")

    to_fix: list[Path] = []
    n_speed = 0
    n_unknown = 0
    sign_counts: dict[str, int] = {}
    for r in routes:
        sign = resolve_sign(r, extra, uid2sign)
        if sign is None:
            n_unknown += 1
            # Conservative: still fix if looks like constant-20 dump (checked later)
            # Prefer peeking boxes already done in resolve_sign; treat unknown as non-limit.
            sign = "unknown"
        sign_counts[sign] = sign_counts.get(sign, 0) + 1
        if sign in SPEED_LIMIT_SIGNS:
            n_speed += 1
            continue
        if only_signs is not None and sign not in only_signs and sign != "unknown":
            continue
        to_fix.append(r)

    print(f"speed-limit skipped={n_speed} unknown={n_unknown} to_fix={len(to_fix)}")
    print("sign_counts:", dict(sorted(sign_counts.items(), key=lambda x: -x[1])[:20]))

    if args.dry_run:
        # spot-check one route
        if to_fix:
            sample = sorted((to_fix[0] / "measurements").glob("*.json.gz"))[:3]
            for p in sample:
                with gzip.open(p, "rt") as f:
                    d = json.load(f)
                speed = float(d.get("speed", 0) or 0)
                print(
                    f"  dry {p.name}: speed={speed:.3f} "
                    f"target={d.get('target_speed')} brake={d.get('brake')} "
                    f"-> target={min(max(speed,0),20):.3f} brake={speed < 0.5}"
                )
        return 0

    t0 = time.time()
    n_files = 0
    n_changed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, str(r)) for r in to_fix]
        done = 0
        for fut in as_completed(futs):
            name, nf, nc, _ = fut.result()
            n_files += nf
            n_changed += nc
            done += 1
            if done % 200 == 0 or done == len(futs):
                print(
                    f"  routes {done}/{len(futs)} files={n_files} changed={n_changed} "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    print(
        f"DONE retrofit: routes={len(to_fix)} files={n_files} changed={n_changed} "
        f"in {time.time()-t0:.1f}s"
    )

    if args.purge_cache is not None:
        names = {r.name for r in to_fix}
        print(f"Purging cache {args.purge_cache} for {len(names)} routes...")
        t1 = time.time()
        n_seen, n_del = purge_cache(args.purge_cache, names)
        print(f"DONE purge: scanned={n_seen} deleted={n_del} in {time.time()-t1:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
