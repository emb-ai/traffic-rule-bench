#!/usr/bin/env python3
"""In-place target_speed patch for non-speed-limit routes in a PlanT diskcache.

Efficient path: enumerate keys from SPLIT route lists (train+val) — never Cache.iterkeys().
Skips SPEED_LIMIT_SIGNS = {3.24, 4.6, 5.21, 5.31}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

from diskcache import Cache

from _paths import shepelev

SHEPELEV = shepelev()
PLAN_T = SHEPELEV / "traffic-rule-bench/plant2/PlanT"
SPLIT_FULL = SHEPELEV / "plant2_l1_fv_experts_split_signs"
CACHE_DEFAULT = Path("/tmp/plant2_ds_cache_spatial_aug")

SEQ_LEN = 1
WPS_LEN = 8
_MAX_TS = 20.0
SPEED_LIMIT_SIGNS = frozenset({"3.24", "4.6", "5.21", "5.31"})

sys.path.insert(0, str(PLAN_T))
from util.sign_id import (  # noqa: E402
    load_split_meta_route2sign,
    load_uid2sign,
    resolve_route_sign,
)


def target_from_meas(boxes_key: str) -> float | None:
    ks = boxes_key[: -len("_aug")] if boxes_key.endswith("_aug") else boxes_key
    p = Path(ks)
    if p.parent.name != "boxes":
        return None
    meas = p.parent.parent / "measurements" / p.name
    if not meas.is_file():
        meas = p.resolve().parent.parent / "measurements" / p.name
    if not meas.is_file():
        return None
    with gzip.open(meas, "rt", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("brake"):
        return 0.0
    ts = float(d.get("target_speed", d.get("ego_speed", d.get("speed", 0.0))) or 0.0)
    return min(max(ts, 0.0), _MAX_TS)


def patch_sample(sample: dict, new_ts: float) -> dict:
    sample = dict(sample)
    sample["target_speed"] = new_ts
    if "ego_speed" not in sample and "speed" in sample:
        sample["ego_speed"] = sample["speed"]
    return sample


def resolve_sign(route_name: str, extra: dict[str, str], uid2sign: dict[str, str]) -> str | None:
    if route_name in extra:
        return extra[route_name]
    return resolve_route_sign(route_name, uid2sign)


def enumerate_nonspeed_keys(split_root: Path) -> tuple[list[str], int, int]:
    """Return unaugmented keys for non-speed-limit routes; counts skipped/included routes."""
    extra = load_split_meta_route2sign(split_root / "split_meta.json")
    uid2sign = load_uid2sign()
    keys: list[str] = []
    n_skip = n_ok = 0
    for split_name in ("train", "val"):
        data = split_root / split_name / "data"
        if not data.is_dir():
            continue
        for route_dir in sorted(p for p in data.iterdir() if p.is_dir()):
            sign = resolve_sign(route_dir.name, extra, uid2sign)
            if sign in SPEED_LIMIT_SIGNS:
                n_skip += 1
                continue
            n_ok += 1
            boxes = route_dir / "boxes"
            if not boxes.is_dir():
                continue
            n_boxes = len(os.listdir(boxes))
            end = n_boxes - WPS_LEN - SEQ_LEN - 2
            for seq in range(5, end):
                keys.append(str(route_dir / "boxes" / f"{seq:04d}.json.gz"))
    return keys, n_ok, n_skip


def probe_sign(
    cache: Cache,
    split_root: Path,
    sign_want: str,
    n_probe: int = 20,
) -> tuple[int, int, int]:
    """Return (probed, neq20, low_lt1) for routes of a given sign."""
    extra = load_split_meta_route2sign(split_root / "split_meta.json")
    uid2sign = load_uid2sign()
    probed = neq20 = low = 0
    for split_name in ("train", "val"):
        data = split_root / split_name / "data"
        if not data.is_dir():
            continue
        for route_dir in sorted(p for p in data.iterdir() if p.is_dir()):
            if resolve_sign(route_dir.name, extra, uid2sign) != sign_want:
                continue
            boxes = route_dir / "boxes"
            if not boxes.is_dir():
                continue
            n_boxes = len(os.listdir(boxes))
            end = n_boxes - WPS_LEN - SEQ_LEN - 2
            if end <= 5:
                continue
            mid = (5 + end) // 2
            k = str(route_dir / "boxes" / f"{mid:04d}.json.gz")
            if k not in cache:
                continue
            t = float(cache[k]["target_speed"])
            probed += 1
            if abs(t - 20.0) > 1e-6:
                neq20 += 1
            if t < 1.0:
                low += 1
            if probed >= n_probe:
                return probed, neq20, low
    return probed, neq20, low


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    ap.add_argument("--split", type=Path, default=SPLIT_FULL)
    ap.add_argument("--cache-size-gb", type=float, default=1800.0)
    ap.add_argument("--log-every", type=int, default=20000)
    ap.add_argument("--exclude-speed-limit", action="store_true", default=True)
    args = ap.parse_args()

    print(f"Enumerating non-speed-limit keys under {args.split} …", flush=True)
    t0 = time.time()
    base_keys, n_routes, n_skip_sl = enumerate_nonspeed_keys(args.split)
    print(
        f"routes_ok={n_routes} skipped_speed_limit_routes={n_skip_sl} "
        f"base_keys={len(base_keys)} enumerate={time.time()-t0:.1f}s",
        flush=True,
    )

    cache = Cache(str(args.cache), size_limit=int(args.cache_size_gb * 1024**3))
    print("PRE_PROBE 2.5:", probe_sign(cache, args.split, "2.5"), flush=True)
    for sl in sorted(SPEED_LIMIT_SIGNS):
        print(f"PRE_PROBE {sl}:", probe_sign(cache, args.split, sl), flush=True)

    n_changed = n_skip = n_miss = n_err = 0
    t0 = time.time()
    try:
        for i, key in enumerate(base_keys):
            new_ts = target_from_meas(key)
            if new_ts is None:
                n_err += 1
                if n_err <= 10:
                    print(f"ERR no_meas {key}", flush=True)
                continue
            for suffix in ("", "_aug"):
                k = key + suffix
                if k not in cache:
                    n_miss += 1
                    continue
                try:
                    sample = cache[k]
                    old = float(sample.get("target_speed", -999))
                    if abs(old - new_ts) < 1e-6:
                        n_skip += 1
                        continue
                    cache[k] = patch_sample(sample, new_ts)
                    n_changed += 1
                except Exception as e:
                    n_err += 1
                    if n_err <= 10:
                        print(f"ERR {k}: {type(e).__name__}: {e}", flush=True)
            if (i + 1) % args.log_every == 0:
                print(
                    f"  {i+1}/{len(base_keys)} changed={n_changed} skip={n_skip} "
                    f"miss={n_miss} err={n_err} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
    finally:
        print("POST_PROBE 2.5:", probe_sign(cache, args.split, "2.5"), flush=True)
        for sl in sorted(SPEED_LIMIT_SIGNS):
            print(f"POST_PROBE {sl}:", probe_sign(cache, args.split, sl), flush=True)
        cache.close()

    print(
        f"PATCH_DONE changed={n_changed} skip={n_skip} miss={n_miss} err={n_err} "
        f"skipped_speed_limit_routes={n_skip_sl} elapsed={time.time()-t0:.0f}s",
        flush=True,
    )
    return 0 if n_err == 0 or n_changed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
