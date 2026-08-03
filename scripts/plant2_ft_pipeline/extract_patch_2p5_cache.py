#!/usr/bin/env python3
"""Build a small 2.5-only diskcache with corrected target_speed.

Fast path (no iterkeys over the 1.6TB spatial_aug cache):
  1) Enumerate sample keys from the 2.5 subset split (train+val).
  2) For each key (+ `_aug` sibling): copy from the big cache if present,
     set target_speed from measurements (brake→0), write into the small cache.
  3) Optionally materialize missing keys via PlanTDataset.__getitem__.

Sample index formula (seq_len=1, wps_len=8):
  seq in range(5, n_boxes - wps_len - seq_len - 2)  →  n_boxes - 16 starts.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

from diskcache import Cache

from _paths import shepelev

SHEPELEV = shepelev()
PLAN_T = SHEPELEV / "traffic-rule-bench/plant2/PlanT"
SPLIT_2P5 = SHEPELEV / "plant2_l1_fv_experts_split_signs_2.5"
SPLIT_FULL = SHEPELEV / "plant2_l1_fv_experts_split_signs"
SRC_DEFAULT = Path("/tmp/plant2_ds_cache_spatial_aug")
DST_DEFAULT = Path("/tmp/plant2_ds_cache_2p5_tsfix")

SEQ_LEN = 1
WPS_LEN = 8
_MAX_TS = 20.0


def target_from_meas(boxes_key: str) -> float | None:
    """boxes/.../NNNN.json.gz[_aug] → measurements target (brake→0)."""
    ks = boxes_key[: -len("_aug")] if boxes_key.endswith("_aug") else boxes_key
    p = Path(ks)
    if p.parent.name != "boxes":
        return None
    meas = p.parent.parent / "measurements" / p.name
    if not meas.is_file():
        # follow symlink route → full split measurements
        meas = p.resolve().parent.parent / "measurements" / p.name
    if not meas.is_file():
        return None
    with gzip.open(meas, "rt", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("brake"):
        return 0.0
    ts = float(d.get("target_speed", d.get("ego_speed", d.get("speed", 0.0))) or 0.0)
    return min(max(ts, 0.0), _MAX_TS)


def enumerate_sample_keys(split_root: Path) -> list[str]:
    """Return unaugmented cache keys (…/boxes/NNNN.json.gz) for train+val."""
    keys: list[str] = []
    for split_name in ("train", "val"):
        data = split_root / split_name / "data"
        if not data.is_dir():
            continue
        for route_dir in sorted(p for p in data.iterdir() if p.is_dir()):
            boxes = route_dir / "boxes"
            if not boxes.is_dir():
                continue
            n_boxes = len(os.listdir(boxes))
            end = n_boxes - WPS_LEN - SEQ_LEN - 2
            for seq in range(5, end):
                keys.append(str(route_dir / "boxes" / f"{seq:04d}.json.gz"))
    return keys


def to_full_key(key_2p5: str) -> str:
    return key_2p5.replace(str(SPLIT_2P5), str(SPLIT_FULL), 1)


def patch_sample(sample: dict, new_ts: float) -> dict:
    sample = dict(sample)
    sample["target_speed"] = new_ts
    if "ego_speed" not in sample and "speed" in sample:
        sample["ego_speed"] = sample["speed"]
    return sample


def materialize_missing(
    missing_base_keys: list[str],
    dst: Cache,
    log_every: int = 200,
) -> tuple[int, int]:
    """Materialize unaugmented (+aug if training.augment) samples into dst."""
    if not missing_base_keys:
        return 0, 0
    sys.path.insert(0, str(PLAN_T))
    os.chdir(PLAN_T)
    from omegaconf import OmegaConf, open_dict
    from dataset import PlanTDataset

    cfg = OmegaConf.load("config/config.yaml")
    cfg = OmegaConf.merge(
        cfg,
        {
            "user": OmegaConf.load("config/user/arbelyaev.yaml"),
            "model": OmegaConf.load("config/model/PlanT.yaml"),
        },
    )
    with open_dict(cfg):
        cfg.use_caching = True
        cfg.model.training.augment = True
        cfg.model.training.augment_parked = False
        cfg.model.training.filter_routes = False

    # Index by first-label path for both splits.
    path_to_index: dict[str, tuple[PlanTDataset, int]] = {}
    for split_name in ("train", "val"):
        root = str(SPLIT_2P5 / split_name / "data")
        ds = PlanTDataset(root, cfg, shared_dict=dst)
        for i in range(len(ds)):
            k = ds.labels[i][0].decode()
            path_to_index[k] = (ds, i)

    n_ok = n_err = 0
    t0 = time.time()
    for j, key in enumerate(missing_base_keys):
        try:
            ds, idx = path_to_index[key]
            # Force rebuild (bypass stale cache entry)
            if key in dst:
                del dst[key]
            aug_k = key + "_aug"
            if aug_k in dst:
                del dst[aug_k]
            _ = ds[idx]  # writes into dst via shared_dict
            # Re-patch target_speed from measurements (dataset already does brake→0,
            # but be explicit in case of any path quirks).
            for k in (key, aug_k):
                if k not in dst:
                    continue
                new_ts = target_from_meas(k)
                if new_ts is None:
                    continue
                dst[k] = patch_sample(dst[k], new_ts)
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 20:
                print(f"MATERIALIZE_ERR {key}: {type(e).__name__}: {e}", flush=True)
        if (j + 1) % log_every == 0:
            print(
                f"  materialize {j+1}/{len(missing_base_keys)} ok={n_ok} err={n_err} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return n_ok, n_err


def verify_dst(dst: Cache, keys: list[str], n_probe: int = 40) -> int:
    """Return number of near-stop targets among probes; raise if all still 20."""
    import random

    rng = random.Random(0)
    probe = keys if len(keys) <= n_probe else rng.sample(keys, n_probe)
    n_neq20 = n_low = n_miss = 0
    examples = []
    for k in probe:
        if k not in dst:
            n_miss += 1
            continue
        t = float(dst[k]["target_speed"])
        if abs(t - 20.0) > 1e-6:
            n_neq20 += 1
        if t < 1.0:
            n_low += 1
        if len(examples) < 8:
            examples.append((Path(k).parent.parent.name[:40], t))
    print(
        f"VERIFY probe={len(probe)} miss={n_miss} neq20={n_neq20} low_lt1={n_low} "
        f"examples={examples}",
        flush=True,
    )
    if n_neq20 == 0:
        raise SystemExit("VERIFY_FAIL: all probed target_speed still 20")
    if n_low == 0:
        raise SystemExit("VERIFY_FAIL: no near-stop target_speed in probes")
    return n_low


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT)
    ap.add_argument("--split", type=Path, default=SPLIT_2P5)
    ap.add_argument("--cache-size-gb", type=float, default=400.0)
    ap.add_argument("--reset-dst", action="store_true", help="rm -rf dst before write")
    ap.add_argument(
        "--materialize-missing",
        action="store_true",
        help="PlanTDataset __getitem__ for keys missing in src",
    )
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    if args.reset_dst and args.dst.exists():
        print(f"RESET {args.dst}", flush=True)
        # Prefer shell rm -rf: more reliable on busy diskcache shard dirs.
        import subprocess

        subprocess.run(["rm", "-rf", str(args.dst)], check=False)
        if args.dst.exists():
            shutil.rmtree(args.dst, ignore_errors=True)
        if args.dst.exists():
            raise SystemExit(f"FAILED to reset dst={args.dst}")
    args.dst.mkdir(parents=True, exist_ok=True)

    print(f"Enumerating keys under {args.split} …", flush=True)
    t0 = time.time()
    base_keys = enumerate_sample_keys(args.split)
    print(
        f"base_keys={len(base_keys)} (≈{2*len(base_keys)} with _aug) "
        f"enumerate={time.time()-t0:.1f}s",
        flush=True,
    )

    src = Cache(str(args.src))
    dst = Cache(str(args.dst), size_limit=int(args.cache_size_gb * 1024**3))

    n_copy = n_skip = n_miss = n_err = n_aug = 0
    missing_base: list[str] = []
    t0 = time.time()
    try:
        for i, key in enumerate(base_keys):
            new_ts = target_from_meas(key)
            if new_ts is None:
                n_err += 1
                if n_err <= 10:
                    print(f"ERR no_meas {key}", flush=True)
                continue

            got_any = False
            for suffix in ("", "_aug"):
                k = key + suffix
                sample = None
                if k in src:
                    sample = src[k]
                else:
                    k_full = to_full_key(k)
                    if k_full in src:
                        sample = src[k_full]
                if sample is None:
                    continue
                got_any = True
                try:
                    old = sample.get("target_speed")
                    if (
                        k in dst
                        and old is not None
                        and abs(float(dst[k].get("target_speed", -999)) - new_ts) < 1e-6
                        and abs(float(old) - new_ts) < 1e-6
                    ):
                        n_skip += 1
                        continue
                    dst[k] = patch_sample(sample, new_ts)
                    n_copy += 1
                    if suffix:
                        n_aug += 1
                except Exception as e:
                    n_err += 1
                    if n_err <= 10:
                        print(f"ERR {k}: {type(e).__name__}: {e}", flush=True)

            if not got_any:
                n_miss += 1
                missing_base.append(key)

            if (i + 1) % args.log_every == 0:
                print(
                    f"  {i+1}/{len(base_keys)} copy={n_copy} aug={n_aug} skip={n_skip} "
                    f"miss={n_miss} err={n_err} elapsed={time.time()-t0:.0f}s "
                    f"dst_len={len(dst)}",
                    flush=True,
                )
    finally:
        src.close()

    print(
        f"COPY_DONE copy={n_copy} aug_included≈{n_aug} skip={n_skip} miss={n_miss} "
        f"err={n_err} dst_len={len(dst)} elapsed={time.time()-t0:.0f}s",
        flush=True,
    )

    if missing_base and args.materialize_missing:
        print(f"MATERIALIZE {len(missing_base)} missing base keys …", flush=True)
        ok, err = materialize_missing(missing_base, dst)
        print(f"MATERIALIZE_DONE ok={ok} err={err} dst_len={len(dst)}", flush=True)
    elif missing_base:
        print(
            f"WARN {len(missing_base)} missing (use --materialize-missing to fill)",
            flush=True,
        )

    if not args.skip_verify:
        verify_dst(dst, base_keys)

    dst.close()
    # size on disk
    try:
        import subprocess

        du = subprocess.check_output(["du", "-sh", str(args.dst)], text=True).split()[0]
    except Exception:
        du = "?"
    print(f"DONE dst={args.dst} size={du}", flush=True)
    return 0 if n_copy > 0 or n_skip > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
