#!/usr/bin/env python3
"""PlanT2 diskcache prefill.

Subcommands:
  shard      Fill one index range (single worker)
  parallel   Shard train + val across workers (calls shard)
  2p5        Extract+patch 2.5 keys from full cache (fast path for 2.5-only FT)

Examples::

  # Full spatial split → /tmp/plant2_ds_cache_spatial_aug (~1.7T with augment)
  python prefill_diskcache.py parallel \\
    --ds $SHEPELEV/plant2_l1_fv_experts_split_signs/train \\
    --ds-val $SHEPELEV/plant2_l1_fv_experts_split_signs/val \\
    --ds-local /tmp/plant2_ds_cache_spatial_aug \\
    --cache-size-gb 1800 \\
    --max-workers 32

  # Dry-run (show shard plan only)
  python prefill_diskcache.py parallel --dry-run \\
    --ds .../train --ds-val .../val --ds-local /tmp/plant2_ds_cache_spatial_aug

  # Single shard (parallel invokes this internally)
  python prefill_diskcache.py shard \\
    --ds .../train --ds-local /tmp/plant2_ds_cache_spatial_aug \\
    --split train --start 0 --end 50000 --cache-size-gb 1800

  # 2.5-only cache after retrofit measurements (fast extract from full cache)
  python prefill_diskcache.py 2p5 \\
    --src /tmp/plant2_ds_cache_spatial_aug \\
    --dst /tmp/plant2_ds_cache_2p5_tsfix \\
    --split $SHEPELEV/plant2_l1_fv_experts_split_signs_2.5 \\
    --cache-size-gb 400 \\
    --reset-dst --materialize-missing
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

from lib.env import pipeline_dir, resolve_python, shepelev
from lib.utils import count_plant2_samples, default_prefill_max_workers, iso_now

# --- shard core (from prefill_plant2_diskcache.py) ---


def _plan_t():
    from lib.env import plan_t

    pt = plan_t()
    os.chdir(pt)
    if str(pt) not in sys.path:
        sys.path.insert(0, str(pt))
    return pt


def _log(msg: str, log_path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def _cache_key(dataset, index: int) -> str:
    """The key the dataset itself would use.

    Under WPS_STRIDE>1 the dataset suffixes the key, since the stride changes
    the stored waypoints. Recomputing the bare label here would look up an
    entry that never exists: the base would count as missing right after being
    filled, and the augmented variant would be skipped for every sample.
    """
    labels = dataset.labels[index]
    key_fn = getattr(dataset, "_cache_key", None)
    return key_fn(labels) if key_fn is not None else labels[0].decode()


def _fill_one(dataset, cache, index: int, do_aug: bool) -> tuple[str, str]:
    key = _cache_key(dataset, index)
    old_tf = dataset.transform
    dataset.transform = None
    try:
        if key not in cache:
            _ = dataset[index]
            base_st = "filled"
        else:
            base_st = "skipped"
    except Exception as e:  # noqa: BLE001
        dataset.transform = old_tf
        return f"error:{type(e).__name__}:{e}", "skipped"

    aug_st = "skipped"
    if do_aug and old_tf is not None:
        aug_key = key + "_aug"
        try:
            if aug_key not in cache:
                if key not in cache:
                    raise RuntimeError("base missing after fill")
                sample = old_tf(copy.deepcopy(cache[key]))
                sample.pop("BEV_aug", None)
                sample.pop("output_floating", None)
                cache[aug_key] = sample
                aug_st = "filled"
        except Exception as e:  # noqa: BLE001
            aug_st = f"error:{type(e).__name__}:{e}"
    dataset.transform = old_tf
    return base_st, aug_st


def _prefill_split(
    *,
    ds_root: str,
    cache,
    cfg,
    log_path: Path,
    do_aug: bool,
    start: int,
    end: int | None,
    log_every: int,
    stop_bytes: int,
    tag: str,
) -> dict:
    from dataset import PlanTDataset

    dataset = PlanTDataset(ds_root.rstrip("/") + "/data", cfg, shared_dict=cache)
    if do_aug and cfg.model.training.augment:
        dataset.transform = dataset.aug_sample
    else:
        dataset.transform = None
        do_aug = False

    n = len(dataset)
    if end is None:
        end = n
    end = min(max(end, start), n)
    start = max(0, min(start, n))
    _log(f"{tag}: dataset_len={n} range=[{start},{end}) do_aug={do_aug}", log_path)

    t0 = time.time()
    filled_base = filled_aug = skipped = errors = 0
    stopped_early = False
    i = start - 1
    for i in range(start, end):
        if cache.volume() >= stop_bytes:
            stopped_early = True
            _log(f"{tag}: STOP_NEAR_LIMIT i={i}/{n} volume_gb={cache.volume()/1024**3:.2f}", log_path)
            break
        base_st, aug_st = _fill_one(dataset, cache, i, do_aug)
        for st in (base_st, aug_st):
            if st == "skipped":
                skipped += 1
            elif st.startswith("error:"):
                errors += 1
        if base_st == "filled":
            filled_base += 1
        if aug_st == "filled":
            filled_aug += 1
        done = i - start + 1
        if done % log_every == 0 or i == start:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            _log(
                f"{tag}: progress i={i+1}/{n} done={done} filled_base={filled_base} "
                f"filled_aug={filled_aug} errors={errors} rate={rate:.2f}/s",
                log_path,
            )

    return {
        "tag": tag,
        "stopped_early": stopped_early,
        "filled_base": filled_base,
        "filled_aug": filled_aug,
        "skipped": skipped,
        "errors": errors,
        "elapsed_s": time.time() - t0,
        "volume_gb": cache.volume() / 1024**3,
        "len": len(cache),
    }


def cmd_shard(args: argparse.Namespace) -> int:
    from diskcache import Cache
    from omegaconf import OmegaConf, open_dict

    pt = _plan_t()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ds_train = str(args.ds).rstrip("/")
    ds_val = str(args.ds_val).rstrip("/") if args.ds_val else ""
    args.ds_local.mkdir(parents=True, exist_ok=True)
    size_limit = int(args.cache_size_gb * 1024**3)

    cfg = OmegaConf.load(pt / "config" / "config.yaml")
    cfg = OmegaConf.merge(
        cfg,
        {"user": OmegaConf.load(pt / "config" / "user" / "arbelyaev.yaml"),
         "model": OmegaConf.load(pt / "config" / "model" / "PlanT.yaml")},
    )
    with open_dict(cfg):
        cfg.use_caching = True
        cfg.model.training.augment = bool(args.augment)
        cfg.model.training.augment_parked = False
        cfg.model.training.filter_routes = False

    cache = Cache(directory=str(args.ds_local), size_limit=size_limit)
    stop_bytes = int(args.stop_frac * size_limit)
    results = []

    if args.split in ("train", "both"):
        results.append(
            _prefill_split(
                ds_root=ds_train, cache=cache, cfg=cfg, log_path=log_path,
                do_aug=args.augment, start=args.start, end=args.end,
                log_every=args.log_every, stop_bytes=stop_bytes, tag="train",
            )
        )
    if args.split in ("val", "both"):
        if not ds_val:
            cand = Path(ds_train).parent / "val"
            if (cand / "data").is_dir():
                ds_val = str(cand)
        if ds_val and (Path(ds_val) / "data").is_dir():
            v_start = args.start if args.split == "val" else 0
            v_end = args.end if args.split == "val" else None
            results.append(
                _prefill_split(
                    ds_root=ds_val, cache=cache, cfg=cfg, log_path=log_path,
                    do_aug=args.augment, start=v_start, end=v_end,
                    log_every=args.log_every, stop_bytes=stop_bytes, tag="val",
                )
            )
    for r in results:
        _log(f"PREFILL_DONE_{r['tag']} filled_base={r['filled_base']} errors={r['errors']}", log_path)
    cache.close()
    return 0


def cmd_parallel(args: argparse.Namespace) -> int:
    py = resolve_python(args.python_exe)
    shard_py = pipeline_dir() / "data" / "prefill_diskcache.py"
    nproc = os.cpu_count() or 8
    max_workers = args.max_workers or default_prefill_max_workers(nproc)
    args.ds_local.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    train_n = count_plant2_samples(args.ds)
    val_n = count_plant2_samples(args.ds_val) if (args.ds_val / "data").is_dir() else 0
    print(f"TRAIN_N={train_n} VAL_N={val_n} max_workers={max_workers}")

    tasks: list[tuple[str, int, int]] = []
    n_shards = max(1, min(max_workers, train_n if train_n > 0 else 1))
    for s in range(n_shards):
        st = s * train_n // n_shards
        en = (s + 1) * train_n // n_shards
        if en > st:
            tasks.append(("train", st, en))
    if val_n > 0:
        tasks.append(("val", 0, val_n))

    if args.dry_run:
        for t in tasks:
            print(t)
        return 0

    procs: list[subprocess.Popen] = []
    fail = 0

    def wait_slot() -> None:
        while sum(1 for p in procs if p.poll() is None) >= max_workers:
            time.sleep(2)
        procs[:] = [p for p in procs if p.poll() is None]

    for idx, (split, start, end) in enumerate(tasks):
        wait_slot()
        logf = args.log_dir / f"w{idx}_{split}_{start}_{end}_{ts}.log"
        cmd = [
            str(py), str(shard_py), "shard",
            "--ds", str(args.ds),
            "--ds-val", str(args.ds_val),
            "--ds-local", str(args.ds_local),
            "--cache-size-gb", str(args.cache_size_gb),
            "--split", split,
            "--start", str(start),
            "--end", str(end),
            "--log", str(logf),
        ]
        if args.augment:
            cmd.append("--augment")
        else:
            cmd.append("--no-augment")
        outer = args.log_dir / f"w{idx}.outer"
        with outer.open("w") as out:
            procs.append(subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT))
        print(f"[spawn] #{idx} {split} [{start},{end})")
        if args.spawn_stagger_sec > 0:
            time.sleep(args.spawn_stagger_sec)

    for p in procs:
        p.wait()
        if p.returncode != 0:
            fail += 1
    print(f"=== parallel done fail={fail} ===")
    return fail


def cmd_2p5(args: argparse.Namespace) -> int:
    """Delegate to extract_patch_2p5_cache."""
    from data import extract_patch_2p5_cache as ep

    argv = [
        "--src", str(args.src),
        "--dst", str(args.dst),
        "--split", str(args.split),
        "--cache-size-gb", str(args.cache_size_gb),
    ]
    if args.reset_dst:
        argv.append("--reset-dst")
    if args.materialize_missing:
        argv.append("--materialize-missing")
    if args.skip_verify:
        argv.append("--skip-verify")
    sys.argv = ["extract_patch_2p5_cache.py", *argv]
    return ep.main()


def main(argv: list[str] | None = None) -> int:
    sh = shepelev()
    default_split = sh / "plant2_l1_fv_experts_split_signs"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_shard = sub.add_parser("shard", help="Single worker / index range")
    p_shard.add_argument("--ds", type=Path, required=True)
    p_shard.add_argument("--ds-val", type=Path, default=None)
    p_shard.add_argument("--ds-local", type=Path, required=True)
    p_shard.add_argument("--cache-size-gb", type=float, default=1800)
    p_shard.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p_shard.add_argument("--split", choices=("train", "val", "both"), default="both")
    p_shard.add_argument("--start", type=int, default=0)
    p_shard.add_argument("--end", type=int, default=None)
    p_shard.add_argument("--stop-frac", type=float, default=0.97)
    p_shard.add_argument("--log-every", type=int, default=500)
    p_shard.add_argument("--log", type=Path, default=Path("/tmp/plant2_cache_prefill.log"))

    p_par = sub.add_parser("parallel", help="Parallel sharded prefill")
    p_par.add_argument("--ds", type=Path, default=default_split / "train")
    p_par.add_argument("--ds-val", type=Path, default=default_split / "val")
    p_par.add_argument("--ds-local", type=Path, default=Path("/tmp/plant2_ds_cache_spatial_aug"))
    p_par.add_argument("--cache-size-gb", type=int, default=1800)
    p_par.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p_par.add_argument("--max-workers", type=int, default=None)
    p_par.add_argument("--spawn-stagger-sec", type=float, default=3.0)
    p_par.add_argument("--dry-run", action="store_true")
    p_par.add_argument("--python", dest="python_exe", default=None)
    p_par.add_argument("--log-dir", type=Path, default=Path("/tmp/plant2_prefill_logs"))

    p25 = sub.add_parser("2p5", help="Extract+patch 2.5 cache from full spatial cache")
    p25.add_argument("--src", type=Path, default=Path("/tmp/plant2_ds_cache_spatial_aug"))
    p25.add_argument("--dst", type=Path, default=Path("/tmp/plant2_ds_cache_2p5_tsfix"))
    p25.add_argument("--split", type=Path, default=sh / "plant2_l1_fv_experts_split_signs_2.5")
    p25.add_argument("--cache-size-gb", type=float, default=400)
    p25.add_argument("--reset-dst", action="store_true")
    p25.add_argument("--materialize-missing", action="store_true")
    p25.add_argument("--skip-verify", action="store_true")

    args = parser.parse_args(argv)
    handlers = {"shard": cmd_shard, "parallel": cmd_parallel, "2p5": cmd_2p5}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
