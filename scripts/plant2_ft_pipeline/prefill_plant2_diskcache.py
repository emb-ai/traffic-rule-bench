#!/usr/bin/env python3
"""Prefill PlanTDataset diskcache (supports augment=True dual keys).

Env:
  DS, DS_VAL (optional), DS_LOCAL, CACHE_SIZE_GB
  PREFILL_AUGMENT=1          fill base + *_aug keys (default 1)
  PREFILL_SPLIT=train|val|both  (default both)
  PREFILL_LOG, PREFILL_LOG_EVERY, PREFILL_STOP_FRAC
  PREFILL_START, PREFILL_END  index range within the chosen split
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

from _paths import plan_t

PLAN_T = plan_t()
os.chdir(PLAN_T)
if str(PLAN_T) not in sys.path:
    sys.path.insert(0, str(PLAN_T))

from omegaconf import OmegaConf, open_dict
from diskcache import Cache
from dataset import PlanTDataset


def log(msg: str, log_path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def _cache_key(dataset: PlanTDataset, index: int) -> str:
    """The key the dataset itself would use.

    Under WPS_STRIDE>1 the dataset suffixes the key, since the stride changes
    the stored waypoints. Recomputing the bare label here would look up an
    entry that never exists: the base would count as missing right after being
    filled, and the augmented variant would be skipped for every sample.
    """
    labels = dataset.labels[index]
    key_fn = getattr(dataset, "_cache_key", None)
    return key_fn(labels) if key_fn is not None else labels[0].decode()


def _fill_one(dataset: PlanTDataset, cache: Cache, index: int, do_aug: bool) -> tuple[str, str]:
    """Return (base_status, aug_status) in {filled,skipped,error:...}."""
    key = _cache_key(dataset, index)
    base_st = "skipped"
    aug_st = "skipped"

    # Always materialize unaugmented entry first (keeps BEV_aug when cfg.augment).
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
            else:
                aug_st = "skipped"
        except Exception as e:  # noqa: BLE001
            aug_st = f"error:{type(e).__name__}:{e}"

    dataset.transform = old_tf
    return base_st, aug_st


def prefill_split(
    *,
    ds_root: str,
    cache: Cache,
    cfg,
    log_path: Path,
    do_aug: bool,
    start: int,
    end: int | None,
    log_every: int,
    stop_bytes: int,
    tag: str,
) -> dict:
    dataset = PlanTDataset(ds_root.rstrip("/") + "/data", cfg, shared_dict=cache)
    # Keep aug_sample available even though we force paths in _fill_one.
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
    log(f"{tag}: dataset_len={n} range=[{start},{end}) do_aug={do_aug}", log_path)

    t0 = time.time()
    filled_base = filled_aug = skipped = errors = 0
    stopped_early = False
    i = start - 1

    for i in range(start, end):
        if cache.volume() >= stop_bytes:
            stopped_early = True
            log(
                f"{tag}: STOP_NEAR_LIMIT i={i}/{n} volume_gb={cache.volume()/1024**3:.2f}",
                log_path,
            )
            break
        base_st, aug_st = _fill_one(dataset, cache, i, do_aug)
        for st in (base_st, aug_st):
            if st == "filled":
                pass
            elif st == "skipped":
                skipped += 1
            elif st.startswith("error:"):
                errors += 1
                if errors <= 20 or errors % 100 == 0:
                    log(f"{tag}: ERROR i={i} {st}", log_path)
        if base_st == "filled":
            filled_base += 1
        if aug_st == "filled":
            filled_aug += 1

        done = i - start + 1
        span = max(end - start, 1)
        if done % log_every == 0 or i == start:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            log(
                f"{tag}: progress i={i+1}/{n} done={done}/{span} "
                f"filled_base={filled_base} filled_aug={filled_aug} "
                f"skipped={skipped} errors={errors} "
                f"volume_gb={cache.volume()/1024**3:.2f} "
                f"rate={rate:.2f} samp/s elapsed_s={elapsed:.0f}",
                log_path,
            )

    elapsed = time.time() - t0
    return {
        "tag": tag,
        "stopped_early": stopped_early,
        "i_final": i + 1 if i >= 0 else 0,
        "n": n,
        "filled_base": filled_base,
        "filled_aug": filled_aug,
        "skipped": skipped,
        "errors": errors,
        "elapsed_s": elapsed,
        "volume_gb": cache.volume() / 1024**3,
        "len": len(cache),
    }


def main() -> None:
    log_path = Path(os.environ.get("PREFILL_LOG", "/tmp/plant2_cache_prefill.log"))
    log_every = int(os.environ.get("PREFILL_LOG_EVERY", "500"))
    stop_frac = float(os.environ.get("PREFILL_STOP_FRAC", "0.97"))
    do_aug = os.environ.get("PREFILL_AUGMENT", "1").strip() not in ("0", "false", "False", "")
    split = os.environ.get("PREFILL_SPLIT", "both").strip().lower()

    ds_train_root = os.environ["DS"].rstrip("/")
    ds_val_root = os.environ.get("DS_VAL", "").rstrip("/")
    tmp_folder = os.environ["DS_LOCAL"]
    cache_gb = float(os.environ.get("CACHE_SIZE_GB", "1600"))
    size_limit = int(cache_gb * 1024**3)
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(PLAN_T / "config" / "config.yaml")
    user_cfg = OmegaConf.load(PLAN_T / "config" / "user" / "arbelyaev.yaml")
    model_cfg = OmegaConf.load(PLAN_T / "config" / "model" / "PlanT.yaml")
    cfg = OmegaConf.merge(cfg, {"user": user_cfg, "model": model_cfg})
    with open_dict(cfg):
        cfg.use_caching = True
        cfg.model.training.augment = bool(do_aug)
        # parked aug needs missing car_data.npy on this cluster; keep off
        cfg.model.training.augment_parked = False
        # split already filtered good routes — skip per-route results/slurm I/O
        cfg.model.training.filter_routes = False

    log(
        f"PREFILL_START DS={ds_train_root} DS_VAL={ds_val_root or '-'} "
        f"DS_LOCAL={tmp_folder} CACHE_SIZE_GB={cache_gb:g} "
        f"augment={do_aug} split={split} stop_frac={stop_frac}",
        log_path,
    )

    cache = Cache(directory=tmp_folder, size_limit=size_limit)
    log(f"cache_open volume_gb={cache.volume()/1024**3:.2f} len={len(cache)}", log_path)
    stop_bytes = int(stop_frac * size_limit)

    start = int(os.environ.get("PREFILL_START", "0"))
    end_env = os.environ.get("PREFILL_END")
    end = int(end_env) if end_env not in (None, "") else None

    results = []
    if split in ("train", "both"):
        results.append(
            prefill_split(
                ds_root=ds_train_root,
                cache=cache,
                cfg=cfg,
                log_path=log_path,
                do_aug=do_aug,
                start=start,
                end=end,
                log_every=log_every,
                stop_bytes=stop_bytes,
                tag="train",
            )
        )
    if split in ("val", "both"):
        if not ds_val_root:
            # convention: sibling val next to train split root
            cand = Path(ds_train_root).parent / "val"
            if (cand / "data").is_dir():
                ds_val_root = str(cand)
        if ds_val_root and (Path(ds_val_root) / "data").is_dir():
            # val usually small: full range unless explicitly sliced with SPLIT=val
            v_start = start if split == "val" else 0
            v_end = end if split == "val" else None
            results.append(
                prefill_split(
                    ds_root=ds_val_root,
                    cache=cache,
                    cfg=cfg,
                    log_path=log_path,
                    do_aug=do_aug,
                    start=v_start,
                    end=v_end,
                    log_every=log_every,
                    stop_bytes=stop_bytes,
                    tag="val",
                )
            )
        else:
            log(f"WARN: val data missing (DS_VAL={ds_val_root!r})", log_path)

    for r in results:
        log(
            f"PREFILL_DONE_{r['tag']} stopped_early={r['stopped_early']} "
            f"i_final={r['i_final']}/{r['n']} filled_base={r['filled_base']} "
            f"filled_aug={r['filled_aug']} skipped={r['skipped']} errors={r['errors']} "
            f"volume_gb={r['volume_gb']:.2f} len={r['len']} elapsed_s={r['elapsed_s']:.0f}",
            log_path,
        )
    cache.close()


if __name__ == "__main__":
    main()
