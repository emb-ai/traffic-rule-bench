#!/usr/bin/env python3
"""
Sanity-check Plant2 sign-aware trajectory `.pt` files produced by
`collect_benchmark_sign_trajectories.py`.

Verifies:
  - Episode keys (pdd_code, sign_type, sign_id, return, num_steps).
  - `plant2_batch` has all keys expected by HFLM.forward().
  - `plant2_batch["sign_id"]` is a 1-element int matching SIGN_ID_MAP[pdd_code].
  - Tensor shapes/dtypes per Plant2 conventions.
  - No NaN/Inf in numeric arrays.
  - Per-sign aggregates (count, mean num_steps, arrived %, mean return).

Usage:
  python validate_v4.py \\
      --data-dir pdd-bench/outputs/benchmark_sign_trajectories_v4 \\
      [--per-sign-sample 3]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# v2 (2026-04-28): 34 PDD codes. Original 32 ids unchanged; 2.5 / 5.16 added
# at ids 32 / 33 (see collect_benchmark_sign_trajectories.SIGN_ID_MAP).
SIGN_ID_MAP: Dict[str, int] = {
    "2.1": 0, "2.2": 1, "2.3.1": 2, "2.3.2": 3, "2.3.3": 4, "2.4": 5,
    "3.1": 6, "3.2": 7, "3.18.2": 8, "3.19": 9, "3.20": 10, "3.24": 11,
    "3.25": 12, "3.27": 13, "3.31": 14,
    "4.2.1": 15, "4.2.2": 16, "4.2.3": 17, "4.6": 18,
    "5.11.1": 19, "5.11.2": 20, "5.12.1": 21, "5.12.2": 22,
    "5.13.1": 23, "5.13.2": 24, "5.13.3": 25, "5.13.4": 26,
    "5.14.1": 27, "5.14.2": 28, "5.14.3": 29, "5.31": 30, "5.32": 31,
    "2.5": 32, "5.16": 33,
}

EXPECTED_PB_KEYS = {
    "idxs", "x_objs", "route_original", "speed_limit",
    "BEV", "input_ego_speed", "sign_id", "target_speed",
}

# Spec: shape & dtype for each plant2_batch key (only fields we care about).
SPEC: Dict[str, Tuple[Tuple[Optional[int], ...], np.dtype]] = {
    "idxs":            ((1, 30),   np.int32),
    "x_objs":          ((31, 7),   np.float32),
    "route_original":  ((1, 20, 2), np.float32),
    "speed_limit":     ((1,),      np.int64),
    "BEV":             ((1, 3, 128, 128), np.float32),
    "input_ego_speed": ((1, 1),    np.float32),
    "sign_id":         ((1,),      np.int64),
    "target_speed":    ((1, 1),    np.float32),
}


def _shape_matches(a_shape: Tuple[int, ...], spec: Tuple[Optional[int], ...]) -> bool:
    if len(a_shape) != len(spec):
        return False
    return all(s is None or s == d for s, d in zip(spec, a_shape))


def _check_step(pb: Dict[str, Any], strict: bool) -> List[str]:
    errors: List[str] = []
    missing = EXPECTED_PB_KEYS - set(pb.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    for key, (shape_spec, dtype_spec) in SPEC.items():
        v = pb.get(key)
        if v is None:
            if key in {"sign_id", "target_speed"} and strict:
                errors.append(f"{key} is None")
            continue
        a = np.asarray(v)
        if not _shape_matches(a.shape, shape_spec):
            errors.append(f"{key} shape {a.shape} != {shape_spec}")
        if a.dtype != dtype_spec:
            errors.append(f"{key} dtype {a.dtype} != {dtype_spec}")
        if a.dtype.kind in "fc" and (np.isnan(a).any() or np.isinf(a).any()):
            errors.append(f"{key} contains NaN/Inf")
    return errors


def validate_episode(path: Path, strict: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": str(path), "errors": []}
    try:
        ep = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:
        out["errors"].append(f"torch.load failed: {exc}")
        return out

    pdd     = ep.get("pdd_code")
    sign_id = ep.get("sign_id")
    out.update(
        pdd_code=pdd,
        sign_type=ep.get("sign_type"),
        sign_id=int(sign_id) if sign_id is not None else None,
        ret=float(ep.get("return", 0.0)),
        num_steps=int(ep.get("num_steps", 0)),
    )

    if pdd is None:
        out["errors"].append("missing pdd_code")
    elif pdd in SIGN_ID_MAP and sign_id is not None and int(sign_id) != SIGN_ID_MAP[pdd]:
        out["errors"].append(
            f"episode sign_id={sign_id} != SIGN_ID_MAP[{pdd}]={SIGN_ID_MAP[pdd]}"
        )

    steps = ep.get("steps", [])
    if not steps:
        out["errors"].append("no steps")
        return out

    n_step_errs = 0
    for i, step in enumerate(steps):
        pb = step.get("plant2_batch")
        if pb is None:
            out["errors"].append(f"step {i}: plant2_batch is None")
            n_step_errs += 1
            continue
        # Per-step sign_id should match episode sign_id.
        per_step_sid = pb.get("sign_id")
        if per_step_sid is not None and sign_id is not None:
            if int(np.asarray(per_step_sid).flat[0]) != int(sign_id):
                out["errors"].append(
                    f"step {i}: plant2_batch.sign_id "
                    f"{int(np.asarray(per_step_sid).flat[0])} != episode {int(sign_id)}"
                )
                n_step_errs += 1
        errs = _check_step(pb, strict=strict)
        if errs:
            for e in errs[:2]:
                out["errors"].append(f"step {i}: {e}")
            n_step_errs += 1
        if n_step_errs >= 5:
            out["errors"].append(f"... (truncating; {n_step_errs}+ bad steps)")
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--per-sign-sample",
        type=int,
        default=3,
        help="Validate this many .pt files per sign (0 = all).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing optional keys (sign_id/target_speed) as errors.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: not a directory: {data_dir}")
        return 1

    files = sorted(data_dir.glob("*.pt"))
    print(f"Found {len(files)} .pt files in {data_dir}")

    by_sign: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        slug = f.stem.rsplit("_ep", 1)[0]
        by_sign[slug].append(f)

    print(f"Codes: {len(by_sign)}")
    sample_files: List[Path] = []
    for slug, fs in by_sign.items():
        if args.per_sign_sample > 0:
            sample_files.extend(fs[: args.per_sign_sample])
        else:
            sample_files.extend(fs)

    print(f"Validating {len(sample_files)} files ...")
    results = [validate_episode(f, strict=args.strict) for f in sample_files]

    n_ok      = sum(1 for r in results if not r["errors"])
    n_bad     = len(results) - n_ok
    print(f"\nResults: {n_ok} OK, {n_bad} with errors")

    if n_bad:
        print("\nFiles with errors:")
        for r in results:
            if r["errors"]:
                print(f"  {Path(r['path']).name}")
                for e in r["errors"][:5]:
                    print(f"      {e}")

    print("\n=== Per-sign aggregates (sampled files only) ===")
    print(f"{'code':<8s} {'sign_id':>8s} {'n':>4s} {'mean_steps':>12s} {'mean_return':>12s} {'errs':>5s}")
    by_code: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        code = r.get("pdd_code") or "?"
        by_code[code].append(r)

    for code in sorted(by_code.keys(), key=lambda c: SIGN_ID_MAP.get(c, 999)):
        rs = by_code[code]
        sid = SIGN_ID_MAP.get(code, "?")
        n = len(rs)
        ms = np.mean([r["num_steps"] for r in rs]) if n else 0.0
        mr = np.mean([r["ret"]       for r in rs]) if n else 0.0
        ne = sum(1 for r in rs if r["errors"])
        print(f"{code:<8s} {str(sid):>8s} {n:>4d} {ms:>12.1f} {mr:>12.1f} {ne:>5d}")

    print("\n=== Per-sign file counts (full dataset) ===")
    for slug in sorted(by_sign.keys(),
                       key=lambda s: SIGN_ID_MAP.get(s.replace("_", "."), 999)):
        code = slug.replace("_", ".")
        sid = SIGN_ID_MAP.get(code, "?")
        print(f"  {code:<8s} sign_id={sid:>3}  files={len(by_sign[slug])}")

    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
