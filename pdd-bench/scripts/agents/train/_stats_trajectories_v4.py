#!/usr/bin/env python3
"""One-off: aggregate return / num_steps / counts over v4 .pt trajectories."""
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch


def load_meta(fp: str):
    ep = torch.load(fp, weights_only=False, map_location="cpu")
    code = ep.get("pdd_code") or "?"
    return (
        code,
        float(ep.get("return", 0.0)),
        int(ep.get("num_steps", 0)),
        ep.get("sign_type"),
        os.path.basename(fp),
    )


def sort_key(code: str):
    return [int(x) if x.isdigit() else x for x in code.split(".")]


def main() -> int:
    v4 = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/jovyan/shares/SR006.nfs2/arbelyaev/sdc/pdd-bench/"
        "outputs/benchmark_sign_trajectories_v4"
    )
    files = sorted(glob.glob(os.path.join(v4, "*.pt")))
    print(f"Directory: {v4}")
    print(f"Total .pt files: {len(files)}  (expected 3200 = 32 codes × 100)\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    by_code: dict = defaultdict(list)
    fail = []
    nw = min(12, max(4, len(files) // 100))
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(load_meta, fp): fp for fp in files}
        for fut in as_completed(futs):
            fp = futs[fut]
            try:
                code, ret, nst, stype, _bn = fut.result()
                by_code[code].append(
                    {"return": ret, "steps": nst, "sign_type": stype}
                )
            except Exception as e:
                fail.append((fp, str(e)))

    if fail:
        print(f"LOAD FAILURES: {len(fail)}")
        for fp, e in fail[:8]:
            print(" ", os.path.basename(fp), e)

    codes_sorted = sorted(by_code.keys(), key=sort_key)
    print("=== Per PDD code ===")
    hdr = (
        f"{'pdd_code':<10} {'n':>5} {'mean_steps':>11} {'std_steps':>10} "
        f"{'mean_return':>12} {'std_ret':>9} {'min_ret':>8} {'max_ret':>8}"
    )
    print(hdr)
    for code in codes_sorted:
        xs = by_code[code]
        n = len(xs)
        st = [d["steps"] for d in xs]
        rt = [d["return"] for d in xs]
        print(
            f"{code:<10} {n:5d} {np.mean(st):11.1f} {np.std(st):10.1f} "
            f"{np.mean(rt):12.2f} {np.std(rt):9.2f} {np.min(rt):8.1f} {np.max(rt):8.1f}"
        )

    all_steps = [d["steps"] for xs in by_code.values() for d in xs]
    all_ret = [d["return"] for xs in by_code.values() for d in xs]
    print("\n=== Global (all loaded episodes) ===")
    print(f"  Episodes:     {len(all_steps)}")
    print(f"  Codes:        {len(by_code)} / 32")
    if all_steps:
        print(
            f"  mean_steps:   {np.mean(all_steps):.1f}  "
            f"(std {np.std(all_steps):.1f}, min {np.min(all_steps)}, max {np.max(all_steps)})"
        )
        print(
            f"  mean_return:  {np.mean(all_ret):.2f}  "
            f"(std {np.std(all_ret):.2f}, min {np.min(all_ret):.2f}, max {np.max(all_ret):.2f})"
        )

    EXPECTED = [
        "2.1",
        "2.2",
        "2.3.1",
        "2.3.2",
        "2.3.3",
        "2.4",
        "3.1",
        "3.2",
        "3.18.2",
        "3.19",
        "3.20",
        "3.24",
        "3.25",
        "3.27",
        "3.31",
        "4.2.1",
        "4.2.2",
        "4.2.3",
        "4.6",
        "5.11.1",
        "5.11.2",
        "5.12.1",
        "5.12.2",
        "5.13.1",
        "5.13.2",
        "5.13.3",
        "5.13.4",
        "5.14.1",
        "5.14.2",
        "5.14.3",
        "5.31",
        "5.32",
    ]
    missing = [c for c in EXPECTED if c not in by_code]
    short = [(c, len(by_code[c])) for c in EXPECTED if c in by_code and len(by_code[c]) != 100]
    if missing:
        print(f"\n  Missing codes (no files): {missing}")
    if short:
        print("\n  Codes with n != 100:")
        for c, n in sorted(short, key=lambda x: sort_key(x[0])):
            print(f"    {c}: {n}")

    print("\n=== sign_type (first episode per code) ===")
    for code in codes_sorted:
        st0 = by_code[code][0]["sign_type"]
        print(f"  {code:<10}  {st0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
