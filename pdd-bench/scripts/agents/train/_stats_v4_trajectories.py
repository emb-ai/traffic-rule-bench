#!/usr/bin/env python3
"""Quick stats over benchmark_sign_trajectories_v4/*.pt (I/O-bound: use threads)."""
from __future__ import annotations

import sys

# Episodes were pickled with references to `traffic_signs` / MetaDrive; match
# collector sys.path so torch.load can resolve classes.
_SDC_S = "/home/jovyan/shares/SR006.nfs2/smirnova/sdc"
for _p in (
    f"{_SDC_S}/pdd-bench",
    f"{_SDC_S}/metadrive",
    f"{_SDC_S}/pdd-bench/scripts/per_sign_bench",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(
    "/home/jovyan/shares/SR006.nfs2/arbelyaev/sdc/pdd-bench/outputs/"
    "benchmark_sign_trajectories_v4"
)
EXPECTED = [
    "2.1", "2.2", "2.3.1", "2.3.2", "2.3.3", "2.4",
    "3.1", "3.2", "3.18.2", "3.19", "3.20", "3.24", "3.25", "3.27", "3.31",
    "4.2.1", "4.2.2", "4.2.3", "4.6",
    "5.11.1", "5.11.2", "5.12.1", "5.12.2",
    "5.13.1", "5.13.2", "5.13.3", "5.13.4",
    "5.14.1", "5.14.2", "5.14.3", "5.31", "5.32",
]


def _load_one(p: Path):
    try:
        ep = torch.load(p, weights_only=False, map_location="cpu")
        return (
            ep.get("pdd_code", "?"),
            float(ep.get("return", 0.0)),
            int(ep.get("num_steps", 0)),
            None,
        )
    except Exception as exc:
        return ("?", 0.0, 0, str(exc))


def main() -> None:
    files = sorted(ROOT.glob("*.pt"))
    print(f"Directory: {ROOT}")
    print(f"Total .pt episodes: {len(files)}")
    print(f"Expected (32 codes × 100): 3200\n")

    by_code: dict[str, list[tuple[float, int]]] = defaultdict(list)
    errs: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(zip(files, ex.map(_load_one, files)))
    for p, (code, r, ns, err) in rows:
        if err:
            errs.append((p.name, err))
        else:
            by_code[code].append((r, ns))

    if errs:
        print(f"LOAD ERRORS: {len(errs)}")
        for name, msg in errs[:8]:
            print(f"  {name}: {msg}")
        print()

    all_r = [t[0] for xs in by_code.values() for t in xs]
    all_s = [t[1] for xs in by_code.values() for t in xs]
    print("=== Global (all episodes) ===")
    print(f"  mean return:       {np.mean(all_r):.2f}")
    print(f"  std return:        {np.std(all_r):.2f}")
    print(f"  min / max return:  {np.min(all_r):.2f} / {np.max(all_r):.2f}")
    print(f"  mean num_steps:    {np.mean(all_s):.1f}")
    print(f"  std num_steps:     {np.std(all_s):.1f}")
    print(f"  min / max steps:   {int(np.min(all_s))} / {int(np.max(all_s))}")
    print()

    codes = sorted(
        by_code.keys(),
        key=lambda c: [int(x) if x.isdigit() else x for x in c.split(".")],
    )
    print("=== Per PDD code ===")
    hdr = f"{'code':<10} {'n':>4} {'mean_ret':>10} {'std_ret':>9} {'mean_steps':>11} {'std_steps':>10}"
    print(hdr)
    print("-" * len(hdr))
    for c in codes:
        xs = by_code[c]
        n = len(xs)
        r = [t[0] for t in xs]
        s = [t[1] for t in xs]
        print(
            f"{c:<10} {n:>4} {np.mean(r):>10.2f} {np.std(r):>9.2f} "
            f"{np.mean(s):>11.1f} {np.std(s):>10.1f}"
        )

    bad = [(c, len(by_code[c])) for c in codes if len(by_code[c]) != 100]
    print()
    if bad:
        print("=== Codes not at 100 episodes ===")
        for c, n in bad:
            print(f"  {c}: {n}")
    else:
        print("All present codes have exactly 100 episodes.")

    missing = [c for c in EXPECTED if c not in by_code]
    if missing:
        print()
        print(f"=== Missing codes (no .pt): {len(missing)} ===")
        print(" ", ", ".join(missing))


if __name__ == "__main__":
    main()
