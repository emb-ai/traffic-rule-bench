#!/usr/bin/env python3
"""Track eval progress for run_signs_parallel.sh and estimate ETA.

Counts episode lines under ``data/runs/<sign>/<split>/eval_out/.../policy_eval``.
Default baselines match the parallel script (16 total):
idm/idm_rule × {default,s1–s4} + ppo_lidar/ppo_rule + carl/carl_rule + plant2/plant2_rule.

Usage (repo root)::

    python tools/eval_progress.py
    python tools/eval_progress.py --split test --watch 30
    python tools/eval_progress.py --policies carl,carl_rule --watch
    python tools/eval_progress.py --signs stop,yield,main_road,secondary --watch 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Keep in sync with traffic_bench/eval/run/run_signs_parallel.sh
SIGN_SPECS: list[tuple[str, str]] = [
    ("main_road", "main_road"),
    ("secondary", "secondary_road"),
    ("yield", "yield"),
    ("stop", "stop"),
    ("roundabout", "roundabout"),
    ("blocked_road", "blocked_road"),
    ("no_entry", "no_entry"),
    ("no_turn/right", "no_turn_right"),
    ("no_turn/left", "no_turn_left"),
    ("direction/straight", "direction_straight"),
    ("direction/right", "direction_right"),
    ("direction/left", "direction_left"),
    ("direction/straight_right", "direction_straight_right"),
    ("direction/straight_left", "direction_straight_left"),
    ("direction/left_right", "direction_left_right"),
    ("one_way/right", "one_way_right"),
    ("one_way/left", "one_way_left"),
    ("detour/right", "detour_right"),
    ("detour/left", "detour_left"),
    ("detour/either", "detour_either"),
    ("speed_limit", "speed_limit"),
    ("min_speed", "min_speed"),
    ("residential_zone", "residential_zone"),
    ("zone_speed_limit", "zone_speed_limit"),
    ("crosswalk", "crosswalk"),
]

DEFAULT_POLICIES = [
    "idm",
    "idm_rule",
    "ppo_lidar",
    "ppo_rule",
    "carl",
    "carl_rule",
    "plant2",
    "plant2_rule",
]
IDM_FAMILY = {"idm", "idm_rule"}
DEFAULT_EGO_VARIANTS = ["default", "s1", "s2", "s3", "s4"]
DEFAULT_EGO = ",".join(DEFAULT_EGO_VARIANTS)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _count_lines(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _run_done(run_dir: Path, policy: str) -> int:
    """Episodes finished for one baseline dir (merged or in-flight shards)."""
    merged = _count_lines(run_dir / f"episodes_{policy}.jsonl")
    shards_dir = run_dir / "_shards"
    shard_n = 0
    if shards_dir.is_dir():
        for path in shards_dir.glob("w*/episodes*.jsonl"):
            shard_n += _count_lines(path)
    # After merge both exist — don't double-count.
    return max(merged, shard_n)


@dataclass
class BaselineProgress:
    run_name: str
    policy: str
    done: int
    total: int

    @property
    def pct(self) -> float:
        return 100.0 * self.done / self.total if self.total else 100.0


@dataclass
class SignProgress:
    sign: str
    disk: str
    scenes: int
    baselines: list[BaselineProgress]

    @property
    def done(self) -> int:
        return sum(b.done for b in self.baselines)

    @property
    def total(self) -> int:
        return sum(b.total for b in self.baselines)

    @property
    def pct(self) -> float:
        return 100.0 * self.done / self.total if self.total else 100.0


def plan_baselines(policies: list[str], ego_variants: list[str]) -> list[tuple[str, str]]:
    """Match ``traffic_bench.eval.run.policies.plan_baselines``: IDM gets all egos."""
    out: list[tuple[str, str]] = []
    for policy in policies:
        if policy in IDM_FAMILY:
            out.extend((policy, ego) for ego in ego_variants)
        else:
            out.append((policy, "default"))
    return out


def collect(
    root: Path,
    *,
    split: str,
    policies: list[str],
    ego_variants: list[str],
    signs_filter: list[str] | None = None,
) -> list[SignProgress]:
    wanted = set(signs_filter) if signs_filter else None
    planned = plan_baselines(policies, ego_variants)
    out: list[SignProgress] = []
    for sign, disk in SIGN_SPECS:
        if wanted is not None and sign not in wanted:
            continue
        man = root / "data" / "runs" / disk / split / "real_manifest.jsonl"
        if not man.is_file():
            continue
        scenes = _count_lines(man)
        pe = (
            root
            / "data"
            / "runs"
            / disk
            / split
            / "eval_out"
            / "benchmark"
            / "full"
            / "policy_eval"
        )
        baselines: list[BaselineProgress] = []
        for policy, ego in planned:
            run_name = f"{policy}_{ego}"
            done = _run_done(pe / run_name, policy) if pe.is_dir() else 0
            baselines.append(
                BaselineProgress(
                    run_name=run_name,
                    policy=policy,
                    done=min(done, scenes),
                    total=scenes,
                )
            )
        out.append(SignProgress(sign=sign, disk=disk, scenes=scenes, baselines=baselines))
    return out


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN
        return "?"
    if seconds > 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    td = timedelta(seconds=int(seconds))
    days = td.days
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    if days:
        return f"{days}d {h:02d}h{m:02d}m"
    if h:
        return f"{h:02d}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(width * min(max(pct, 0.0), 100.0) / 100.0))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(
    signs: list[SignProgress],
    *,
    rate_ep_h: float | None,
    detail: bool,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now()
    done = sum(s.done for s in signs)
    total = sum(s.total for s in signs)
    left = max(0, total - done)
    pct = 100.0 * done / total if total else 100.0
    eta_s = (left / rate_ep_h * 3600.0) if rate_ep_h and rate_ep_h > 0 else None
    eta_at = (now + timedelta(seconds=eta_s)).strftime("%Y-%m-%d %H:%M") if eta_s else "?"

    lines: list[str] = []
    lines.append(
        f"{now.strftime('%H:%M:%S')}  "
        f"{_bar(pct)} {pct:5.1f}%  "
        f"{done:,}/{total:,} ep  left={left:,}"
    )
    rate_txt = f"{rate_ep_h:,.0f} ep/h" if rate_ep_h and rate_ep_h > 0 else "warming up"
    lines.append(f"  rate={rate_txt}  ETA≈{_fmt_dur(eta_s)}  (~{eta_at})")
    lines.append("")

    # Per-sign compact row
    name_w = max((len(s.sign) for s in signs), default=8)
    for s in signs:
        incomplete = [b for b in s.baselines if b.done < b.total]
        if not incomplete and not detail:
            status = "done"
        else:
            parts = [f"{b.policy}:{b.done}/{b.total}" for b in (s.baselines if detail else incomplete)]
            status = " ".join(parts) if parts else "done"
        lines.append(
            f"  {s.sign:<{name_w}}  {_bar(s.pct, 12)} {s.pct:5.1f}%  "
            f"{s.done:5d}/{s.total:<5d}  {status}"
        )

    # Policy totals
    lines.append("")
    pol_done: dict[str, int] = {}
    pol_tot: dict[str, int] = {}
    for s in signs:
        for b in s.baselines:
            pol_done[b.policy] = pol_done.get(b.policy, 0) + b.done
            pol_tot[b.policy] = pol_tot.get(b.policy, 0) + b.total
    for policy in pol_tot:
        d, t = pol_done[policy], pol_tot[policy]
        p = 100.0 * d / t if t else 100.0
        lines.append(f"  policy {policy:<12} {_bar(p, 12)} {p:5.1f}%  {d:,}/{t:,}")

    return "\n".join(lines)


def _parse_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None, help="Repo root (default: parent of tools/)")
    p.add_argument("--split", default="train", choices=("train", "test", "val"))
    p.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated policy ids (default: parallel-script set)",
    )
    p.add_argument(
        "--signs",
        default="",
        help=(
            "Comma/space-separated hydra sign ids to track "
            "(e.g. stop,yield,main_road,secondary). Empty = all."
        ),
    )
    p.add_argument(
        "--ego",
        default=DEFAULT_EGO,
        help=(
            "Comma-separated ego variants for idm/idm_rule "
            f"(default: {DEFAULT_EGO}); other policies always use default"
        ),
    )
    p.add_argument("--detail", action="store_true", help="Show every baseline per sign, not only incomplete")
    p.add_argument(
        "--watch",
        nargs="?",
        const=30,
        type=int,
        metavar="SEC",
        help="Refresh every SEC seconds (default 30) and estimate rate/ETA",
    )
    p.add_argument(
        "--rate-window",
        type=int,
        default=6,
        help="Number of watch samples used for rate (default 6)",
    )
    args = p.parse_args(argv)

    root = (args.root or _repo_root()).resolve()
    policies = _parse_list(args.policies)
    ego_variants = _parse_list(args.ego) or list(DEFAULT_EGO_VARIANTS)
    signs_filter = _parse_list(args.signs) or None
    if not policies:
        print("no policies", file=sys.stderr)
        return 2
    if signs_filter is not None:
        known = {sign for sign, _ in SIGN_SPECS}
        unknown = [s for s in signs_filter if s not in known]
        if unknown:
            print(f"unknown --signs ids: {unknown}", file=sys.stderr)
            print(f"known: {sorted(known)}", file=sys.stderr)
            return 2

    history: list[tuple[float, int]] = []

    def once(*, clear: bool) -> None:
        signs = collect(
            root,
            split=args.split,
            policies=policies,
            ego_variants=ego_variants,
            signs_filter=signs_filter,
        )
        if not signs:
            print(f"no manifests under {root}/data/runs/*/ {args.split}/", file=sys.stderr)
            return
        done = sum(s.done for s in signs)
        now = time.time()
        history.append((now, done))
        # Keep a bounded window
        while len(history) > max(2, args.rate_window):
            history.pop(0)
        rate = None
        if len(history) >= 2:
            t0, d0 = history[0]
            t1, d1 = history[-1]
            dt = t1 - t0
            if dt >= 1.0:
                rate = (d1 - d0) / dt * 3600.0
        text = render(signs, rate_ep_h=rate, detail=args.detail)
        if clear and sys.stdout.isatty():
            # Clear screen for watch mode
            sys.stdout.write("\033[H\033[J")
        print(text, flush=True)

    if args.watch is None:
        once(clear=False)
        return 0

    interval = max(5, int(args.watch))
    try:
        while True:
            once(clear=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 0


if __name__ == "__main__":
    # Avoid BLAS thread storms if imported next to eval
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
