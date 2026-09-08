#!/usr/bin/env python3
"""Track oracle collect.sh progress and estimate ETA.

Reads ledgers under ``data/trajectories/<sign>/{final|trajectories_*}/<policy>/``.
Prefers ``final/`` when present (same resume target as collect.sh).

Default policies match collect.sh:
``idm_rule ppo_rule carl_rule plant2_rule`` with IDM ego variants
``default,s1..s4`` (EXTRA_SAMPLES_COMPREHENSIVE=4).

Usage (repo root)::

    python tools/collect_progress.py
    python tools/collect_progress.py --watch 30
    python tools/collect_progress.py --signs yield,stop,crosswalk --watch
    python tools/collect_progress.py --policies idm_rule,carl_rule --detail
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Keep in sync with traffic_bench/eval/run/run_signs_parallel.sh / collect signs
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

# Keep in sync with collect.sh defaults
DEFAULT_POLICIES = ["idm_rule", "ppo_rule", "carl_rule", "plant2_rule"]
IDM_FAMILY = {"idm", "idm_rule"}
DEFAULT_EXTRA_SAMPLES = 4  # → default + s1..sN


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _count_manifest_rows(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n += 1
                continue
            if row.get("valid") is False:
                continue
            n += 1
    return n


def _ego_variants(extra_samples: int) -> list[str]:
    n = max(0, int(extra_samples))
    return ["default"] + [f"s{i}" for i in range(1, n + 1)]


def plan_variants(policies: list[str], extra_samples: int) -> list[tuple[str, int]]:
    """(policy, n_variants) — IDM-family expands to 1+extra_samples."""
    out: list[tuple[str, int]] = []
    for policy in policies:
        if policy in IDM_FAMILY:
            out.append((policy, 1 + max(0, extra_samples)))
        else:
            out.append((policy, 1))
    return out


def resolve_out_base(traj_sign_dir: Path) -> Path | None:
    """Prefer final/, else newest trajectories_* (same as collect.sh resume)."""
    final = traj_sign_dir / "final"
    if final.is_dir():
        return final
    cands = [
        p
        for p in traj_sign_dir.glob("trajectories_*")
        if p.is_dir() and not p.name.startswith("trajectories_*.")
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _ledger_done(pol_dir: Path) -> int:
    """Unique (scene_uid, policy, variant) rows across main + shard ledgers."""
    if not pol_dir.is_dir():
        return 0
    ledgers = list(pol_dir.glob("all_runs.jsonl"))
    ledgers += list(pol_dir.glob("all_runs.w*.jsonl"))
    if not ledgers:
        ledgers = list(pol_dir.glob("*/all_runs.jsonl")) + list(
            pol_dir.glob("*/all_runs.w*.jsonl")
        )
    seen: set[tuple] = set()
    for path in ledgers:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for ln in text.splitlines():
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            key = (row.get("scene_uid"), row.get("policy"), row.get("variant"))
            seen.add(key)
    return len(seen)


@dataclass
class PolicyProgress:
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
    out_base: Path | None
    scenes: int
    policies: list[PolicyProgress]

    @property
    def done(self) -> int:
        return sum(p.done for p in self.policies)

    @property
    def total(self) -> int:
        return sum(p.total for p in self.policies)

    @property
    def pct(self) -> float:
        return 100.0 * self.done / self.total if self.total else 100.0


def collect(
    root: Path,
    *,
    split: str,
    policies: list[str],
    extra_samples: int,
    signs_filter: list[str] | None = None,
    scenes_cap: int | None = None,
) -> list[SignProgress]:
    wanted = set(signs_filter) if signs_filter else None
    planned = plan_variants(policies, extra_samples)
    out: list[SignProgress] = []
    for sign, disk in SIGN_SPECS:
        if wanted is not None and sign not in wanted and disk not in wanted:
            continue
        man = root / "data" / "runs" / disk / split / "real_manifest.jsonl"
        traj_root = root / "data" / "trajectories" / disk
        out_base = resolve_out_base(traj_root) if traj_root.is_dir() else None
        # Include signs that have a manifest and/or an active collection dir.
        if not man.is_file() and out_base is None:
            continue
        scenes = _count_manifest_rows(man) if man.is_file() else 0
        if scenes_cap is not None and scenes_cap > 0 and scenes:
            scenes = min(scenes, scenes_cap)
        # Fall back: infer scene count from catalog under out_base
        if scenes <= 0 and out_base is not None:
            cat = out_base / "catalog.jsonl"
            if not cat.is_file():
                cat = out_base / "_merged" / "catalog.jsonl"
            scenes = _count_manifest_rows(cat)
            if scenes_cap is not None and scenes_cap > 0 and scenes:
                scenes = min(scenes, scenes_cap)

        pols: list[PolicyProgress] = []
        for policy, n_var in planned:
            target = scenes * n_var if scenes else 0
            done = 0
            if out_base is not None:
                done = _ledger_done(out_base / policy)
            if target:
                done = min(done, target)
            pols.append(PolicyProgress(policy=policy, done=done, total=target))
        out.append(
            SignProgress(
                sign=sign,
                disk=disk,
                out_base=out_base,
                scenes=scenes,
                policies=pols,
            )
        )
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
        f"{done:,}/{total:,} traj  left={left:,}"
    )
    rate_txt = f"{rate_ep_h:,.0f} traj/h" if rate_ep_h and rate_ep_h > 0 else "warming up"
    lines.append(f"  rate={rate_txt}  ETA≈{_fmt_dur(eta_s)}  (~{eta_at})")
    lines.append("")

    name_w = max((len(s.sign) for s in signs), default=8)
    for s in signs:
        label = s.out_base.name if s.out_base is not None else "(no out)"
        incomplete = [p for p in s.policies if p.done < p.total]
        if not incomplete and s.total > 0 and not detail:
            status = f"done  [{label}]"
        else:
            parts = [
                f"{p.policy}:{p.done}/{p.total}"
                for p in (s.policies if detail else incomplete)
            ]
            status = (" ".join(parts) if parts else "—") + f"  [{label}]"
        lines.append(
            f"  {s.sign:<{name_w}}  {_bar(s.pct, 12)} {s.pct:5.1f}%  "
            f"{s.done:5d}/{s.total:<5d}  {status}"
        )

    lines.append("")
    pol_done: dict[str, int] = {}
    pol_tot: dict[str, int] = {}
    for s in signs:
        for p in s.policies:
            pol_done[p.policy] = pol_done.get(p.policy, 0) + p.done
            pol_tot[p.policy] = pol_tot.get(p.policy, 0) + p.total
    for policy in pol_tot:
        d, t = pol_done[policy], pol_tot[policy]
        p = 100.0 * d / t if t else 100.0
        lines.append(f"  policy {policy:<12} {_bar(p, 12)} {p:5.1f}%  {d:,}/{t:,}")

    return "\n".join(lines)


def _parse_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=None, help="Repo root (default: parent of tools/)")
    p.add_argument(
        "--split",
        default="train",
        choices=("train", "test", "val", "debug"),
        help="Manifest split under data/runs/<sign>/<split>/ (default: train)",
    )
    p.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated policy ids (default: collect.sh set)",
    )
    p.add_argument(
        "--signs",
        default="",
        help="Comma/space-separated sign ids to track. Empty = all with data.",
    )
    p.add_argument(
        "--extra-samples",
        type=int,
        default=DEFAULT_EXTRA_SAMPLES,
        help=(
            "EXTRA_SAMPLES_COMPREHENSIVE for idm/idm_rule "
            f"(default: {DEFAULT_EXTRA_SAMPLES} → default+s1..sN)"
        ),
    )
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Cap scenes per sign (same as collect COUNT / ROWS_LIMIT)",
    )
    p.add_argument(
        "--detail",
        action="store_true",
        help="Show every policy per sign, not only incomplete",
    )
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
    signs_filter = _parse_list(args.signs) or None
    if not policies:
        print("no policies", file=sys.stderr)
        return 2
    if signs_filter is not None:
        known = {sign for sign, disk in SIGN_SPECS} | {disk for _, disk in SIGN_SPECS}
        unknown = [s for s in signs_filter if s not in known]
        if unknown:
            print(f"unknown --signs ids: {unknown}", file=sys.stderr)
            print(f"known: {sorted({s for s, _ in SIGN_SPECS})}", file=sys.stderr)
            return 2

    history: list[tuple[float, int]] = []

    def once(*, clear: bool) -> None:
        signs = collect(
            root,
            split=args.split,
            policies=policies,
            extra_samples=args.extra_samples,
            signs_filter=signs_filter,
            scenes_cap=args.count,
        )
        if not signs:
            print(
                f"no trajectories/manifests under {root}/data/trajectories|runs",
                file=sys.stderr,
            )
            return
        done = sum(s.done for s in signs)
        now = time.time()
        history.append((now, done))
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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
