"""How a run is scored, in one place.

`sign_compliance` as the benchmark reports it means "no violation step was ever
recorded". For the detour signs that can be satisfied without performing the
manoeuvre, in two ways:

1. **End the episode before the zone.** ``DetourSign._is_violating`` only
   evaluates inside [zone_start, zone_end]. A car that leaves the road on the
   approach records zero violations and scores a perfect run. This was the
   dominant case: the checkpoint with the best headline number (0.814) reached
   the zone in only 42% of runs.
   ``in_zone_total_steps`` does NOT reveal it — ``_ego_in_sign_zone`` counts
   the approach lookahead as in-zone too.

2. **Leave the drivable surface inside the zone.** Fixed in
   ``pdd-bench/traffic_signs/detour_sign.py``: the zone test now runs before
   the drivable-area test, and crediting the lane change requires ``on_lane``.

So compliance here additionally requires ``reached_zone``, a per-episode flag
the sign now exports. On the same checkpoints the strict and lenient numbers
differ by roughly 0.03; on the pre-fix ones they differed by 0.7.

The stop benchmark (2.5) is scored by a different runner and has its own
quirks; see `stop_summary`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Summary:
    """Aggregate over one benchmark run."""

    benchmark: str
    runs: int
    compliance: float
    success: float
    #: Fraction that actually entered the sign's zone of effect. A low value
    #: with a high compliance is the signature of the escape hatch above.
    reached: float | None
    off_road: float
    crashed: float
    mean_distance_m: float

    def as_row(self) -> str:
        reached = "   —  " if self.reached is None else f"{self.reached:6.2f}"
        return (f"{self.benchmark:10s} {self.runs:5d} {self.compliance:11.3f} "
                f"{self.success:8.3f} {reached} {self.off_road:8.2f} "
                f"{self.crashed:7.2f} {self.mean_distance_m:8.1f}")

    @staticmethod
    def header() -> str:
        return (f"{'benchmark':10s} {'runs':>5} {'compliance':>11} {'success':>8} "
                f"{'reached':>6} {'off_road':>8} {'crash':>7} {'dist_m':>8}\n"
                + "-" * 68)


def _load(paths: list[Path]) -> list[dict]:
    episodes: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def _reached_zone(episode: dict) -> bool | None:
    """None when the eval predates the flag, so old dirs cannot be misread."""
    flags = episode.get("reached_zone_by_class")
    if flags is None:
        return None
    return any(name.startswith("Detour") and value for name, value in flags.items())


def _episodes(eval_family_dir: Path) -> list[dict]:
    files = sorted(eval_family_dir.glob("parts/*/benchmark/policy_eval/*/episodes_*.jsonl"))
    if not files:
        raise SystemExit(f"No episodes under {eval_family_dir}")
    return _load(files)


def family_summary(eval_family_dir: Path, family) -> Summary:
    """Score one family's held-out runs.

    `compliance` is "no violation step was recorded"; for the detour signs it
    additionally requires `reached_zone`, because there the benchmark's own
    number can be satisfied by never arriving. The other families have no zone
    flag, so `reached` is reported as unknown and their compliance is the
    benchmark's own definition.
    """
    episodes = [e for e in _episodes(eval_family_dir) if e.get("ok", True)]
    if not episodes:
        raise SystemExit(f"No episode in {eval_family_dir} started successfully")
    n = len(episodes)

    if family.group == "detour":
        flags = [_reached_zone(e) for e in episodes]
        if any(f is None for f in flags):
            raise SystemExit(
                f"{eval_family_dir} predates `reached_zone_by_class`. Re-run the "
                "eval rather than reporting a number that cannot be compared.")
        compliant = sum(1 for f, e in zip(flags, episodes)
                        if f and not e["sign_violations"])
        reached = sum(1 for f in flags if f) / n
    else:
        compliant = sum(1 for e in episodes if not e.get("sign_violations"))
        reached = None

    return Summary(
        benchmark=f"{family.name} ({family.sign})",
        runs=n,
        compliance=compliant / n,
        success=sum(1 for e in episodes
                    if e.get("success") or e.get("reached_dest")) / n,
        reached=reached,
        off_road=sum(1 for e in episodes if e.get("out_of_road")) / n,
        crashed=sum(1 for e in episodes if e.get("crashed")) / n,
        mean_distance_m=sum(e.get("distance_travelled_m", 0.0) for e in episodes) / n,
    )


def report(exp) -> str:
    """One table: a row per family, then a row per group."""
    from . import config as cfg

    lines = [Summary.header()]
    per_group: dict[str, list[Summary]] = {}
    for name in exp.families:
        family = cfg.FAMILIES[name]
        summary = family_summary(exp.eval_dir / name, family)
        lines.append(summary.as_row())
        per_group.setdefault(family.group, []).append(summary)

    lines.append("")
    for group, rows in per_group.items():
        runs = sum(r.runs for r in rows)
        weighted = lambda pick: sum(pick(r) * r.runs for r in rows) / runs
        lines.append(Summary(
            benchmark=f"ALL {group}",
            runs=runs,
            compliance=weighted(lambda r: r.compliance),
            success=weighted(lambda r: r.success),
            reached=(weighted(lambda r: r.reached)
                     if all(r.reached is not None for r in rows) else None),
            off_road=weighted(lambda r: r.off_road),
            crashed=weighted(lambda r: r.crashed),
            mean_distance_m=weighted(lambda r: r.mean_distance_m),
        ).as_row())
    return "\n".join(lines)
