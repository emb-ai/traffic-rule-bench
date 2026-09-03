"""Traffic density drawn from the nuPlan distribution, not bucketed into tiers.

The three tiers this replaces (low/medium/high at nuPlan percentiles 25/50/75)
presented three points of a distribution as three kinds of scene, and the value
each of them carried came from `count / 80` -- a divisor no measurement
supported, applied to a per-frame total that is not comparable to a per-lane
spawn fraction. A scene now draws its own density, so the benchmark's traffic
spans the distribution instead of sitting on three points of it.

The mapping from a uniform draw to `traffic_density` is quantile matching
against `density_calibration_sumo.json`: the u-th quantile of nuPlan's
`count_moving_r150_per_lane` is looked up on the curve that SumoTrafficManager
was measured to produce on the benchmark's own scenes. Outside the reachable
range the table clamps, which the calibration file records explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

CALIBRATION_NAME = "density_calibration_sumo.json"


def _calibration_path() -> Path:
    return Path(__file__).resolve().parent / "nuplan_statistics" / CALIBRATION_NAME


_TABLE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _table() -> Tuple[np.ndarray, np.ndarray]:
    """(u, density) of the sampling table, read once."""
    global _TABLE
    if _TABLE is None:
        path = _calibration_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"traffic density calibration not found: {path}. Run "
                "tools/nuplan_resample/calibrate_density_sumo.py to produce it."
            )
        data = json.loads(path.read_text())["sampling_table"]
        _TABLE = (np.asarray(data["u"], dtype=float),
                  np.asarray(data["density"], dtype=float))
    return _TABLE


def sample_traffic_density(seed: int) -> float:
    """One scene's density. Deterministic in the seed, so a manifest rebuilds
    identically and a variant of the same cell gets its own traffic."""
    us, ds = _table()
    u = float(np.random.default_rng(int(seed) & 0xFFFFFFFF).random())
    return float(np.interp(u, us, ds))


def density_quantiles(qs=(5, 25, 50, 75, 95)) -> dict:
    """What the sampler spans, for the line the expanders print."""
    _, ds = _table()
    return {int(q): float(np.percentile(ds, q)) for q in qs}


# --- Compatibility ------------------------------------------------------------
# The expanders used to multiply a scene by a list of tiers. They now sample per
# row, but the manifest keeps the two level fields as None so readers that group
# by them degrade to a single group instead of raising.

@dataclass(frozen=True)
class TrafficDensityLevel:
    id: Optional[int]
    name: Optional[str]
    percentile: Optional[int]
    nuplan_vehicles_per_frame: Optional[float]
    traffic_density: float

    def describe(self) -> str:
        return f"sampled: MetaDrive density {self.traffic_density:.4f}"


def sampled_density_level(seed: int) -> TrafficDensityLevel:
    """A level-shaped carrier for one sampled density, so the expanders keep
    their existing plumbing without pretending the value is a tier."""
    return TrafficDensityLevel(
        id=None, name=None, percentile=None, nuplan_vehicles_per_frame=None,
        traffic_density=sample_traffic_density(seed),
    )


def list_traffic_density_levels(num_levels: int = 1, **_) -> List[TrafficDensityLevel]:
    raise RuntimeError(
        "traffic density tiers are gone; a scene samples its own density. Use "
        "sample_traffic_density(seed) or sampled_density_level(seed)."
    )
