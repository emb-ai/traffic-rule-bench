"""Traffic density: calibrated nuPlan quantile probes + optional full-distribution draw.

Controlled augmentation uses three fixed probes at nuPlan p25/p50/p75 of
``count_moving_r150_per_lane``, mapped to MetaDrive ``traffic_density`` through
``density_calibration_sumo.json``. That is an explicit stress-test grid, not a
claim that traffic has three modes.

``sample_traffic_density`` remains for callers that still want one draw from the
full calibrated curve (e.g. legacy rows). Expand paths that honour the shared
grid should call ``list_traffic_density_levels`` instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

CALIBRATION_NAME = "density_calibration_sumo.json"
# Empirical probes of nuPlan count_moving_r150_per_lane (see calibration file).
DEFAULT_DENSITY_PERCENTILES: Tuple[int, ...] = (25, 50, 75)

META_DENSITY_SCALE = 80.0
META_DENSITY_CAP = 0.5
MAX_TRAFFIC_DENSITY_LEVELS = 3


def _calibration_path() -> Path:
    return Path(__file__).resolve().parent / "nuplan_statistics" / CALIBRATION_NAME


_TABLE: Optional[Tuple[np.ndarray, np.ndarray]] = None
_CALIB: Optional[dict] = None


def _calibration() -> dict:
    global _CALIB
    if _CALIB is None:
        path = _calibration_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"traffic density calibration not found: {path}. Run "
                "tools/nuplan_resample/calibrate_density_sumo.py to produce it."
            )
        _CALIB = json.loads(path.read_text())
    return _CALIB


def _table() -> Tuple[np.ndarray, np.ndarray]:
    """(u, density) of the sampling table, read once."""
    global _TABLE
    if _TABLE is None:
        data = _calibration()["sampling_table"]
        _TABLE = (
            np.asarray(data["u"], dtype=float),
            np.asarray(data["density"], dtype=float),
        )
    return _TABLE


def sample_traffic_density(seed: int) -> float:
    """One density draw from the full calibrated curve (not a fixed probe)."""
    us, ds = _table()
    u = float(np.random.default_rng(int(seed) & 0xFFFFFFFF).random())
    return float(np.interp(u, us, ds))


def density_quantiles(qs=(5, 25, 50, 75, 95)) -> dict:
    """What the calibrated sampler spans, for expander log lines."""
    _, ds = _table()
    return {int(q): float(np.percentile(ds, q)) for q in qs}


def density_at_quantile(q: float) -> float:
    """MetaDrive traffic_density at nuPlan quantile q ∈ [0, 1]."""
    us, ds = _table()
    return float(np.interp(float(q), us, ds))


@dataclass(frozen=True)
class TrafficDensityLevel:
    id: int
    name: str
    percentile: int
    nuplan_per_lane: float
    traffic_density: float

    @property
    def nuplan_vehicles_per_frame(self) -> float:
        """Aux-credit unit: density × legacy MetaDrive scale (not per-lane)."""
        return float(self.traffic_density) * META_DENSITY_SCALE

    def describe(self) -> str:
        return (
            f"{self.name}: nuPlan p{self.percentile} "
            f"({self.nuplan_per_lane:.2f}/lane) → density {self.traffic_density:.4f}"
        )


def _nuplan_per_lane_at(percentile: int) -> float:
    raw = _calibration().get("nuplan_per_lane") or {}
    key = str(int(percentile))
    if key in raw:
        return float(raw[key])
    # Fallback: interpolate from known keys if a custom percentile is requested.
    items = sorted((int(k), float(v)) for k, v in raw.items())
    if not items:
        return float("nan")
    xs = np.asarray([p for p, _ in items], dtype=float)
    ys = np.asarray([v for _, v in items], dtype=float)
    return float(np.interp(float(percentile), xs, ys))


def list_traffic_density_levels(
    num_levels: int = MAX_TRAFFIC_DENSITY_LEVELS,
    percentiles: Sequence[int] | None = None,
    **_,
) -> List[TrafficDensityLevel]:
    """Fixed calibrated probes (default nuPlan p25/p50/p75)."""
    qs = tuple(int(p) for p in (percentiles or DEFAULT_DENSITY_PERCENTILES))
    if num_levels is not None and int(num_levels) > 0:
        qs = qs[: int(num_levels)]
    names = ("sparse", "typical", "dense", "extra")
    out: List[TrafficDensityLevel] = []
    for i, p in enumerate(qs):
        dens = density_at_quantile(p / 100.0)
        out.append(
            TrafficDensityLevel(
                id=i,
                name=names[i] if i < len(names) else f"p{p}",
                percentile=int(p),
                nuplan_per_lane=_nuplan_per_lane_at(int(p)),
                traffic_density=round(float(dens), 4),
            )
        )
    return out


def resolve_traffic_density_levels(sim: object | None = None) -> List[TrafficDensityLevel]:
    """Levels from ``sim.traffic_density_levels`` floats, or default probes."""
    raw = getattr(sim, "traffic_density_levels", None) if sim is not None else None
    if raw:
        vals = [float(x) for x in raw if float(x) >= 0.0]
        if vals:
            defaults = list_traffic_density_levels(num_levels=max(len(vals), 1))
            out: List[TrafficDensityLevel] = []
            for i, dens in enumerate(vals):
                base = defaults[min(i, len(defaults) - 1)]
                out.append(
                    TrafficDensityLevel(
                        id=i,
                        name=base.name if i < len(defaults) else f"d{i}",
                        percentile=base.percentile if i < len(defaults) else -1,
                        nuplan_per_lane=base.nuplan_per_lane if i < len(defaults) else float("nan"),
                        traffic_density=round(float(dens), 4),
                    )
                )
            return out
    return list_traffic_density_levels()


def sampled_density_level(seed: int) -> TrafficDensityLevel:
    """Level-shaped carrier for one full-curve draw (no fixed percentile)."""
    dens = sample_traffic_density(seed)
    return TrafficDensityLevel(
        id=-1,
        name="sampled",
        percentile=-1,
        nuplan_per_lane=float("nan"),
        traffic_density=float(dens),
    )
