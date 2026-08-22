"""Fixed traffic-density tiers derived from nuPlan frame-level vehicle counts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

MAX_TRAFFIC_DENSITY_LEVELS = 3
META_DENSITY_SCALE = 80.0
META_DENSITY_CAP = 0.5

# nuPlan densities.csv percentiles -> MetaDrive traffic_density (count / 80, capped).
_DEFAULT_PERCENTILES = (25, 50, 75)
_DEFAULT_NAMES = ("low", "medium", "high")


@dataclass(frozen=True)
class TrafficDensityLevel:
    id: int
    name: str
    percentile: int
    nuplan_vehicles_per_frame: float
    traffic_density: float

    def describe(self) -> str:
        return (
            f"{self.name}: {self.nuplan_vehicles_per_frame:.1f} vehicles/frame "
            f"(nuPlan p{self.percentile}) -> MetaDrive density {self.traffic_density:.4f}"
        )


def _nuplan_stats_dir() -> Path:
    """CSV stats live next to ``nuplan_sampler`` under ``lib/profiles/``."""
    return Path(__file__).resolve().parents[1] / "profiles" / "nuplan_statistics"


def _load_density_counts(stats_dir: Path | None = None) -> np.ndarray:
    import pandas as pd

    stats_dir = stats_dir or _nuplan_stats_dir()
    densities_path = stats_dir / "densities.csv"
    if not densities_path.is_file():
        raise FileNotFoundError(f"nuPlan densities not found: {densities_path}")
    return pd.read_csv(densities_path)["count"].dropna().to_numpy(dtype=float)


def build_traffic_density_levels(
    *,
    percentiles: tuple[int, ...] = _DEFAULT_PERCENTILES,
    density_cap: float = META_DENSITY_CAP,
) -> dict[int, TrafficDensityLevel]:
    counts = _load_density_counts()
    levels: dict[int, TrafficDensityLevel] = {}
    for idx, percentile in enumerate(percentiles[:MAX_TRAFFIC_DENSITY_LEVELS], start=1):
        raw = float(np.percentile(counts, percentile))
        meta = round(float(np.clip(raw / META_DENSITY_SCALE, 0.0, density_cap)), 4)
        name = _DEFAULT_NAMES[idx - 1] if idx - 1 < len(_DEFAULT_NAMES) else f"p{percentile}"
        levels[idx] = TrafficDensityLevel(
            id=idx,
            name=name,
            percentile=int(percentile),
            nuplan_vehicles_per_frame=raw,
            traffic_density=meta,
        )
    return levels


def list_traffic_density_levels(
    num_levels: int = MAX_TRAFFIC_DENSITY_LEVELS,
    *,
    density_cap: float = META_DENSITY_CAP,
) -> List[TrafficDensityLevel]:
    count = min(MAX_TRAFFIC_DENSITY_LEVELS, max(1, int(num_levels)))
    levels = build_traffic_density_levels(density_cap=density_cap)
    return [levels[i] for i in range(1, count + 1)]
