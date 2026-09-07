"""Shared route × density × ego-speed × NPC-profile expansion cells.

Every sign family expands the same world grid unless it opts out of the ego
spawn-speed axis (speed plates: task-conditioned brake/accel approach).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from traffic_bench.eval.engine.traffic.spawn_velocity_levels import (
    resolve_spawn_velocity_levels_ms,
)
from traffic_bench.eval.engine.traffic.traffic_density_levels import (
    TrafficDensityLevel,
    resolve_traffic_density_levels,
)


DEFAULT_HORIZON_STEPS = 600
DEFAULT_ROUTE_LENGTH_LEVELS_M: tuple[float, ...] = (90.0, 120.0)
DEFAULT_MAX_PATH_LENGTH_M = 90.0


@dataclass(frozen=True)
class WorldAxisCell:
    route_length_m: float
    density: TrafficDensityLevel
    npc_var: int
    spawn_velocity_ms: Optional[float]
    spawn_velocity_level_id: Optional[int]

    def seed_tags(self) -> tuple:
        return (
            int(round(float(self.route_length_m))),
            int(self.density.id),
            int(self.npc_var),
            int(self.spawn_velocity_level_id)
            if self.spawn_velocity_level_id is not None
            else -1,
        )

    def scene_suffix(self, *, route_augment: bool) -> str:
        parts: List[str] = []
        if route_augment:
            parts.append(f"rl{int(round(float(self.route_length_m)))}")
        parts.append(f"td{int(self.density.percentile)}")
        if self.spawn_velocity_level_id is not None and self.spawn_velocity_ms is not None:
            parts.append(f"sv{int(self.spawn_velocity_level_id)}")
        parts.append(f"v{int(self.npc_var)}")
        return "_".join(parts)


def iter_world_axis_cells(
    *,
    route_levels: Sequence[float],
    sim: object,
    task_conditioned_spawn: bool = False,
) -> Iterator[WorldAxisCell]:
    """Cartesian product of route × density × (optional ego speed) × NPC vars."""
    dens_levels = resolve_traffic_density_levels(sim)
    n_variations = max(1, int(getattr(sim, "n_variations", 3) or 3))
    if task_conditioned_spawn:
        speed_slots: List[tuple[Optional[int], Optional[float]]] = [(None, None)]
    else:
        speeds = resolve_spawn_velocity_levels_ms(sim)
        speed_slots = [(i, float(v)) for i, v in enumerate(speeds)]

    for path_len_m in route_levels:
        for dens in dens_levels:
            for spd_id, spd in speed_slots:
                for npc_var in range(n_variations):
                    yield WorldAxisCell(
                        route_length_m=float(path_len_m),
                        density=dens,
                        npc_var=int(npc_var),
                        spawn_velocity_ms=spd,
                        spawn_velocity_level_id=spd_id,
                    )


def describe_world_axes(sim: object, *, task_conditioned_spawn: bool = False) -> str:
    from traffic_bench.eval.engine.spawn.route_length_levels import list_route_length_levels

    routes = list(list_route_length_levels(sim))
    dens = resolve_traffic_density_levels(sim)
    n_var = max(1, int(getattr(sim, "n_variations", 3) or 3))
    dens_s = ",".join(f"p{d.percentile}={d.traffic_density:.3f}" for d in dens)
    if task_conditioned_spawn:
        speed_s = "task-conditioned"
    else:
        speeds = resolve_spawn_velocity_levels_ms(sim)
        speed_s = ",".join(f"{v:.2f}" for v in speeds)
    return (
        f"route={routes} dens=[{dens_s}] spawn_v=[{speed_s}] "
        f"n_npc={n_var} horizon={int(getattr(sim, 'horizon', DEFAULT_HORIZON_STEPS) or DEFAULT_HORIZON_STEPS)}"
    )


def sample_profile_for_cell(cell: WorldAxisCell, *, seed: int, sim: object) -> dict:
    """NPC IDM draw with density fixed to the cell's calibrated probe."""
    from traffic_bench.eval.engine.traffic.agent_profile_bank import sample_one_profile

    return sample_one_profile(
        int(seed),
        density_cap=float(getattr(sim, "profile_density_cap", 1.0) or 1.0),
        horizon_steps=int(getattr(sim, "horizon", DEFAULT_HORIZON_STEPS) or DEFAULT_HORIZON_STEPS),
        traffic_density=float(cell.density.traffic_density),
        nuplan_vehicles_per_frame=float(cell.density.nuplan_vehicles_per_frame),
    )


def stamp_world_axis_fields(row: dict, cell: WorldAxisCell) -> dict:
    """Attach density / spawn-speed probe metadata used in summaries."""
    out = dict(row)
    dens = cell.density
    out["density_level_id"] = int(dens.id)
    out["density_level_name"] = str(dens.name)
    out["density_percentile"] = int(dens.percentile)
    out["nuplan_per_lane"] = float(dens.nuplan_per_lane)
    if cell.spawn_velocity_ms is not None:
        out["spawn_velocity_ms"] = float(cell.spawn_velocity_ms)
        out["spawn_velocity_level_id"] = int(cell.spawn_velocity_level_id or 0)
    return out
