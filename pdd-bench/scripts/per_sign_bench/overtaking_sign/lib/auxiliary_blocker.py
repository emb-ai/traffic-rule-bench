"""Spawn a single stationary mid-lane blocker for 3.20 no-overtaking."""

from __future__ import annotations

import logging
from typing import Optional

from metadrive.manager.base_manager import BaseManager
from metadrive.policy.base_policy import BasePolicy

logger = logging.getLogger(__name__)

MIN_SPAWN_LONGITUDE_M = 3.0


class StationaryPolicy(BasePolicy):
    """Keep the vehicle completely stationary."""

    def act(self, *args, **kwargs):
        return [0.0, -0.5]


class StationaryBlockerManager(BaseManager):
    """One stationary NPC on a given lane at an absolute longitude."""

    def __init__(
        self,
        spawn_lane_index: str,
        spawn_long_m: float,
        destination_lane: Optional[str] = None,
    ):
        super().__init__()
        self._spawn_lane_index = spawn_lane_index
        self._spawn_long_m = float(spawn_long_m)
        self._destination_lane = destination_lane
        self._aux_vehicles: list = []

    def after_reset(self):
        self._aux_vehicles = []
        self._spawn_one()

    def _spawn_one(self) -> bool:
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle

        road_network = self.engine.current_map.road_network
        lane = road_network.get_lane(self._spawn_lane_index)
        if lane is None:
            logger.error("[OvertakeAux] lane not found: %s", self._spawn_lane_index)
            return False

        spawn_long = max(
            MIN_SPAWN_LONGITUDE_M,
            min(self._spawn_long_m, float(lane.length) - 1.0),
        )
        try:
            vehicle_config = dict(self.engine.global_config["vehicle_config"])
            vehicle_config.update(
                {
                    "spawn_lane_index": self._spawn_lane_index,
                    "spawn_longitude": spawn_long,
                    "spawn_lateral": 0.0,
                    "enable_reverse": False,
                }
            )
            aux = self.spawn_object(DefaultVehicle, vehicle_config=vehicle_config)
            pos = lane.position(spawn_long, 0.0)
            heading = lane.heading_theta_at(spawn_long)
            aux.set_position([float(pos[0]), float(pos[1])])
            aux.set_heading_theta(heading)
            if aux.navigation is not None:
                aux.reset_navigation(lane)
                if self._destination_lane:
                    aux.navigation.set_route(
                        self._spawn_lane_index, self._destination_lane
                    )
            aux.set_velocity([0.0, 0.0], in_local_frame=True)
            self.add_policy(aux.id, StationaryPolicy, aux, self.generate_seed())
            self._aux_vehicles.append(aux)
            print(
                f"[OvertakeAux] stationary blocker on {self._spawn_lane_index} "
                f"at long={spawn_long:.1f}m (len={lane.length:.1f}m)"
            )
            return True
        except Exception as exc:
            logger.warning("[OvertakeAux] spawn failed: %s", exc)
            print(f"[OvertakeAux] spawn failed: {exc}")
            return False


def spawn_stationary_blocker(
    env,
    *,
    spawn_lane_index: str,
    spawn_long_m: float,
    destination_lane: Optional[str] = None,
):
    """Place one stationary NPC at absolute longitude ``spawn_long_m``."""
    if not hasattr(env, "engine") or env.engine is None:
        logger.error("[OvertakeAux] env has no engine")
        return None
    try:
        lane = env.engine.current_map.road_network.get_lane(spawn_lane_index)
    except Exception as exc:
        logger.error("[OvertakeAux] lane %s: %s", spawn_lane_index, exc)
        return None
    if lane is None:
        logger.error("[OvertakeAux] lane not found: %s", spawn_lane_index)
        return None

    mgr = StationaryBlockerManager(
        spawn_lane_index=spawn_lane_index,
        spawn_long_m=spawn_long_m,
        destination_lane=destination_lane,
    )
    env.engine.register_manager("auxiliary_agent_manager", mgr)
    mgr.after_reset()
    return mgr
