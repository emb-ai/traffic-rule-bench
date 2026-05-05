"""TrafficSignEnv variant that uses CityMap (looped grid) instead of PGMap.

Provides a CityMapManager that overrides the default PGMapManager so that
CityMap is used for procedural generation. The rest of TrafficSignEnv
behaviour (sign manager, pedestrians, etc.) is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

PDD_BENCH_DIR = Path(__file__).resolve().parent.parent.parent.parent
METADRIVE_DIR = PDD_BENCH_DIR.parent / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metadrive.component.map.city_map import CityMap, CityBIG
from metadrive.manager.pg_map_manager import PGMapManager

from envs.traffic_sign_env import TrafficSignEnv


class DeterministicCityMap(CityMap):
    """CityMap subclass that propagates the map seed into CityBIG.

    Upstream `CityMap._generate()` calls `CityBIG(...)` WITHOUT passing
    `random_seed`, so every instance starts CityBIG with a fresh random state
    and the SAME map_config["seed"] yields a DIFFERENT graph each process.

    This subclass passes `random_seed=self._config["seed"]` so that the same
    seed always produces the same CityMap graph, both within a single process
    and across separate processes / runs.
    """

    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        seed = self._config.get("seed")
        big_map = CityBIG(
            self._config[self.LANE_NUM],
            self._config[self.LANE_WIDTH],
            self.road_network,
            parent_node_path,
            physics_world,
            random_seed=seed,
        )
        big_map.generate(self._config[self.GENERATE_TYPE], self._config[self.GENERATE_CONFIG])
        self.blocks = big_map.blocks


class CityMapManager(PGMapManager):
    """PG-style map manager that spawns DeterministicCityMap (looped grid)."""

    def reset(self):
        config = self.engine.global_config.copy()
        current_seed = self.engine.global_seed

        if self.maps[current_seed] is None:
            map_config = config["map_config"]
            map_config.update({"seed": current_seed})
            map_config = self.add_random_to_map(map_config)
            map = self.spawn_object(DeterministicCityMap, map_config=map_config, random_seed=None)
            self.current_map = map
            if self.engine.global_config["store_map"]:
                self.maps[current_seed] = map
        else:
            map = self.maps[current_seed]
            self.load_map(map)


class CityMapTrafficSignEnv(TrafficSignEnv):
    """TrafficSignEnv variant using CityMap for procedural maps with loops."""

    def setup_engine(self):
        # Reuse parent setup (registers traffic_sign_manager etc.)
        super().setup_engine()
        # Replace the PGMapManager with our CityMapManager.
        # The parent class registers map_manager during MetaDriveEnv.setup_engine(),
        # so by the time we reach this point map_manager is PGMapManager.
        # We swap it for CityMapManager which uses CityMap instead.
        # Replace map_manager with CityMapManager (use update_manager since it already exists)
        self.engine.update_manager("map_manager", CityMapManager(), destroy_previous_manager=True)
