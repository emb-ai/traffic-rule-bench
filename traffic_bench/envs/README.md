# `envs/`

SUMO/MetaDrive environments and supporting managers used by the benchmark.

> **Evaluation:** official eval sets use `skip_auto_signs=True` and place traffic-sign plates explicitly via `eval/signs/*/place.py`.

| File                   | Role                                                                     |
| ---------------------- | ------------------------------------------------------------------------ |
| `sumo.py`              | `TrafficSignSumoEnv`: environment config, reset, reward, and termination |
| `auto_spawn.py`        | Pick-lane / trap / ego-before-sign utilities                             |
| `traffic.py`           | `SumoTrafficManager`                                                     |
| `npc_idm.py`           | `SumoTrajectoryIDMPolicy`                                                |
| `pedestrians.py`       | Crosswalk pedestrian management                                          |
| `crosswalk_enforce.py` | Forced yielding to pedestrians                                           |
| `lane_node_patch.py`   | MetaDrive lane-node width patch                                          |

Lane-key parsing is implemented in `eval/engine/map/lane_keys.py`.
