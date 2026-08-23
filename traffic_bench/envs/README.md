# envs/

SUMO MetaDrive env and managers. Official eval sets `skip_auto_signs=True`
and places plates via `eval/signs/*/place.py`.

| File | Role |
|---|---|
| `sumo.py` | `TrafficSignSumoEnv`: config, reset, reward, done |
| `auto_spawn.py` | Pick-lane / trap / ego-before-sign (tools path) |
| `traffic.py` | `SumoTrafficManager` |
| `npc_idm.py` | `SumoTrajectoryIDMPolicy` |
| `pedestrians.py` | Crosswalk pedestrians |
| `crosswalk_enforce.py` | Forced yield-to-pedestrian brake |
| `lane_node_patch.py` | MetaDrive lane-node width patch |

Lane-key parsing lives in `eval/engine/map/lane_keys.py`.
