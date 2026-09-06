# signs/ — one folder per sign group

Each group is a set of eval ids that share spawn geometry and plate placement. 


| Group                              | Eval ids                                                           | Sign codes                |
| ---------------------------------- | ------------------------------------------------------------------ | ------------------------- |
| [junction](junction/README.md)     | `main_road`, `secondary`, `yield`, `stop`                          | 2.1, 2.3, 2.4, 2.5        |
| [roundabout](roundabout/README.md) | `roundabout`                                                       | 4.3                       |
| [blocked](blocked/README.md)       | `blocked_road`                                                     | 3.2                       |
| [dual_path](dual_path/README.md)   | `direction_*`, `one_way_*`, `no_turn_*`, `no_entry`                | 4.1.x, 5.7.x, 3.18.x, 3.1 |
| [crosswalk](crosswalk/README.md)   | `crosswalk`                                                        | 5.19                      |
| [detour](detour/README.md)         | `detour_right`, `detour_left`, `detour_either`                     | 4.2.1–4.2.3               |
| [speed](speed/README.md)           | `speed_limit`, `min_speed`, `residential_zone`, `zone_speed_limit` | 3.24, 4.6, 5.21, 5.31     |
| [restricted_lane](restricted_lane/README.md) | `bus_lane`, `bike_lane`, `bus_lane_road`, `bike_lane_road` | 5.14.1, 5.14.2, 5.11.1, 5.11.2 |


Per-group files (skip if unused):


| File        | Role                                             |
| ----------- | ------------------------------------------------ |
| `expand.py` | `generate(cfg, scenes)` + per-scene row builders |
| `spawn.py`  | Legal ego / aux approach combinations            |
| `place.py`  | Where plates go in MetaDrive                     |
| `spec.py`   | Plate class / crop-meta bridge                   |


Dual-path spawn lives in `scene.py`; navigation is `nav.py`. Roundabout aux
placement is `aux.py`. Junction `generate` also serves roundabout.