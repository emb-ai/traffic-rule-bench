# signs/ — one folder per sign family

Each family is a group of eval ids that share spawn geometry and plate
placement. Dispatch is `sign_registry.family` → `signs.<family>.expand` /
`.place`, not `if sign == one_way`.

| Family | Eval ids | Sign codes |
| --- | --- | --- |
| [junction](junction/README.md) | `main`, `secondary`, `yield`, `stop` | 2.1, 2.3, 2.4, 2.5 |
| [roundabout](roundabout/README.md) | `roundabout` | 4.3 |
| [blocked](blocked/README.md) | `blocked_road` | 3.2 |
| [dual_path](dual_path/README.md) | `direction_*`, `one_way_*`, `no_turn_*`, `no_entry` | 4.1.x, 5.7.x, 3.18.x, 3.1 |
| [crosswalk](crosswalk/README.md) | `crosswalk` | 5.19 |
| [detour](detour/README.md) | `detour_right`, `detour_left`, `detour_either` | 4.2.1–4.2.3 |
| [speed](speed/README.md) | `speed_limit`, `min_speed`, `residential_zone`, `zone_speed_limit` | 3.24, 4.6, 5.21, 5.31 |

Per-family files (skip if unused):

| File | Role |
| --- | --- |
| `expand.py` | Manifest rows (spawn × aux × variations) |
| `spawn.py` | Legal ego / aux approach combinations |
| `place.py` | Where plates go in MetaDrive |
| `spec.py` | Plate class / dual-path crop-meta bridge |

`dual_path/` and `blocked/` have code. Other family READMEs list today's
files and the target names.
