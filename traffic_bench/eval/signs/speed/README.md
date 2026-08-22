# signs/speed — speed limit, min speed, zone plates (3.24 / 4.6 / 5.21 / 5.31)

Straight segment corridor. Spawn at the start of `meta.road_id`; sign offset
is chosen in expansion (braking runway on the same edge).

| Eval id | Sign code |
| --- | --- |
| `speed_limit` | 3.24 |
| `min_speed` | 4.6 |
| `residential_zone` | 5.21 |
| `zone_speed_limit` | 5.31 |

## Today → tomorrow

| Today | Tomorrow |
| --- | --- |
| `core/manifest/speed_expansion.py` | `expand.py` |
| `core/scenarios/speed_scene_design.py` | `spec.py` |
| `run_benchmark.py` speed-zone placement | `place.py` |
