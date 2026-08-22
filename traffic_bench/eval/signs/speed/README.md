# signs/speed — speed limit, min speed, zone plates (3.24 / 4.6 / 5.21 / 5.31)

Straight segment corridor. Spawn at the start of `meta.road_id`; sign offset
is chosen in expansion (braking runway on the same edge).

| Eval id | Sign code |
| --- | --- |
| `speed_limit` | 3.24 |
| `min_speed` | 4.6 |
| `residential_zone` | 5.21 |
| `zone_speed_limit` | 5.31 |

| File | Role |
| --- | --- |
| `spec.py` | Limit assignment, braking/accel approach formulas |
| `expand.py` | lane × density → manifest rows |
| `place.py` | Start plate + paired end plate |

`generate_speed_manifest` still lives in `generate_manifest.py`.
`core.manifest.speed_expansion` / `core.scenarios.speed_scene_design` are shims.
