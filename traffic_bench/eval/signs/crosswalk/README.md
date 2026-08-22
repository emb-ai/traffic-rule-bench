# signs/crosswalk — pedestrian crossing (5.19)

Segment maps. Zebra is injected at materialize
(`scene_collection` prepare), then eval parses the SUMO crossing and
spawns on the approach to it.

| Eval id | Sign code |
| --- | --- |
| `crosswalk` | 5.19 |

## Today → tomorrow

| Today | Tomorrow |
| --- | --- |
| `core/manifest/crosswalk_expansion.py` | `expand.py` |
| `core/layout/crosswalk_layout.py` | `spec.py` |
| `core/scenarios/pedestrian_presets.py` | `spawn.py` |
| `run_benchmark.py` crosswalk placement | `place.py` |
