# signs/crosswalk — pedestrian crossing (5.19)

Segment maps. Zebra is injected at materialize
(`scene_collection` prepare), then eval parses the SUMO crossing and
spawns on the approach to it.

| Eval id | Sign code |
| --- | --- |
| `crosswalk` | 5.19 |

| File | Role |
| --- | --- |
| `spec.py` | Parse SUMO crossing approaches |
| `spawn.py` | Pedestrian preset bank |
| `expand.py` | lane × density × preset → manifest rows |
| `place.py` | Reconstruct zebra + place 5.19 plate |

`generate_crosswalk_manifest` still lives in `generate_manifest.py`.
Old `core.layout.crosswalk_layout` / `core.manifest.crosswalk_expansion`
/ `core.scenarios.pedestrian_presets` imports are shims.
