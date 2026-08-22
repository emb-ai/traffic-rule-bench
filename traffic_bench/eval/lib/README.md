# lib/ — shared eval engine (no sign rules)

SUMO parse, IDM profiles, MetaDrive patches, HUD/GIF overlays, T/X arm
geometry. Nothing here cares which plate is under test.

Imports: `traffic_bench.eval.lib.*`. Old `traffic_bench.eval.core.*` paths
are shims to these modules.

| Path | Stores |
| --- | --- |
| `lib/sumo/` | Parse `map.net.xml`, lane keys, `meta.json` / net path |
| `lib/profiles/` | nuPlan IDM samples, ego s1–s4 defaults, stable hashes |
| `lib/runtime/` | MetaDrive SUMO patches, sign-zone / crash helpers, checkpoint paths |
| `lib/patches/` | HUD / GIF overlays, RecordManager |
| `lib/layout/` | T/X arm classification, sign offsets, roundabout topology |
| `lib/manifest/` | Shared defaults, expansion axes, viability filters |
| `lib/scenarios/` | Shared `SpawnScenario` / lane-parse helpers + runtime aux |

Family enumerators live in [`../signs/`](../signs/README.md), not here.
`lib/scenarios/scene_augmentation.py` only keeps the shared types and
dispatches to `signs/<family>/spawn.py`.

`lib/layout/junction_crop.py` is harvest leftover. Crop already lives in
[`scene_collection/collect`](../../scene_collection/collect/).
