# lib/ — shared eval engine (no sign rules)

Target home for everything that does not care which plate is under test.
Today this code still lives in [`../core/`](../core/README.md). Imports stay
`traffic_bench.eval.core.*` until the move.

| Target | Today | Stores |
| --- | --- | --- |
| `lib/sumo/` | `core/sumo/` | Parse `map.net.xml`, lane keys, `meta.json` / net path |
| `lib/profiles/` | `core/profiles/` | nuPlan IDM samples, ego s1–s4 defaults, stable hashes |
| `lib/runtime/` | `core/runtime/` | MetaDrive SUMO patches, sign-zone / crash helpers, checkpoint paths |
| `lib/patches/` | `core/patches/` | HUD / GIF overlays, RecordManager |
| `lib/layout/` | `core/layout/junction_priority_layout.py`, `junction_sign_placement.py` | T/X arm classification used by several families |

`core/layout/junction_crop.py` is harvest leftover. Crop already lives in
[`scene_collection/collect`](../../scene_collection/collect/). Keep a thin
import here until that leftover is deleted.

Sign-specific modules that currently sit under `core/manifest/` and
`core/scenarios/` (`*_expansion.py`, `*_bridge.py`, `*_sign_spec.py`,
`blocked_road_*`, `roundabout_*`, `scene_augmentation.py` branches) move to
[`../signs/`](../signs/README.md), not here.

`core/manifest/manifest_config.py` and `manifest_viability.py` stay shared
(defaults + pre-manifest filters) → `lib/` or `manifest/` when that package
exists.
