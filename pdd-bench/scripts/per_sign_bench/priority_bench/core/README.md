# `core/` — shared library for priority-junction benches

Python package used by `priority_bench` (manifest generation, simulation,
scene pool) and by `moscow_junctions` crop scripts. Imports always go through
`core.<subpackage>.<module>` — there are no flat modules at the `core/` root.

## Layout

```
core/
├── README.md
├── sumo/           SUMO .net.xml parsing, lane keys, scene file helpers
├── layout/         Junction geometry, sign placement, roundabout topology, crop
├── scenarios/      Ego/aux spawn enumeration and runtime aux agents
├── manifest/       Manifest defaults, expansion axes, viability filters
├── pool/           Moscow scene pool + keep/reject selection state
└── runtime/        MetaDrive patches applied before simulation
```

## Subpackages

### `sumo/`

Low-level SUMO / MetaDrive lane identity.

| Module | Role |
|--------|------|
| `lane_keys.py` | Parse/build `lane_<edge>_<num>` keys (handles `#` in edge ids) |
| `sumo_utils.py` | Drivable-lane filters, route index, `meta.json` / net path helpers |

**Used by:** almost everything that reads a cropped `map.net.xml`.

### `layout/`

Junction structure derived from a cropped net.

| Module | Role |
|--------|------|
| `junction_priority_layout.py` | T/X/O arms, main vs secondary, conflict tables |
| `junction_sign_placement.py` | Longitudinal/lateral offsets for MetaDrive signs |
| `junction_crop.py` | Crop city net to junction bbox (moscow harvest + legacy) |
| `roundabout_topology.py` | Ring + spoke layout for 4.3 |
| `roundabout_yield_zone.py` | Entry conflict arcs / yield zones on the ring |

**Used by:** `generate_manifest`, `run_benchmark`, `moscow_junctions/scripts/crop_scenes.py`.

### `scenarios/`

Which ego/aux spawn + destination combinations exist per scene.

| Module | Role |
|--------|------|
| `scene_augmentation.py` | `SpawnStrategy` enumerators (equal / yield / roundabout / blocked_road) |
| `auxiliary_agent.py` | Aux convoy spawn, placement, release logic at runtime |
| `roundabout_aux.py` | Conflict-arc placement + spillover convoy on the ring |
| `blocked_road_route.py` | Forbidden-lane geometry checks for 3.2 |

**Used by:** manifest expansion, viability reject, `run_benchmark` aux spawn.

### `manifest/`

Turning scenes into `real_manifest.jsonl` rows.

| Module | Role |
|--------|------|
| `manifest_config.py` | Shared defaults (`spawn_distance_before_end`, stop wait, …) |
| `manifest_expansion.py` | Layout × aux cartesian product (2.1 / 2.4 / 4.3) |
| `manifest_viability.py` | Pre-manifest scene filters (`reject_unusable_scenes`) |
| `blocked_road_expansion.py` | Layout × traffic-density tiers for 3.2 |
| `traffic_density_levels.py` | nuPlan p25/p50/p75 → MetaDrive `traffic_density` |

**Used by:** `generate_manifest.py`, `build_scenes/reject_unusable_scenes.py`.

### `pool/`

Moscow allocation bookkeeping under `data/<sign>/scenes/`.

| Module | Role |
|--------|------|
| `moscow_pool.json` helpers in `moscow_pool.py` | Train/test split map, pool snapshot |
| `scene_selection.py` | Visual review keep/reject + `_rejected/` moves |

**Used by:** `materialize_scenes`, `review_scenes`, manifest split filter.

### `runtime/`

| Module | Role |
|--------|------|
| `metadrive_sumo_patch.py` | Strip invalid SUMO via-chains before MetaDrive graph build |

**Used by:** `run_benchmark.py`, `eval_pipeline.py`, `tools/run_simulation.py`.

## Dependency flow (high level)

```mermaid
flowchart TB
  pool --> layout
  sumo --> layout
  sumo --> scenarios
  layout --> scenarios
  scenarios --> manifest
  sumo --> manifest
  layout --> manifest
  manifest --> runtime
```

1. **Pool** picks junction crops (`materialize_scenes`).
2. **Layout + sumo** parse each crop’s net and classify arms.
3. **Scenarios** enumerate valid ego/aux or through-path variants.
4. **Manifest** expands axes, filters viability, writes JSONL.
5. **Runtime** patches MetaDrive; bench reads manifest rows.

## Import examples

```python
# Preferred — explicit subpackage
from core.sumo.lane_keys import make_lane_key
from core.scenarios.scene_augmentation import SpawnStrategy
from core.manifest.manifest_config import DEFAULT_SPAWN_DISTANCE_BEFORE_END

# Convenience re-exports (stable public surface)
from core import SpawnScenario, build_junction_priority_layout
```

## Adding a new sign family

1. Add a `SignProfile` in `signs/base.py` with `spawn_strategy`.
2. If needed, extend `scenarios/scene_augmentation.py` with a new enumerator.
3. Wire expansion in `manifest/` (generic `manifest_expansion.py` or a sign-specific module like `blocked_road_expansion.py`).
4. Add viability rules in `manifest/manifest_viability.py`.
5. Document the sign in this README’s scenario/manifest tables.

Future **3.1 `blocked_entry`** should live alongside `blocked_road_*` under
`scenarios/` and `manifest/` (shared route helpers, separate expansion if rules differ).
