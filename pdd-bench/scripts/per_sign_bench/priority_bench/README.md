# priority_bench — unified 2.1 (main road) + 2.4 (yield)

Shared junction-priority evaluation bench for real OpenStreetMap / SUMO scenes.
Sign-specific behavior lives in `signs/` profiles; shared engine in `core/`.

## Setup

```bash
conda activate zinkovich-plant2
cd traffic-rule-bench/pdd-bench/scripts/per_sign_bench/priority_bench
```

## Folder structure

```
priority_bench/
├── core/                 # shared libs (layout, crop, aux, augmentation, viability, …)
├── signs/                # SignProfile registry (main_road=2.1, yield=2.4)
├── configs/              # Hydra (configs/sign/{main_road,yield}.yaml)
├── data/
│   ├── main_road/{scenes,output}   # symlinks → former main_sign trees
│   └── yield/{scenes,output}       # symlinks → former yield_sign trees
├── tools/
│   ├── filter_scenes/    # catalog import → crop → review pool
│   ├── build_scene.py    # OSM → SUMO network
│   ├── run_simulation.py
│   ├── render_map.py
│   └── review_benchmark_gifs.py
├── generate_manifest.py
├── run_benchmark.py
└── eval_pipeline.py
```

Compatibility shims remain under `main_sign/` and `yield_sign/`.

Trajectory collection (oracle / PlanT2) still lives next to the original benches:
- [`../yield_sign/collect_trajectories/`](../yield_sign/collect_trajectories/README.md) (2.4)
- [`../main_sign/collect_trajectories/`](../main_sign/collect_trajectories/README.md) (2.1)

## Sign rules

### 2.1 — main road / equal priority (`sign=main_road`)

All incoming roads carry **MainRoadSign**. The plate itself is informational.

Conflict resolution uses the **right-hand rule**: traffic from the right has priority.
Violations are tracked by an invisible `RightHandYieldSign` on the ego approach
(same zone logic as yield 2.4, but watching the **right** conflicting arm only).

Auxiliary agents spawn **only on the right incoming arm** relative to ego.

### 2.4 — yield (`sign=yield`)

Ego is on a **secondary** approach with **YieldSign**; main-road arms get
**MainRoadSign**. Ego must not leave the yield zone while main-road traffic is
present. Rule-based experts stop / creep to a stop line about **5 m before**
the junction end.

Auxiliary agents spawn on **main-road** incoming lanes (gated IDM: released when
ego is near its spawn-lane end so both meet at the junction).

## Workflow

### Step 1A: Build scene pool (catalog → crop → review)

Full sequence is documented in
[`tools/filter_scenes/README.md`](tools/filter_scenes/README.md). Short version:

```bash
# Import qualifying catalog maps into scenes/core/
python tools/filter_scenes/import_catalog_scenes.py --limit 30

# Crop until enough manifest-viable junction scenes exist
python tools/filter_scenes/build_scene_pool.py crop --target 100

# Review keep/reject in browser
python tools/filter_scenes/review_junction_scenes.py

# Progress / drop analysis
python tools/filter_scenes/build_scene_pool.py status --target 100
python tools/analyze_manifest_drops.py
```

Cropping runs the same viability checks as `generate_manifest.py` (junction
shape must be **T** or **X** — two-arm stubs are rejected; layout, aux lane
length for at least one lead vehicle, routable ego/aux).
Invalid junctions are skipped before review. Disable with
`--no-require-manifest-viable` if needed.

### Augmentation axes

Axes are declared in `configs/sign/*.yaml` under `augmentation:` (defaults off
in the root config). Priority signs enable both:

| Axis | Meaning |
|------|---------|
| `layout` | Ego × aux arm/lane/destination scenarios (`core/scene_augmentation.py`). Ego dest + aux dest follow the T/X conflict table (aux never on the opposite arm; ego-right requires aux on the left; aux may turn left/right/straight per case, not only straight-through) |
| `auxiliary` | Cartesian product of convoy `1..N` and occupied lanes `1..M` |

Expansion (product, short-road skip, geometry dedupe, cap) lives in
[`core/manifest_expansion.py`](core/manifest_expansion.py); `generate_manifest.py`
only discovers scenes and writes the jsonl.

A future sign without crossing traffic (e.g. 5.19) would set
`augmentation.auxiliary: false` and omit / disable the `auxiliary:` block.

When expanding convoy sizes, rows whose aux approach is too short for the full
convoy (`length < distance + (N-1)*gap + 3m`) are dropped so a truncated spawn
does not duplicate a smaller-convoy scenario. Rows that differ only by which
aux lane is listed first but occupy the same set of lanes (common with
`lanes_occupied >=` number of main arms) are also deduplicated.

After review, apply rejects and generate a manifest. If `scene_selection.json`
still lists `reject` for scenes that remain on disk, `generate_manifest.py`
exits with an error until you run:

```bash
python tools/filter_scenes/review_junction_scenes.py --apply
```

### Step 1B: Build a single scene from OSM

Each scene folder under `data/<sign>/scenes/` (or the legacy `scenes/` trees)
must contain:

- `map.osm` — OpenStreetMap extract
- `center.json` — crop center: `{"lat": ..., "lon": ...}`

```bash
python tools/build_scene.py <scene_name> --radius <meters>
# example:
python tools/build_scene.py savvinskaya_3 --radius 100
```

Creates `map.net.xml`, `cropped.osm`, and `meta.json`.

### Step 2: Generate evaluation manifest

```bash
# Equal-priority / main road (2.1)
python generate_manifest.py sign=main_road

# Yield (2.4)
python generate_manifest.py sign=yield

# Common overrides
python generate_manifest.py sign=yield gif.enabled=true gif.policy=comprehensive_rule_expert
python generate_manifest.py sign=yield auxiliary.lanes_occupied=2 auxiliary.convoy_size=2
```

Output lands under `data/<sign>/output/<timestamp>/`:

- `real_manifest.jsonl` — scenario definitions
- `config.yaml` — resolved Hydra config
- `gifs/` — if `gif.enabled=true`

### Step 3: Run policy evaluation

```bash
python eval_pipeline.py \
    --policies idm \
    --manifest data/yield/output/<timestamp> \
    --scenes-root data/yield/scenes
```

The manifest row already carries `pdd_code` / `sign_type`.

## Debug tools

```bash
python -m tools.run_simulation <scene_name>
python -m tools.run_simulation <scene_name> --policy carl
python -m tools.run_simulation <scene_name> --policy plant2 --max-steps 400

python -m tools.render_map <scene_name>
python -m tools.render_map <scene_name> --out custom_output.png

# Review GIFs after a run
python tools/review_benchmark_gifs.py data/yield/output/<timestamp>
```

## Trajectory collection + oracle (aux agents)

To collect expert trajectories the same way as the general bench
(`collect_trajectories.sh` → oracle selection), with priority auxiliary agents:

- Yield (2.4): see [`../yield_sign/collect_trajectories/README.md`](../yield_sign/collect_trajectories/README.md)
- Main road (2.1): see [`../main_sign/collect_trajectories/README.md`](../main_sign/collect_trajectories/README.md)

Quick visual smoke test (yield):

```bash
cd ../yield_sign/collect_trajectories
SMOKE=1 ./collect_trajectories.sh
# GIFs under output/trajectories_*/comprehensive_rule_expert/2_4/gifs/
```

## Configuration

See `configs/config.yaml` and `configs/sign/{main_road,yield}.yaml`.

| Group | Key examples |
|-------|----------------|
| `paths.*` | `scenes_dir`, `output_base` (`data/<sign>/output`), `experiment_name` |
| `scenario.*` | `max_scenarios` (cap **after** all axes) |
| `augmentation.*` | `enabled`, `layout`, `auxiliary` — which axes run (per sign yaml) |
| `simulation.*` | `spawn_velocity_ms`, `horizon`, `spawn_distance_before_end` (default **15 m**) |
| `auxiliary.*` | Params when `augmentation.auxiliary` is on: `enabled`, `distance_from_intersection`, `convoy_size`, `lanes_occupied`, `convoy_gap_m` (scalar or list, e.g. `[5, 10]`), `release_when_ego_within_m` |
| `gif.*` | `enabled`, `policy`, `scaling` (px/m; higher = more zoomed in), `hide_signs` |

Override on the CLI:

```bash
python generate_manifest.py sign=yield simulation.horizon=800 auxiliary.convoy_size=3
python generate_manifest.py sign=yield augmentation.auxiliary=false  # layout only
python generate_manifest.py sign=yield scenario.max_scenarios=5
```

Notes on timing / caps:

- `simulation.spawn_distance_before_end` — where **ego** is placed on its approach
  (meters before lane end / junction). Default 15 m.
- `auxiliary.release_when_ego_within_m` — when **gated aux** starts: ego's remaining
  distance to lane end ≤ this value. Keep ≥ spawn distance so aux is not held while
  a yielding ego waits outside the release radius.
- `scenario.max_scenarios` — after layout × convoy × lanes × gaps, short-road
  skips, and geometry dedupe, **shuffle** then keep at most this many rows
  **per scene** (default 15).