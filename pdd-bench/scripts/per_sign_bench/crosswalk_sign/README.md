# Pedestrian Crossing Sign (5.19) Benchmark

Benchmark for evaluating autonomous driving policies at pedestrian crossings (PDD 5.19).
Uses `CrosswalkPedestrianManager` for synthetic pedestrian tracks and `PedestrianYieldRule` for verification.

## Setup

```bash
conda activate zinkovich-plant2
```

## Folder Structure

```
crosswalk_sign/
├── generate_manifest.py    # Create evaluation manifest from scenes with SUMO crossings
├── eval_pipeline.py        # Run policy evaluation
├── run_benchmark.py        # Benchmark runner (used internally)
├── lib/
│   ├── crosswalk_layout.py # Parse SUMO crossing edges and approach lanes
│   └── ...
├── tools/filter_scenes/    # Catalog import and scene review
├── config/                 # Hydra configuration
├── scenes/                 # Scene data (SUMO networks with crossings)
└── benchmark_output/       # Evaluation results
```

## Scene requirements

Each scene must contain a SUMO `.net.xml` with at least one `function="crossing"` edge.
Ego spawns on a vehicle lane approaching the crosswalk; pedestrians are spawned by `CrosswalkPedestrianManager`.

## Catalog import

Import scenes from the 5.19 catalog into `scenes/core/` (filters nets that contain crossings):

```bash
python tools/filter_scenes/import_catalog_scenes.py --limit 30
```

Source catalog (default): `pdd-bench/scenes/5.19`.

## Scene pool workflow

```bash
python tools/filter_scenes/import_catalog_scenes.py --limit 30
python tools/filter_scenes/build_scene_pool.py crop --target 100
python tools/filter_scenes/review_junction_scenes.py
python tools/filter_scenes/build_scene_pool.py fill --target 100
python generate_manifest.py
python eval_pipeline.py --policies idm --manifest benchmark_output/5_19/<timestamp> --scenes-root scenes
```

Output: `benchmark_output/5_19/<timestamp>/`.

## Verification

- **Rule**: `PedestrianYieldRule` in `pdd-bench/traffic_signs/pedestrian_yield_rule.py`
- **Spawn**: `CrosswalkPedestrianManager` in `pdd-bench/envs/pedestrian_manager.py`
- **Visualization**: zebra crosswalk polygons in MetaDrive top-down renderer

Violations are counted in the `crosswalk_violations` metric bucket.
