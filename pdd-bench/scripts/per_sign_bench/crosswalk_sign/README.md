# Secondary Road Sign (2.3) Benchmark

Benchmark for evaluating autonomous driving policies at junctions with secondary-road warning signs (2.3.x) on main approaches and yield signs (2.4) on the secondary stem.

## Setup

```bash
conda activate zinkovich-plant2
```

## Folder Structure

```
secondary_sign/
├── build_scene.py          # Step 1: OSM → SUMO network
├── generate_manifest.py    # Step 2: Create evaluation manifest
├── eval_pipeline.py        # Step 3: Run policy evaluation
├── run_benchmark.py        # Benchmark runner (used internally)
├── lib/                    # Core library modules
├── tools/filter_scenes/    # Catalog import, crop, review
├── config/                 # Hydra configuration
├── scenes/                 # Scene data (OSM + SUMO networks)
└── benchmark_output/       # Evaluation results
```

## Sign placement (GIF / simulation)

| Junction | Main-road arms | Secondary arm |
|----------|----------------|---------------|
| **X** (4-arm) | **2.3.1** (`SecondaryRoadSign`) | **2.4** yield |
| **T** (3-arm) | **2.3.2** left (`SecondaryRoadLeftSign`), **2.3.3** right (`SecondaryRoadRightSign`) | **2.4** yield |

Ego spawns on the secondary arm; auxiliary agents occupy main-road lanes.

## Catalog import (equal split)

Import from three catalogs with round-robin selection:

```bash
python tools/filter_scenes/import_catalog_scenes.py --limit 30
```

Sources (default): `pdd-bench/scenes/2.3.1`, `scenes/2.3.2`, `scenes/2.3.3`. Takes equally from each; when one folder is exhausted, continues with the rest.

## Workflow

```bash
python tools/filter_scenes/import_catalog_scenes.py --limit 30
python tools/filter_scenes/build_scene_pool.py crop --target 100
python generate_manifest.py
python eval_pipeline.py --policies idm --manifest benchmark_output/2_3/<timestamp> --scenes-root scenes
```

Output: `benchmark_output/2_3/<timestamp>/`.
