# Stop Sign (2.5) Benchmark

Benchmark for evaluating autonomous driving policies on stop sign scenarios using real-world OpenStreetMap data.

## Setup

```bash
conda activate zinkovich-plant2
```

## Folder Structure

```
stop_sign/
├── build_scene.py          # Step 1: OSM → SUMO network
├── generate_manifest.py    # Step 2: Create evaluation manifest
├── eval_pipeline.py        # Step 3: Run policy evaluation
├── run_benchmark.py        # Benchmark runner (used internally)
├── lib/                    # Core library modules
│   ├── auxiliary_agent.py
│   ├── junction_priority_layout.py
│   ├── manifest_config.py
│   ├── scene_augmentation.py
│   └── sumo_utils.py
├── tools/                  # Debug/visualization utilities
│   ├── run_simulation.py   # Test scene with policy
│   └── render_map.py       # Render static map image
├── config/                 # Hydra configuration
│   └── config.yaml
├── scenes/                 # Scene data (OSM + SUMO networks)
└── benchmark_output/       # Evaluation results
```

## Workflow

### Step 1: Build Scene from OSM

Each scene folder in `scenes/` must contain:
- `map.osm` — OpenStreetMap extract
- `center.json` — Crop center: `{"lat": ..., "lon": ...}`; optional `"save_service_roads": true` keeps `highway=service` ways connected to main roads

Convert OSM to SUMO network:

```bash
python build_scene.py <scene_name> --radius <meters>
```

### Step 2: Generate Evaluation Manifest

```bash
python generate_manifest.py
```

Output is saved to `benchmark_output/2_5/<timestamp>/`.

### Step 3: Run Policy Evaluation

```bash
python eval_pipeline.py \
    --policies idm \
    --manifest benchmark_output/2_5/<timestamp> \
    --scenes-root scenes
```

## Debug Tools

```bash
python -m tools.run_simulation <scene_name>
python -m tools.render_map <scene_name>
```

## Sign rule (2.5)

Stop sign compliance matches yield (2.4) junction logic — do not proceed through the approach zone while main-road traffic is present — **plus** a mandatory complete stop before the sign line.
