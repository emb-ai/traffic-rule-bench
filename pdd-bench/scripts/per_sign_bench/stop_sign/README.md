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
├── run_benchmark_real.py   # Benchmark runner (used internally)
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
- `center.json` — Crop center: `{"lat": ..., "lon": ...}`

Convert OSM to SUMO network:

```bash
python build_scene.py <scene_name> --radius <meters>
```

Example:
```bash
python build_scene.py savvinskaya_3 --radius 100
```

This creates:
- `map.net.xml` — SUMO network
- `cropped.osm` — Cropped OSM (for reference)
- `meta.json` — Scene metadata

### Step 2: Generate Evaluation Manifest

Generate scenarios with ego/auxiliary agent configurations:

```bash
python generate_manifest.py
```

With options (Hydra config):
```bash
python generate_manifest.py gif.enabled=true
python generate_manifest.py auxiliary.lanes_occupied=2 auxiliary.convoy_size=2
```

Output is saved to `benchmark_output/2_5/<timestamp>/`:
- `real_manifest.jsonl` — Scenario definitions
- `config.yaml` — Resolved configuration
- `gifs/` — Visualization GIFs (if enabled)

### Step 3: Run Policy Evaluation

Evaluate policies on the generated manifest:

```bash
python eval_pipeline.py \
    --policies idm \
    --manifest benchmark_output/2_5/<timestamp> \
    --scenes-root scenes
```

## Debug Tools

### Test Scene with Policy

Run a simulation to verify scene setup:

```bash
python -m tools.run_simulation <scene_name>
python -m tools.run_simulation <scene_name> --policy carl
python -m tools.run_simulation <scene_name> --policy plant2 --max-steps 400
```

### Render Static Map

Generate a static map image:

```bash
python -m tools.render_map <scene_name>
python -m tools.render_map <scene_name> --out custom_output.png
```

## Configuration

See `config/config.yaml` for available options:
- `scenario.*` — Scenario generation settings
- `simulation.*` — Simulation parameters
- `auxiliary.*` — Auxiliary agent configuration
- `gif.*` — GIF rendering options

Override via command line:
```bash
python generate_manifest.py simulation.horizon=800 auxiliary.convoy_size=3
```
