# Secondary Road Sign (2.3) Benchmark

Benchmark for evaluating autonomous driving policies at **T-junctions** with secondary-road warning signs on main approaches and yield signs on the secondary stem. Logic matches the yield sign (2.4) benchmark; only sign naming and junction shape differ.

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
├── config/                 # Hydra configuration
├── scenes/                 # Scene data (T-junction OSM + SUMO networks)
└── benchmark_output/       # Evaluation results
```

## Workflow

### Step 1: Build Scene from OSM

Each scene must be a **T junction** (3 incoming arms). Scene folder needs `map.osm`, `center.json`, and after build `map.net.xml`.

```bash
python build_scene.py <scene_name> --radius <meters>
```

### Step 2: Generate Evaluation Manifest

```bash
python generate_manifest.py
```

Non-T scenes are rejected with an assertion. Output: `benchmark_output/2_3/<timestamp>/`.

### Step 3: Run Policy Evaluation

```bash
python eval_pipeline.py \
    --policies idm \
    --manifest benchmark_output/2_3/<timestamp> \
    --scenes-root scenes
```

## Sign rule (2.3)

Same priority logic as yield (2.4):

- **Main arms**: informational secondary-road signs — **2.3.1** (`SecondaryRoadSign`) when the stem is on the left, **2.3.2** (`SecondaryRoadRightSign`) when on the right.
- **Secondary arm**: **YieldSign** (2.4) — must not leave the approach zone while main-road traffic is present.
- **Ego** spawns on the secondary arm; **auxiliary agents** spawn on main-road lanes.

Only **T-shaped** junctions are supported (`assert shape == "T"` in layout build and manifest generation).
