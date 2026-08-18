# priority_bench

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
├── signs/                # SignProfile registry (main=2.1 … 4.1.x direction, 5.7 one_way)
├── configs/              # Hydra (configs/sign/{main,secondary,yield,stop,roundabout}.yaml)
├── data/
│   ├── main_road/{scenes,output,trajectories}
│   ├── secondary_road/{scenes,output,trajectories}
│   ├── yield/{scenes,output,trajectories}
│   ├── stop/{scenes,output,trajectories}
│   └── roundabout/{scenes,output,trajectories}
├── build_scenes/         # materialize moscow allocations → review pool
│   ├── materialize_scenes.py
│   ├── review_scenes.py
│   └── legacy/           # old catalog / Overpass flow
├── tools/                # ad-hoc debug (GIF review, map render, drop analysis, …)
├── collect_trajectories/ # oracle / PlanT2 expert collection (SIGN=…|secondary)
├── generate_manifest.py
├── run_benchmark.py
└── eval_pipeline.py
```

Compatibility shims remain under `main_sign/` and `yield_sign/`.

Trajectory collection (oracle / PlanT2):
`[collect_trajectories/](collect_trajectories/README.md)` — set `SIGN=yield|main|stop|secondary|roundabout`.
Old paths under `yield_sign/` / `main_sign/` forward here.

## Sign rules

### 2.1 — main road / equal priority (`sign=main`)

All incoming roads carry **MainRoadSign**. The plate itself is informational.

Conflict resolution uses the **right-hand rule**: traffic from the right has priority.
Violations are tracked by an invisible `RightHandYieldSign` on the ego approach
(same zone logic as yield 2.4, but watching the **right** conflicting arm only).

Auxiliary agents spawn **only on the right incoming arm** relative to ego.

### 2.3 — secondary road (`sign=secondary`)

Unified family for plates **2.3.1 / 2.3.2 / 2.3.3**. Same junction geometry / ego /
aux axes as yield (2.4): ego on a **secondary** approach with **YieldSign**; aux on
**main**. Allocation is one key `"2.3"` with balanced T/X (`x_share: 0.5`).

Plate placement on **main** arms:

| Shape | Plates |
|-------|--------|
| **X** | **2.3.1** (`SecondaryRoadSign`) on every main approach |
| **T** | **2.3.2** (`SecondaryRoadRightSign`) and **2.3.3** (`SecondaryRoadLeftSign`) on the two main approaches (stem on the right / left of that approach) |

Secondary approaches always get **YieldSign** (2.4). Metrics / expert yield logic
come from that YieldSign; 2.3 plates mark main-road priority (expert treats them
like MainRoadSign when ego is on that arm).

### 2.4 — yield (`sign=yield`)

Ego is on a **secondary** approach with **YieldSign**; main-road arms get
**MainRoadSign**. Ego must not leave the yield zone while main-road traffic is
present. Rule-based experts stop / creep to a stop line about **5 m before**
the junction end.

Auxiliary agents spawn on **main-road** incoming lanes (gated IDM: released when
ego is near its spawn-lane end so both meet at the junction).

### 2.5 — stop (`sign=stop`)

Same junction geometry / ego / aux axes as yield (2.4). Secondary **ego** arm gets
**StopSign** (2.5). On **X** junctions the opposite secondary arm shows a **YieldSign**
(2.4) plate — priority logic is unchanged (main vs secondary); only that plate differs.
**T** junctions keep a single StopSign on the secondary stem. Main-road arms get
**MainRoadSign**. Ego must yield to main traffic **and** make a mandatory full stop
before the stop line (`StopSign` in `traffic_signs/priority_signs.py`).

### 4.3 — roundabout (`sign=roundabout`)

Ego spawns on a **spoke** (secondary). Ring edges are **main**. Visible
**RoundaboutSign** on the ego spoke; invisible **RoundaboutYieldSign** tracks
violations against the conflict-arc ring (20 m ENTRY_CONFLICT). Aux agents spawn
on the **left** ring segment at ego's entry (upstream extension only when that
segment is short) and share **ego's destination exit**.
Scenes: moscow `scenes/O/` → `data/roundabout/scenes`.

## Workflow

### Step 1A: Build scene pool (moscow → materialize → review)

Full sequence:
`[build_scenes/README.md](build_scenes/README.md)`. Short version for **2.5**:

```bash
# Link allocated moscow junctions into data/stop/scenes/
python build_scenes/materialize_scenes.py --sign 2.5

# Review keep/reject in browser
python build_scenes/review_scenes.py --scenes-dir data/stop/scenes

# Apply rejects, then top up to signs.yaml quotas if short
python build_scenes/review_scenes.py --scenes-dir data/stop/scenes --apply
python build_scenes/materialize_scenes.py --sign 2.5 --refill
```

Prereq: shared harvest + allocations under  
`[../moscow_scenes/](../moscow_scenes/README.md)`.  
Quotas (`n_train` / `n_test`) live in  
`[../moscow_scenes/splits/signs.yaml](../moscow_scenes/splits/signs.yaml)`.  
Old catalog/Overpass scripts live in `build_scenes/legacy/` (do not use for new pools).

### Step 2: Generate evaluation manifest

#### Augmentation axes

Axes are declared in `configs/sign/*.yaml` under `augmentation:` (defaults off in the root config). Priority signs enable both:


| Axis        | Meaning                                                                                                                                                                                                                                                              |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layout`    | Ego × aux arm/lane/destination scenarios (`core/scene_augmentation.py`). Ego dest + aux dest follow the T/X conflict table (aux never on the opposite arm; ego-right requires aux on the left; aux may turn left/right/straight per case, not only straight-through) |
| `auxiliary` | Cartesian product of convoy `1..N` and occupied lanes `1..M`                                                                                                                                                                                                         |


```bash
# Equal-priority / main road (2.1)
python generate_manifest.py sign=main

# Yield (2.4) — all / train / test (filter via moscow_pool.json)
python generate_manifest.py sign=yield
python generate_manifest.py sign=yield paths.split=train
python generate_manifest.py sign=yield paths.split=test

# Secondary road (2.3) — X→2.3.1, T→2.3.2+2.3.3
python generate_manifest.py sign=secondary
python generate_manifest.py sign=secondary paths.split=train

# Stop (2.5)
python generate_manifest.py sign=stop
python generate_manifest.py sign=stop paths.split=train

# Roundabout (4.3)
python generate_manifest.py sign=roundabout
python generate_manifest.py sign=roundabout paths.split=train

# Common overrides
python generate_manifest.py sign=stop gif.enabled=true gif.policy=comprehensive_rule_expert
python generate_manifest.py sign=stop gif.enabled=true gif.policy=plant2_ft
# Finetuned PlanT2: newest *.ckpt under pdd-bench/checkpoints/plant2_finetuned/
python generate_manifest.py sign=stop auxiliary.lanes_occupied=2 auxiliary.convoy_size=2
# Stop dwell at the line (sign=stop only): default 15 steps (~1.5 s); was 30 (~3 s)
python generate_manifest.py sign=stop expert.stop_wait_steps=15
# Debug: shuffle all augmented rows and keep only N total
python generate_manifest.py sign=stop scenario.max_total=20
```

Output lands under `data/<sign>/output/<timestamp>/`:

- `real_manifest.jsonl` — scenario definitions (each row has `split`)
- `repro/` — pool snapshot + split filter + allocations ref (for reproduction)
- `config.yaml` — resolved Hydra config
- `gifs/` — if `gif.enabled=true`

### Step 3: Run policy evaluation

```bash
python eval_pipeline.py \
    --policies idm,comprehensive_rule_expert,plant2,plant2_ft,plant2_rule,carl,carl_rule,ppo_lidar,rule_compliant \
    --manifest data/stop/output/<timestamp> \
    --scenes-root data/stop/scenes
```

The manifest row already carries `pdd_code` / `sign_type`.

Existing test split (do not regenerate the manifest):

```bash
python eval_pipeline.py \
    --policies plant2_ft \
    --manifest data/stop/output/ts_test \
    --scenes-root data/stop/scenes \
    --out-dir data/stop/output/ts_test/eval_out_plant2_ft
```

`plant2_ft` loads the newest `*.ckpt` in `pdd-bench/checkpoints/plant2_finetuned/` (override with `--model-paths plant2_ft:/path/to.ckpt`).

## Debug tools

```bash
python -m tools.run_simulation <scene_name>
python -m tools.run_simulation <scene_name> --policy carl
python -m tools.run_simulation <scene_name> --policy plant2 --max-steps 400

# Review GIFs after a run
python tools/review_benchmark_gifs.py data/stop/output/<timestamp>
```

## Trajectory collection + oracle (aux agents)

See `[collect_trajectories/](collect_trajectories/README.md)` — set `SIGN=yield|main|stop|secondary|roundabout`.

Quick visual smoke test (stop):

```bash
cd collect_trajectories
SIGN=stop SMOKE=1 ./collect_trajectories.sh
# GIFs under data/stop/trajectories/trajectories_*/comprehensive_rule_expert/2_5/gifs/
```

## Configuration

See `configs/config.yaml` and `configs/sign/{main,secondary,yield,stop,roundabout}.yaml`.


| Group            | Key examples                                                                                                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `paths.*`        | `scenes_dir`, `output_base` (`data/<sign>/output`), `experiment_name`                                                                                                                              |
| `scenario.*`     | `max_scenarios` (cap **after** all axes)                                                                                                                                                           |
| `augmentation.`* | `enabled`, `layout`, `auxiliary` — which axes run (per sign yaml)                                                                                                                                  |
| `simulation.`*   | `spawn_velocity_ms`, `horizon`, `spawn_distance_before_end` (default **15 m**)                                                                                                                     |
| `auxiliary.`*    | Params when `augmentation.auxiliary` is on: `enabled`, `distance_from_intersection`, `convoy_size`, `lanes_occupied`, `convoy_gap_m` (scalar or list, e.g. `[5, 10]`), `release_when_ego_within_m` |
| `gif.`*          | `enabled`, `policy`, `scaling` (px/m; higher = more zoomed in), `hide_signs`, `draw_path_conflict` (ego/aux routes + conflict X)                                                                   |


Override on the CLI:

```bash
python generate_manifest.py sign=stop simulation.horizon=800 auxiliary.convoy_size=3
python generate_manifest.py sign=stop augmentation.auxiliary=false  # layout only
python generate_manifest.py sign=stop scenario.max_scenarios=5
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
