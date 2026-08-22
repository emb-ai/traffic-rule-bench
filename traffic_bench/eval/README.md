# eval — manifests, closed-loop runs, metrics

Run from the **repository root** after `pip install -e .`.

Scenes come from [`scene_collection/`](../scene_collection/README.md) or
Hugging Face [`emb-ai/traffic-sign-bench`](https://huggingface.co/datasets/emb-ai/traffic-sign-bench)
into `data/scenes/<sign>/`. Eval writes `data/runs/<sign>/<timestamp>/`.
`sign=main_road` writes under `data/{scenes,runs}/main_road/`.

Family expand/place/spec live under [`signs/`](signs/README.md). Shared
shells are still `generate_manifest.py`, `run_benchmark.py`,
`eval_pipeline.py`, and `core/`. Old commands
(`python -m traffic_bench.eval.generate_manifest`) keep working.

## Folders

| Path | Role |
| --- | --- |
| [`cli.py`](cli.py) | `python -m traffic_bench.eval {manifest,run,pipeline,metrics}` |
| [`sign_registry.py`](sign_registry.py) | eval id → family, spawn, data folder |
| [`configs/`](configs/) | Hydra YAML; paths are relative (`data/scenes/${sign}`) |
| [`manifest/`](manifest/) | *(target)* scenes + config → `real_manifest.jsonl` |
| [`bench/`](bench/) | *(target)* one policy, one manifest, closed-loop episodes |
| [`pipeline/`](pipeline/) | *(target)* many policies + `eval_out/` |
| [`metrics/`](metrics/README.md) | episode JSONL → CSV / markdown |
| [`lib/`](lib/README.md) | shared engine (today: `core/`) |
| [`signs/`](signs/README.md) | per-family expand / spawn / place |

### What each folder stores

**`manifest/`** — "scenes + Hydra → `real_manifest.jsonl`". Shared shell only:
discover scene dirs, write `repro/`, Hydra `main`, dispatch to
`signs/<family>/expand.py`. Not how yield vs crosswalk rows are built.
Today this is [`generate_manifest.py`](generate_manifest.py).

**`bench/`** — today's [`run_benchmark.py`](run_benchmark.py) minus per-sign
placement. One policy, one manifest:

- wrap the MetaDrive / SUMO env
- load IDM / PPO / CaRL / PlanT2
- apply a manifest row (spawn lane, destination, aux convoy)
- call `signs/<family>/place.py` to put plates in the world
- step the episode; record violations, crash, route completion
- optional top-down GIF
- write `episodes_<policy>.jsonl` and `replay.json`

Not multi-policy orchestration (`pipeline/`) and not CSV rollups (`metrics/`).

**`pipeline/`** — today's [`eval_pipeline.py`](eval_pipeline.py): loop policies
(IDM × ego variants s1–s4, then NN), invoke bench, collect `eval_out/`,
trigger `metrics/`.

**`metrics/`** — already here: episode JSONL → per-episode CSV, aggregations,
cumulative markdown.

**`lib/`** — no sign rules. SUMO parse, IDM profiles, MetaDrive patches,
HUD/GIF overlays, T/X arm geometry. Today this is [`core/`](core/README.md).

**`signs/<family>/`** — only what is true for that family. Same file roles
(skip if unused): `expand.py` (manifest rows), `spawn.py` (legal ego/aux
combinations), `place.py` (where plates go in MetaDrive), `spec.py` (plate
class / dual-path crop-meta bridge).

**`configs/`** — Hydra tree. Related signs share a folder
(`direction/__base__.yaml` + `direction/right.yaml`, …). CLI is
`sign=direction/right`. Paths are repo-relative: `data/scenes/${sign}`.
`sign=main_road` / `sign=secondary` write under `main_road` / `secondary_road`.

## Commands

```bash
python -m traffic_bench.eval manifest sign=yield
python -m traffic_bench.eval manifest sign=direction/right
python -m traffic_bench.eval manifest sign=yield paths.split=train
python -m traffic_bench.eval run --policy idm --manifest data/runs/yield/<ts>/real_manifest.jsonl
python -m traffic_bench.eval pipeline --policies idm --manifest data/runs/yield/<ts>
python -m traffic_bench.eval metrics csv --episodes-root <eval_out>/benchmark --out <eval_out>/metrics_per_episode.csv
python -m traffic_bench.eval metrics aggregate --csv <eval_out>/metrics_per_episode.csv --out-dir <eval_out>
python -m traffic_bench.eval metrics report --run-root <eval_out>
```

`run` defaults `--run-name` to `--policy` if you omit it. `pipeline --manifest`
accepts a run folder (reads `real_manifest.jsonl`, writes `<folder>/eval_out/`)
or a `.jsonl` file.

Hydra overrides still work on `manifest`:

```bash
python -m traffic_bench.eval manifest sign=stop gif.enabled=true gif.policy=idm
python -m traffic_bench.eval manifest sign=stop auxiliary.lanes_occupied=2 auxiliary.convoy_size=2
python -m traffic_bench.eval manifest sign=stop scenario.max_total=20
python -m traffic_bench.eval manifest sign=stop augmentation.auxiliary=false
```

## Workflow

### 1. Scene pool

```bash
python -m traffic_bench.scene_collection materialize --sign stop
# or download official scenes:
huggingface-cli download emb-ai/traffic-sign-bench --repo-type dataset --local-dir data
```

### 2. Manifest

Output under `data/runs/<sign>/<timestamp>/` (`sign=main_road` → `data/runs/main_road/`):

- `real_manifest.jsonl` — scenario rows (each has `split`)
- `repro/` — pool snapshot + split filter + allocations ref
- `config.yaml` — resolved Hydra config
- `gifs/` — if `gif.enabled=true`

Augmentation axes live in `configs/sign/*.yaml` under `augmentation:`:

| Axis | Meaning |
| --- | --- |
| `layout` | Ego × aux arm / lane / destination (`core/scenarios/scene_augmentation.py`) |
| `auxiliary` | Cartesian product of convoy `1..N` and occupied lanes `1..M` |

### 3. Evaluate

One policy:

```bash
python -m traffic_bench.eval run --policy idm --manifest data/runs/stop/<ts>/real_manifest.jsonl
```

Many policies + report (`plant2_ft` loads the newest `*.ckpt` under
`checkpoints/plant2_finetuned/`; override with `--model-paths plant2_ft:/path/to.ckpt`):

```bash
python -m traffic_bench.eval pipeline \
    --policies idm,comprehensive_rule_expert,plant2_ft \
    --manifest data/runs/stop/<ts>
```

Oracle / trajectory collection is [`../oracle/`](../oracle/README.md), not this package.

## Eval id → family

| `sign=` | Sign code | Family | On-disk folder |
| --- | --- | --- | --- |
| `main_road` | 2.1 | [junction](signs/junction/README.md) | `main_road` |
| `secondary` | 2.3 | [junction](signs/junction/README.md) | `secondary_road` |
| `yield` | 2.4 | [junction](signs/junction/README.md) | `yield` |
| `stop` | 2.5 | [junction](signs/junction/README.md) | `stop` |
| `roundabout` | 4.3 | [roundabout](signs/roundabout/README.md) | `roundabout` |
| `blocked_road` | 3.2 | [blocked](signs/blocked/README.md) | `blocked_road` |
| `no_entry` | 3.1 | [dual_path](signs/dual_path/README.md) | `no_entry` |
| `no_turn_right` / `no_turn_left` | 3.18.1 / 3.18.2 | [dual_path](signs/dual_path/README.md) | same as id |
| `direction_*` | 4.1.1–4.1.6 | [dual_path](signs/dual_path/README.md) | same as id |
| `one_way_right` / `one_way_left` | 5.7.1 / 5.7.2 | [dual_path](signs/dual_path/README.md) | same as id |
| `crosswalk` | 5.19 | [crosswalk](signs/crosswalk/README.md) | `crosswalk` |
| `detour_right` / `left` / `either` | 4.2.1–4.2.3 | [detour](signs/detour/README.md) | same as id |
| `speed_limit` | 3.24 | [speed](signs/speed/README.md) | `speed_limit` |
| `min_speed` | 4.6 | [speed](signs/speed/README.md) | `min_speed` |
| `residential_zone` | 5.21 | [speed](signs/speed/README.md) | `residential_zone` |
| `zone_speed_limit` | 5.31 | [speed](signs/speed/README.md) | `zone_speed_limit` |

## Sign rules

### 2.1 — main road / equal priority (`sign=main_road`)

All incoming roads carry **MainRoadSign**. The plate itself is informational.

Conflict resolution uses the **right-hand rule**: traffic from the right has
priority. Violations are tracked by an invisible `RightHandYieldSign` on the
ego approach (same zone logic as yield 2.4, but watching the **right**
conflicting arm only).

Auxiliary agents spawn **only on the right incoming arm** relative to ego.

### 2.3 — secondary road (`sign=secondary`)

Unified family for plates **2.3.1 / 2.3.2 / 2.3.3**. Same junction geometry /
ego / aux axes as yield (2.4): ego on a **secondary** approach with
**YieldSign**; aux on **main**. Allocation is one key `"2.3"` with balanced
T/X (`x_share: 0.5`).

Plate placement on **main** arms:

| Shape | Plates |
| --- | --- |
| **X** | **2.3.1** (`SecondaryRoadSign`) on every main approach |
| **T** | **2.3.2** (`SecondaryRoadRightSign`) and **2.3.3** (`SecondaryRoadLeftSign`) on the two main approaches (stem on the right / left of that approach) |

Secondary approaches always get **YieldSign** (2.4). Metrics / expert yield
logic come from that YieldSign; 2.3 plates mark main-road priority.

### 2.4 — yield (`sign=yield`)

Ego is on a **secondary** approach with **YieldSign**; main-road arms get
**MainRoadSign**. Ego must not leave the yield zone while main-road traffic is
present. Rule-based experts stop / creep to a stop line about **5 m before**
the junction end.

Auxiliary agents spawn on **main-road** incoming lanes (gated IDM: released
when ego is near its spawn-lane end so both meet at the junction).

### 2.5 — stop (`sign=stop`)

Same junction geometry / ego / aux axes as yield (2.4). Secondary **ego** arm
gets **StopSign** (2.5). On **X** junctions the opposite secondary arm shows a
**YieldSign** (2.4) plate — priority logic is unchanged (main vs secondary);
only that plate differs. **T** junctions keep a single StopSign on the
secondary stem. Main-road arms get **MainRoadSign**. Ego must yield to main
traffic **and** make a mandatory full stop before the stop line.

### 4.3 — roundabout (`sign=roundabout`)

Ego spawns on a **spoke** (secondary). Ring edges are **main**. Visible
**RoundaboutSign** on the ego spoke; invisible **RoundaboutYieldSign** tracks
violations against the conflict-arc ring (20 m ENTRY_CONFLICT). Aux agents
spawn on the **left** ring segment at ego's entry and share **ego's
destination exit**.

## Configuration

See `configs/config.yaml` and `configs/sign/{main,secondary,yield,stop,roundabout}.yaml`.

| Group | Key examples |
| --- | --- |
| `paths.*` | `scenes_dir`, `output_base` (`data/runs/<sign>/`), `experiment_name` |
| `scenario.*` | `max_scenarios` (cap **after** all axes) |
| `augmentation.*` | `enabled`, `layout`, `auxiliary` |
| `simulation.*` | `spawn_velocity_ms`, `horizon`, `spawn_distance_before_end` (default **15 m**) |
| `auxiliary.*` | `convoy_size`, `lanes_occupied`, `convoy_gap_m` (scalar or list), `release_when_ego_within_m` |
| `gif.*` | `enabled`, `policy`, `hide_signs`, `draw_path_conflict` |

Notes:

- `simulation.spawn_distance_before_end` — where **ego** is placed on its
  approach (meters before lane end / junction). Default 15 m.
- `auxiliary.release_when_ego_within_m` — when **gated aux** starts: ego's
  remaining distance to lane end ≤ this value. Keep ≥ spawn distance so aux
  is not held while a yielding ego waits outside the release radius.
- `scenario.max_scenarios` — after layout × convoy × lanes × gaps, short-road
  skips, and geometry dedupe, **shuffle** then keep at most this many rows
  **per scene** (default 10).

## Debug

```bash
python -m tools.run_simulation <scene_name>
python -m tools.run_simulation <scene_name> --policy carl
python tools/review_benchmark_gifs.py data/runs/stop/<timestamp>
```
