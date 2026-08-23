# eval — manifests, closed-loop runs, metrics

Run from the **repository root** after `pip install -e .`.

Scenes come from [`scene_collection/`](../scene_collection/README.md) or
Hugging Face [`emb-ai/traffic-sign-bench`](https://huggingface.co/datasets/emb-ai/traffic-sign-bench)
into `<repo>/data/scenes/<sign>/`. Eval writes `<repo>/data/runs/<sign>/<split>/`
(`debug` by default). `sign=main_road` writes under `data/{scenes,runs}/main_road/`.

Sign rules live under [`signs/`](signs/README.md). Shared engine code is
[`engine/`](engine/README.md). The CLI has three commands: `manifest`, `run`,
`metrics`.

## Folders

| Path | Role |
| --- | --- |
| [`cli.py`](cli.py) | `python -m traffic_bench.eval {manifest,run,metrics}` |
| [`sign_registry.py`](sign_registry.py) | eval id → group, spawn, data folder |
| [`configs/`](configs/) | Hydra YAML; `sign/` ids and `shared/` knobs |
| [`manifest/`](manifest/) | discover scenes → `signs.<group>.expand.generate` → jsonl |
| [`run/`](run/) | closed-loop episodes; one policy, `policies=[…]`, or `policies=all` |
| [`metrics/`](metrics/README.md) | episode JSONL → CSV / markdown |
| [`engine/`](engine/README.md) | map, traffic, spawn, expand types, MetaDrive glue |
| [`signs/`](signs/README.md) | per-group expand / spawn / place / spec |

**`manifest/`** — scenes + Hydra → `real_manifest.jsonl`.
[`manifest/run.py`](manifest/run.py) is the Hydra entry. Each group builds
rows in `signs/<group>/expand.py` via `generate(cfg, scenes)`.

**`run/`** — wrap the env, load a policy, apply a row, place plates, step,
optional GIF. One policy is `run policy=idm sign=yield`. Several policies:
`run policies=[idm,plant2] sign=yield`. All registered policies:
`run policies=all sign=yield` (or `sign=all`). Without `manifest=`, `run`
reads `data/runs/<sign>/test/`.

**`engine/`** — no sign rules. SUMO parse, IDM profiles, MetaDrive patches,
HUD/GIF overlays, T/X arm geometry.

**`signs/<group>/`** — only what is true for that group. Same file roles
(skip if unused): `expand.py` (rows + `generate`), `spawn.py`, `place.py`,
`spec.py`. Dual-path navigation is `signs/dual_path/nav.py`. Roundabout aux
placement is `signs/roundabout/aux.py`.

**`configs/`** — `sign=` ids only under `configs/sign/`. YAML used by more
than one id lives in `configs/shared/` and is pulled via `defaults:`.
Nested folders stay only where Hydra already needs them (`direction/`,
`one_way/`, `no_turn/`, `detour/`). CLI is `sign=yield` or
`sign=direction/right`, not `sign=junction/yield`.

## Run folders

`paths.split` is the only folder switch. The directory name matches the flag.

| Folder | Command | Who reads it |
| --- | --- | --- |
| `data/runs/<sign>/debug/` | `manifest sign=…` (default) | eyes / GIF (`gif.enabled=true`) |
| `data/runs/<sign>/train/` | `manifest sign=… paths.split=train` | oracle (`MANIFEST=…/train/real_manifest.jsonl`) |
| `data/runs/<sign>/test/` | `manifest sign=… paths.split=test` | `eval run` / metrics |

Default `manifest` uses **test** scenes but writes to `debug/`. Re-running
`manifest` into a folder that already has `eval_out/` or `gifs/` stops; remove
those dirs first. A clean folder (only jsonl / `repro/`) is overwritten in place.

## Commands

```bash
# debug folder (test scenes)
python -m traffic_bench.eval manifest sign=yield
python -m traffic_bench.eval manifest sign=direction/right
python -m traffic_bench.eval manifest sign=yield gif.enabled=true gif.max_scenes=8

# stable train / test folders
python -m traffic_bench.eval manifest sign=yield paths.split=train
python -m traffic_bench.eval manifest sign=yield paths.split=test
python -m traffic_bench.eval manifest sign=all paths.split=test
python -m traffic_bench.eval manifest sign=yield,stop,main_road paths.split=train
# after sign=all / a list: per-sign scenes + rows, then totals

# closed-loop: default input is data/runs/<sign>/test/
python -m traffic_bench.eval run policy=idm sign=yield
python -m traffic_bench.eval run policies=all sign=yield
python -m traffic_bench.eval run policies=all sign=all
python -m traffic_bench.eval run policy=idm manifest=data/runs/yield/debug

python -m traffic_bench.eval metrics csv --episodes-root <eval_out>/benchmark --out <eval_out>/metrics_per_episode.csv
python -m traffic_bench.eval metrics aggregate --csv <eval_out>/metrics_per_episode.csv --out-dir <eval_out>
python -m traffic_bench.eval metrics report --run-root <eval_out>
```

Hydra overrides on `manifest`:

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

Output under `data/runs/<sign>/debug/` by default (`sign=main_road` → `data/runs/main_road/`):

- `real_manifest.jsonl` — scenario rows (each has `split`)
- `repro/` — pool snapshot + split filter + allocations ref
- `config.yaml` — resolved Hydra config (`paths.split` = folder, `paths.scene_split` = train/test scenes)
- `gifs/` — if `gif.enabled=true`

Augmentation axes live in `configs/shared/` and per-sign yaml under `augmentation:`:

| Axis | Meaning |
| --- | --- |
| `layout` | Ego × aux arm / lane / destination (`engine/spawn/scene_augmentation.py` + `signs/<group>/spawn.py`) |
| `auxiliary` | Cartesian product of convoy `1..N` and occupied lanes `1..M` |

### 3. Evaluate

One policy:

```bash
python -m traffic_bench.eval run policy=idm sign=stop
```

Many policies + report (`plant2_ft` loads the newest `*.ckpt` under
`checkpoints/plant2_finetuned/`; override with `model_paths.plant2_ft=/path/to.ckpt`):

```bash
python -m traffic_bench.eval run \
    policies=[idm,comprehensive_rule_expert,plant2_ft] \
    sign=stop
```

`policies=all` is the registered set: `idm` and `comprehensive_rule_expert` each
× `default,s1–s4`, plus `carl`, `carl_rule`, `plant2`, `plant2_rule`,
`plant2_ft`, `rule_compliant`, `ppo_lidar`.

Oracle / trajectory collection is [`../oracle/`](../oracle/README.md), not this package.
Always pass an explicit train manifest:

```bash
SIGN=yield MANIFEST=data/runs/yield/train/real_manifest.jsonl \
  ./traffic_bench/oracle/collect_trajectories/collect_trajectories.sh
```

## Eval id → group

| `sign=` | Sign code | Group | On-disk folder |
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

Unified group for plates **2.3.1 / 2.3.2 / 2.3.3**. Same junction geometry /
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

See `configs/config.yaml` (manifest) and `configs/run.yaml` (closed-loop).
Sign knobs: `configs/sign/{main_road,secondary,yield,stop,roundabout}.yaml`.
Shared knobs: `configs/shared/`.

| Group | Key examples |
| --- | --- |
| `paths.*` | `scenes_dir`, `output_base` (`data/runs/<sign>/`), `split` (`debug` / `train` / `test`) |
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
python tools/review_benchmark_gifs.py data/runs/stop/debug
```
