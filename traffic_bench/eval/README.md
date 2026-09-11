# `eval/` — closed-loop evaluation

Evaluation pipeline for TrafficSignBench:

`scenes → manifest → closed-loop runs → metrics`

Scenes must be available under:

```
data/scenes/<sign>/
```

They can be generated with `scene_collection/` or downloaded from the official `emb-ai/traffic-sign-bench` dataset.

Outputs are written under:

```
data/runs/<sign>/<split>/
```

The CLI has three commands:

```
manifest → run → metrics
```

## Quick start

### 1. Build a manifest

Debug manifest (test scenes):

```
python -m traffic_bench.eval manifest sign=yield
```

*or:*

```
python -m traffic_bench.eval manifest sign=yield paths.split=test    # generate test
python -m traffic_bench.eval manifest sign=yield paths.split=train   # generate train
```

`manifest` is **one sign per invocation**. Loop over signs:

```bash
# Hydra sign ids (same set as run_signs_parallel.sh). Nested: direction/right, …
SIGNS=(
  main_road secondary yield stop roundabout blocked_road no_entry
  no_turn/right no_turn/left
  direction/straight direction/right direction/left
  direction/straight_right direction/straight_left direction/left_right
  one_way/right one_way/left
  detour/right detour/left detour/either
  speed_limit min_speed residential_zone zone_speed_limit crosswalk
)
SPLIT=test   # or train

for s in "${SIGNS[@]}"; do
  echo "======== manifest $s ($SPLIT) ========"
  python -m traffic_bench.eval manifest "sign=$s" "paths.split=$SPLIT" \
    || echo "[FAIL] $s"
done
```

### 2. Run closed-loop evaluation

Single policy:

```
python -m traffic_bench.eval run policy=idm sign=yield
```

*or all registered policies:*

```
python -m traffic_bench.eval run policies=all sign=yield
```

*or run all signs:*

```
python -m traffic_bench.eval run policies=all sign=all
```

### 3. Parallel multi-sign eval (recommended on multi-GPU)

One process pool for CPU policies and one sign per GPU for CARL (same layout as `oracle/collect/collect.sh`):

```
# train (default)
CPU_WORKERS=2 GPUS=1,2,3,4,5,6,7 JOBS=16 JOBS_NN=32 \
  bash traffic_bench/eval/run/run_signs_parallel.sh

# test
SPLIT=test SIGNS="main_road secondary yield stop roundabout" \
CPU_WORKERS=4 GPUS=1,2,3,4,5,6,7 JOBS=16 JOBS_NN=16 \
  bash traffic_bench/eval/run/run_signs_parallel.sh

# subset of policies (skip the other pool). Extra GPUs are packed onto fewer signs.
SPLIT=test SIGNS="no_turn/left" POLICIES=carl,carl_rule \
GPUS=0,1,2,3 JOBS_NN=32 \
  bash traffic_bench/eval/run/run_signs_parallel.sh
```

- `SPLIT` — `train` (default) or `test`; manifests under `data/runs/<sign>/<SPLIT>/`
- Ready manifests are scheduled first; missing ones stay pending until they appear (`WAIT_POLL=30`, `WAIT_TIMEOUT=0` = forever)
- `POLICIES` — optional subset (`carl,carl_rule` or `[idm,ppo_rule]`). Splits into CPU vs GPU lists and skips the empty side. Else `CPU_POLICIES` / `GPU_POLICIES` (set either to empty to skip that pool)
- `CPU_WORKERS` — concurrent signs for CPU policies (`idm` / `idm_rule` / `ppo_lidar` / `ppo_rule`)
- `GPUS` — physical GPU indices for `carl` / `carl_rule` / `plant2` / `plant2_rule`. Free cards are packed onto ready signs (`cuda_devices=`); one sign can use several GPUs
- `JOBS` / `JOBS_NN` — scene workers inside each sign (CPU vs **per GPU** for NN; a 4-GPU sign gets `JOBS_NN×4` workers)
- After every sign’s `eval run` (metrics csv/report already), `metrics combine` writes `data/runs/_all/<SPLIT>/` (`RUN_METRICS=0` to skip combine)
- defaults: full **16 baselines** — `ego_variants=[default,s1,s2,s3,s4]` for IDM family + PPO/CaRL/PlanT2 base & rule
- logs: `data/eval_parallel_logs/<SPLIT>/`

Stop everything:

```
pkill -f 'run_signs_parallel\.sh'
pkill -f 'traffic_bench\.eval'
```

**Orphan ProcessPool workers (important).** CPU eval uses
`ProcessPoolExecutor` + `multiprocessing.spawn` (`jobs=…` in
`run/policies.py`). If the parent is killed with Ctrl+C / `kill` /
`pkill` before a clean `executor.shutdown()`, the spawn workers and
`resource_tracker` processes often survive with `PPID=1`. They sit idle
on MetaDrive/pipes, keep writing to the old log, and can waste hundreds
of GiB of RAM for days.

Symptoms: many
`python -c from multiprocessing.spawn import spawn_main…` (and
`resource_tracker`) with `PPID=1`; stdout pointing at
`data/eval_parallel_logs/<SPLIT>/cpu_*.log`; log ends in
`KeyboardInterrupt` inside `as_completed`. Incident note (2026-09):
~240 orphans held ~340 GiB RAM for 1–5 days after a Ctrl+C’d
`run_signs_parallel` / `jobs=16` test eval.

Cleanup (targets only `PPID=1` spawn/`resource_tracker` — does **not**
touch `oracle/collect`):

```bash
# list orphans (adjust ENV if needed)
ENV=zinkovich-plant2
ps -eo pid,ppid,etime,pcpu,rss,cmd | awk -v e="$ENV" \
  'index($0,e) && (/multiprocessing.spawn/ || /resource_tracker/) && $2==1'

# kill them
ps -eo pid,ppid,cmd --no-headers | awk -v e="$ENV" \
  'index($0,e) && (/multiprocessing.spawn/ || /resource_tracker/) && $2==1 {print $1}' \
  | xargs -r kill -9
```

After every interrupted parallel eval, run the cleanup (or always pair
`pkill -f 'traffic_bench\.eval'` with the spawn/`resource_tracker`
kill above). Prefer letting the run finish or stopping via the parallel
script so pools can shut down cleanly.

Progress + ETA:

```
python tools/eval_progress.py
python tools/eval_progress.py --watch 30
```

### 4. Compute metrics

```
python -m traffic_bench.eval metrics combine sign=all
```

### 5. Baseline comparison plots

After per-sign reports exist under `data/runs/*/train/eval_out/reports/`:

```
python -m traffic_bench.eval metrics plot
```

Writes PNGs + `analysis.md` to `data/runs/_all/train/plots/benchmark/` — per-sign bar charts (base vs rule expert), family overviews, macro summary. See `traffic_bench/eval/metrics/README.md`.

## Evaluation workflow

### Manifest

`manifest` discovers scenes and expands each scene into scenario rows.

Output:

```
data/runs/<sign>/<split>/
├── real_manifest.jsonl
├── config.yaml
├── repro/
└── gifs/                    # optional
```

Each manifest row contains the information required to reproduce one scenario.

Scenario augmentation is controlled by:

- `augmentation.layout` — ego/arm/lane/destination variants;
- `augmentation.auxiliary` — convoy size × occupied lanes;
- `scenario.max_scenarios` — final per-scene cap;
- `scenario.max_total` — global row cap. Unset on train/test means
  `signs.yaml` `n_train`/`n_test` × `max_scenarios` (refill after dropped maps).
  Debug stays uncapped. Override with `scenario.max_total=N`.
  Dual-path: unique **maps** must hit `n_train`/`n_test` (80/20); leftover
  combos are not used to hide missing maps. Looping MetaDrive routes are
  moved aside before refill; see [signs/dual_path](signs/dual_path/README.md#filling-traintest-map-quotas).

`scenario.max_scenarios` is applied **after** augmentation, filtering, geometry deduplication, and shuffling.

### Run

`run` loads a policy, applies a manifest row, places the required signs, and executes the scenario in closed loop.

One policy:

```
python -m traffic_bench.eval run policy=idm sign=yield
```

Several policies:

```
python -m traffic_bench.eval run \
    policies=[idm,plant2] \
    sign=yield
```

`policy=` accepts a single policy name; use `policies=[...]` for multiple policies.

`policies=all` runs the complete registered policy set.

If `manifest=` is not provided, `run` uses:

```
data/runs/<sign>/test/
```

when available, otherwise `debug/latest`.

### Metrics

Metrics operate on closed-loop episode outputs:

```
episode JSONL → per-episode CSV → aggregate/report
```

Use `metrics combine sign=all` for one overall report across signs.

## Run folders

`paths.split` controls the output folder:


| Split   | Command                                 | Purpose                                        |
| ------- | --------------------------------------- | ---------------------------------------------- |
| `debug` | `manifest sign=yield`                   | Timestamped test-scene snapshot for inspection |
| `train` | `manifest sign=yield paths.split=train` | Stable train manifest                          |
| `test`  | `manifest sign=yield paths.split=test`  | Stable evaluation manifest                     |


Debug manifests are written to:

```
data/runs/<sign>/debug/<timestamp>/
```

and `debug/latest` points to the newest snapshot.

`train/` and `test/` are immutable snapshots: remove the directory before rebuilding an existing manifest.

## Configuration

Hydra configuration is split into:

```
configs/
├── config.yaml          # manifest defaults
├── run.yaml             # closed-loop defaults
├── sign/                # sign-specific configuration
└── shared/              # configuration shared by multiple signs
```

Common overrides:

```
# Limit scenarios
python -m traffic_bench.eval manifest \
    sign=yield scenario.max_scenarios=20

# Disable auxiliary augmentation
python -m traffic_bench.eval manifest \
    sign=yield augmentation.auxiliary=false

# GIFs during manifest generation
python -m traffic_bench.eval manifest \
    sign=yield gif.enabled=true gif.max_scenes=8
```

Nested Hydra groups use paths such as:

```
sign=direction/right
sign=detour/right
```

## Signs

Sign-specific logic lives under `signs/`.


| `sign=`                            | Sign code       | Group                                    | On-disk folder     |
| ---------------------------------- | --------------- | ---------------------------------------- | ------------------ |
| `main_road`                        | 2.1             | [junction](signs/junction/README.md)     | `main_road`        |
| `secondary`                        | 2.3             | [junction](signs/junction/README.md)     | `secondary_road`   |
| `yield`                            | 2.4             | [junction](signs/junction/README.md)     | `yield`            |
| `stop`                             | 2.5             | [junction](signs/junction/README.md)     | `stop`             |
| `roundabout`                       | 4.3             | [roundabout](signs/roundabout/README.md) | `roundabout`       |
| `blocked_road`                     | 3.2             | [blocked](signs/blocked/README.md)       | `blocked_road`     |
| `no_entry`                         | 3.1             | [dual_path](signs/dual_path/README.md)   | `no_entry`         |
| `no_turn_right` / `no_turn_left`   | 3.18.1 / 3.18.2 | [dual_path](signs/dual_path/README.md)   | same as id         |
| `direction_*`                      | 4.1.1–4.1.6     | [dual_path](signs/dual_path/README.md)   | same as id         |
| `one_way_right` / `one_way_left`   | 5.7.1 / 5.7.2   | [dual_path](signs/dual_path/README.md)   | same as id         |
| `crosswalk`                        | 5.19            | [crosswalk](signs/crosswalk/README.md)   | `crosswalk`        |
| `detour_right` / `left` / `either` | 4.2.1–4.2.3     | [detour](signs/detour/README.md)         | same as id         |
| `speed_limit`                      | 3.24            | [speed](signs/speed/README.md)           | `speed_limit`      |
| `min_speed`                        | 4.6             | [speed](signs/speed/README.md)           | `min_speed`        |
| `residential_zone`                 | 5.21            | [speed](signs/speed/README.md)           | `residential_zone` |
| `zone_speed_limit`                 | 5.31            | [speed](signs/speed/README.md)           | `zone_speed_limit` |


