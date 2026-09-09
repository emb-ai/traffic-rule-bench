# Trajectory collection + oracle selection

- episodes go through `traffic_bench.eval.run.episode.run_one_episode`
- scenes / manifests under `data/scenes/<sign>/` and `data/runs/<sign>/`
- `replay.pkl` written for PlanT2 training

Works for every eval sign (`yield`, `direction/right`, `detour_left`, …).
`SIGN=` accepts one id, a comma-separated list, or `all`.

Train/test is **not** split here — build the train folder first, then pass it
(or let the script pick `data/runs/<sign>/train/real_manifest.jsonl`):

```bash
python -m traffic_bench.eval manifest sign=yield paths.split=train
SIGN=yield MANIFEST=data/runs/yield/train/real_manifest.jsonl \
  ./collect.sh
```

`MANIFEST=` is optional. If unset: `SPLIT=train` (default) or `debug` when `SMOKE=1`.

## Layout

```
oracle/collect/
├── collect.sh                   # orchestrator
├── run.py                       # per-policy collector → all_runs + pkl + json
└── README.md
```

Selection and tables live in `oracle/select/` and `oracle/report/`.

Output of a run (default under `data/trajectories/<sign>/`):

```
../data/trajectories/<sign>/trajectories_<ts>/
├── _logs/run_node<host>_<ts>/
├── _manifests/real_manifest.jsonl
├── _merged/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   └── var_0/<policy>[_<variant>]_replays.jsonl
├── catalog.jsonl
├── idm_rule/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   ├── gifs/*.gif                 # if SAVE_GIFS=1
│   └── by_scene/<uid>/<policy>_<variant>/
│         replay.json
│         replay.pkl
└── experts/                       # after python -m traffic_bench.oracle.select.coverage
```

## 1. Smoke / visual check (recommended first)

```bash
SIGN=yield SMOKE=1 ./collect.sh
```

Check GIFs and that `replay.pkl` exists next to `replay.json`.

Custom tiny run:

```bash
SIGN=yield COUNT=5 SAVE_GIFS=1 SKIP_CARL=1 SKIP_PLANT2=1 \
POLICIES_CPU="idm_rule" \
SMOKE_EXTRA_SAMPLES=0 \
./collect.sh
```

Several signs in one invocation:

```bash
SIGN=yield,stop,direction/right SMOKE=1 ./collect.sh
SIGN=all SMOKE=1 ./collect.sh
```

**Multi-sign scheduling** (not sequential full runs anymore):

1. **GPU phase** — `carl*` / `plant2*` for all signs run in parallel across
   `GPU_IDS` (round-robin, up to `#GPUs × JOBS_PER_GPU` concurrent jobs).
2. **CPU phase** — `idm*` / `ppo*` run **one sign at a time** (within a sign,
   `IDM_CHUNKS` / `N_WORKERS` still shard). Merge runs after each sign's CPU.

```bash
SIGN=yield,stop,crosswalk \
GPU_IDS=0,1,2,3,4,5,6,7 JOBS_PER_GPU=1 \
N_WORKERS=8 IDM_CHUNKS=8 \
./collect.sh
```

Disable cross-sign GPU parallelism (old behavior: finish one sign
completely, then the next):

```bash
MULTI_SIGN_PARALLEL=0 SIGN=yield,stop,crosswalk GPU_IDS=0,1 NN_CHUNKS=1 ./collect.sh
```

### Resuming `final/`

If `data/trajectories/<sign>/final/` exists, collect uses it as `OUT_BASE` with
`RESUME=1` instead of creating `trajectories_<ts>/`.

Safety check before resume:

- UIDs in `final/_manifests/real_manifest.jsonl` must be ⊆ current manifest
  (same `scene_uid` formula as `run.py`)
- any `*/by_scene/<uid>/` already on disk must also belong to the current manifest

Mismatch → hard fail (won't write into the wrong folder). Disable with `USE_FINAL=0`.

```bash
# after a good run:
mv data/trajectories/yield/trajectories_YYYYMMDD_HHMMSS \
   data/trajectories/yield/final

# later — continues missing episodes only:
SIGN=yield ./collect.sh
```

## 2. Full collection

```bash
SIGN=yield \
PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles EGO_CURVE_AWARE=1 \
EGO_HOLD_V0=1 CARL_LONGITUDINAL=tracking \
MANIFEST=data/runs/yield/debug/2026-08-23_18-08-37/real_manifest.jsonl \
POLICIES_CPU="idm_rule ppo_rule" \
POLICIES_CARL="carl_rule" POLICIES_PLANT2="plant2_rule" \
PLANT2_ACTION_MODE=pid \
GPU_IDS=0,1,2,3,4,5,6,7 \
GPUS_CARL=0,1,2,3 GPUS_PLANT2=4,5,6,7 \
JOBS_PER_GPU=1 NN_CHUNKS=0 \
N_WORKERS=32 IDM_CHUNKS=8 \
EXTRA_SAMPLES_COMPREHENSIVE=4 IDM_SEED_BASE=42 \
MAX_STEPS=1500 RESUME=1 \
OUT_BASE=data/trajectories/yield/traj_full \
bash collect.sh
```

Notes:

- **CPU parallelism:** `N_WORKERS` = max concurrent CPU processes (default 8).
IDM-family policies (`idm_rule`, …) are sharded into
`IDM_CHUNKS` workers (default 8) via `--start/--count/--worker-id`.
Sharding is skipped only when `SAVE_GIFS=1` (Panda3D) or `IDM_CHUNKS=1`.
Without sharding, `POLICIES_CPU="idm_rule ppo_rule"`
would only use **2** CPU processes even if `N_WORKERS=8`.
- **GPU (single sign):** each NN policy (`carl_rule`, `plant2_rule`) is sharded
  into `NN_CHUNKS` workers (default `0` = auto = `#GPUs × JOBS_PER_GPU` for that
  pool) via `--start/--count/--worker-id`, round-robin across the pool's cards.
  Default `GPU_IDS=0,1,2,3` auto-splits to `GPUS_CARL=0,1` and `GPUS_PLANT2=2,3`
  (override with `GPUS_CARL=` / `GPUS_PLANT2=`). Cap concurrency with
  `JOBS_PER_GPU` (default 1). `NN_CHUNKS=1` disables within-policy sharding;
  `SAVE_GIFS=1` forces a single process.
- **CPU + GPU overlap:** default `OVERLAP_CPU_GPU=1` starts `carl`/`plant2`
  immediately, then queues CPU under `N_WORKERS` (they run together). Set
  `OVERLAP_CPU_GPU=0` to finish all CPU workers before any GPU pool (less
  MetaDrive contention on a busy node).
- **CPU reserve for GPU sims:** MetaDrive for `carl`/`plant2` also needs host
  CPU. With overlap, `RESERVE_CPU_FOR_GPU=auto` (default) shrinks `N_WORKERS`
  by `(#GPUS_CARL + #GPUS_PLANT2) × JOBS_PER_GPU` — e.g. `N_WORKERS=32` and
  4 GPU workers → effective `N_WORKERS=28`. Override with an integer, or
  `RESERVE_CPU_FOR_GPU=0` to keep the requested `N_WORKERS`.
- **Live progress:** every `PROGRESS_EVERY_S` seconds (default 30) the shell
prints a per-policy bar (`done/target`) and the last `[i/N]` line from each
worker log. Detail: `tail -f $OUT_BASE/_logs/.../<policy>.wXX.log`.
  Cross-sign dashboard (ETA): `python tools/collect_progress.py --watch 30`.
- Default ckpts (relative to repo root):
  - `CARL_CKPT=checkpoints/carl/nuplan_51479_1B/model_best.pth`
  - `PLANT2_CKPT=checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`
  If the file exists, that pool is auto-enabled (`SKIP_CARL` / `SKIP_PLANT2`).
- Traffic is mainly **aux agents**, not SUMO NPCs.
- `RESUME=1` skips episodes that already have non-empty `replay.pkl` + `replay.json`.
- `EXTRA_SAMPLES_COMPREHENSIVE=4` → ego variants `default,s1,s2,s3,s4`.
- Multi-sign with a custom `OUT_BASE` writes `OUT_BASE/<sign_id>/`.
A shared `MANIFEST=` path is ignored unless it contains `{sign}` or `{id}`.

## 3. Oracle selection

```bash
# from repo root — SIGN once, paths default to data/trajectories/<sign>/final
SIGN=yield HORIZON=1500 ./../select/coverage.sh
# → .../final/experts/

# custom tree:
SIGN=yield ROOT=data/trajectories/yield/trajectories_<ts> \
  HORIZON=1500 ./../select/coverage.sh
```

`--signs` / catalog / out-dir are filled from `SIGN` (eval id like `yield`).

## 4. Metrics table

```bash
SIGN=yield HORIZON=1500 ../report/table.sh
# or with an explicit tree:
SIGN=yield ../report/table.sh data/trajectories/yield/trajectories_<ts>
# → .../oracle_metrics/oracle_metrics_summary_top2.md
```

