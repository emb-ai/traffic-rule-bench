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
├── comprehensive_rule_expert/
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
POLICIES_CPU="comprehensive_rule_expert" \
SMOKE_EXTRA_SAMPLES=0 \
./collect.sh
```

Several signs in one invocation (each writes its own `data/trajectories/<sign>/trajectories_<ts>/`):

```bash
SIGN=yield,stop,direction/right SMOKE=1 ./collect.sh
SIGN=all SMOKE=1 ./collect.sh
```

## 2. Full collection

```bash
SIGN=yield \
PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles EGO_CURVE_AWARE=1 \
EGO_HOLD_V0=1 CARL_LONGITUDINAL=tracking \
MANIFEST=data/runs/yield/train/real_manifest.jsonl \
POLICIES_CPU="comprehensive_rule_expert rule_compliant" \
POLICIES_CARL="carl_rule" POLICIES_PLANT2="plant2_rule" \
PLANT2_ACTION_MODE=pid \
GPU_IDS=0,1,2,3 \
GPUS_CARL=0,1 GPUS_PLANT2=2,3 \
JOBS_PER_GPU=2 \
N_WORKERS=16 IDM_CHUNKS=8 \
EXTRA_SAMPLES_COMPREHENSIVE=4 IDM_SEED_BASE=42 \
MAX_STEPS=1500 RESUME=1 \
OUT_BASE=data/trajectories/yield/traj_full \
bash collect.sh
```

Notes:

- **CPU parallelism:** `N_WORKERS` = max concurrent CPU processes (default 8).
IDM-family policies (`comprehensive_rule_expert`, …) are sharded into
`IDM_CHUNKS` workers (default 8) via `--start/--count/--worker-id`.
Without sharding, `POLICIES_CPU="comprehensive_rule_expert rule_compliant"`
would only use **2** CPU processes even if `N_WORKERS=8`.
- **Live progress:** every `PROGRESS_EVERY_S` seconds (default 30) the shell
prints a per-policy bar (`done/target`) and the last `[i/N]` line from each
worker log. Detail: `tail -f $OUT_BASE/_logs/.../<policy>.wXX.log`.
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
OUT=data/trajectories/yield/trajectories_<ts>

python -m traffic_bench.oracle.select.coverage \
    --root "$OUT" \
    --catalog "$OUT/catalog.jsonl" \
    --signs yield \
    --horizon 1500 \
    --out-dir "$OUT/experts"
```

`--signs` accepts eval ids (`yield`) or official codes (`2.4`).

## 4. Metrics table

```bash
SIGN=yield ../report/table.sh data/trajectories/yield/trajectories_<ts>
# → .../oracle_metrics/oracle_metrics_summary_top2.md
```
