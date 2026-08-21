# Trajectory collection + oracle selection

Collector for sign evaluation (`yield` 2.4 / `main` 2.1 / `stop` 2.5 / …).

- episodes go through `traffic_bench.eval.run_benchmark.run_one_episode`
- scenes / manifests under `data/<sign>/` (repo root)
- `replay.pkl` written for PlanT2 training

Train/test is **not** split here — use `paths.split=train|test` at
`generate_manifest` time (moscow pool), then pass that run's
`real_manifest.jsonl` as `MANIFEST`.

## Per-sign inputs / outputs

`SIGN` selects a data tree under `priority_bench/data/<sign>/`:


|                    | yield                                            | main                                                 | stop                                            | secondary                                       |
| ------------------ | ------------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| scenes             | `data/yield/scenes`                              | `data/main_road/scenes`                              | `data/stop/scenes`                              | `data/secondary_road/scenes`                    |
| auto `MANIFEST`    | latest `data/yield/output/*/real_manifest.jsonl` | latest `data/main_road/output/*/real_manifest.jsonl` | latest `data/stop/output/*/real_manifest.jsonl` | latest `data/secondary_road/output/*/real_manifest.jsonl` |
| default `OUT_BASE` | `data/yield/trajectories/trajectories_<ts>/`     | `data/main_road/trajectories/trajectories_<ts>/`     | `data/stop/trajectories/trajectories_<ts>/`     | `data/secondary_road/trajectories/trajectories_<ts>/` |
| sidecar slug       | `2_4`                                            | `2_1`                                                | `2_5`                                           | `2_3`                                           |


So `SIGN=yield SMOKE=1 ./collect_trajectories.sh` without `MANIFEST` picks the
newest yield eval/manifest run and writes under `data/yield/trajectories/…`
(never into `data/main_road/`). Override with
`MANIFEST=…/real_manifest.jsonl` and/or `OUT_BASE=…` when needed.

## Layout

```
collect_trajectories/
├── collect_trajectories.sh      # orchestrator (SIGN=yield|main|stop|secondary|roundabout)
├── expert_replay_priority.py    # per-policy collector → all_runs + pkl + json
├── select_experts_coverage.py   # oracle top-1 / top-2 / map selection
├── make_oracle_table.sh         # → oracle_metrics_summary_top2.md
└── README.md
```

Output of a run (default under `data/<sign>/trajectories/`):

```
../data/<sign>/trajectories/trajectories_<ts>/
├── _logs/run_node<host>_<ts>/
├── _manifests/<slug>/real_manifest.jsonl
├── _merged/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   └── var_0/<policy>[_<variant>]_replays.jsonl
├── catalog.jsonl
├── comprehensive_rule_expert/<slug>/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   ├── gifs/*.gif                 # if SAVE_GIFS=1
│   └── by_sign/<slug>/by_scene/<uid>/<policy>_<variant>/
│         replay.json
│         replay.pkl
└── experts/                       # after select_experts_coverage.py
```

`<slug>` is `2_4` (yield), `2_1` (main), `2_5` (stop), or `2_3` (secondary).

## 1. Smoke / visual check (recommended first)

```bash
cd traffic-bench/sign_bench/collect_trajectories
conda activate zinkovich-plant2

SIGN=yield SMOKE=1 ./collect_trajectories.sh
# or: SIGN=main SMOKE=1 ./collect_trajectories.sh
# or: SIGN=stop SMOKE=1 ./collect_trajectories.sh
```

Check GIFs and that `replay.pkl` exists next to `replay.json`:

```bash
ls ../data/yield/trajectories/trajectories_*/comprehensive_rule_expert/2_4/gifs/
find ../data/yield/trajectories -name replay.pkl | head
```

Custom tiny run:

```bash
SIGN=yield COUNT=5 SAVE_GIFS=1 SKIP_CARL=1 SKIP_PLANT2=1 \
POLICIES_CPU="comprehensive_rule_expert" \
SMOKE_EXTRA_SAMPLES=0 \
./collect_trajectories.sh
```

## 2. Full collection

Point `MANIFEST` at a train (or test) manifest from an earlier
`generate_manifest.py … paths.split=train` run. If unset, the latest
`data/<sign>/output/*/real_manifest.jsonl` is used.

```bash
cd traffic-bench/sign_bench/collect_trajectories

SIGN=yield \
PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles EGO_CURVE_AWARE=1 \
EGO_HOLD_V0=1 CARL_LONGITUDINAL=tracking \
MANIFEST=../data/yield/output/<ts>/real_manifest.jsonl \
POLICIES_CPU="comprehensive_rule_expert rule_compliant" \
POLICIES_CARL="carl_rule" POLICIES_PLANT2="plant2_rule" \
PLANT2_ACTION_MODE=pid \
GPU_IDS=0,1,2,3 \
GPUS_CARL=0,1 GPUS_PLANT2=2,3 \
JOBS_PER_GPU=2 \
N_WORKERS=16 IDM_CHUNKS=8 \
EXTRA_SAMPLES_COMPREHENSIVE=4 IDM_SEED_BASE=42 \
MAX_STEPS=1500 RESUME=1 \
OUT_BASE=../data/yield/trajectories/traj_full \
bash collect_trajectories.sh
```

Notes:

- `SIGN=yield|main|stop|secondary` (aliases `2.4|2_4|2.1|2_1|main_road|2.5|2_5|stop_sign|2.3|2_3|secondary_road`).
- **CPU parallelism:** `N_WORKERS` = max concurrent CPU processes (default 8).
  IDM-family policies (`comprehensive_rule_expert`, …) are sharded into
  `IDM_CHUNKS` workers (default 8) via `--start/--count/--worker-id`.
  Without sharding, `POLICIES_CPU="comprehensive_rule_expert rule_compliant"`
  would only use **2** CPU processes even if `N_WORKERS=8`.
- **Live progress:** every `PROGRESS_EVERY_S` seconds (default 30) the shell
  prints a per-policy bar (`done/target`) and the last `[i/N]` line from each
  worker log. Detail: `tail -f $OUT_BASE/_logs/.../<policy>.wXX.log`.
- Default ckpts (relative to `collect_trajectories/`):
  - `CARL_CKPT=../../../../checkpoints/carl/nuplan_51479_1B/model_best.pth`
  - `PLANT2_CKPT=../../../../checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`
  If the file exists, that pool is auto-enabled (`SKIP_CARL` / `SKIP_PLANT2`).
- Traffic is mainly **aux agents**, not SUMO NPCs.
- `RESUME=1` skips episodes that already have non-empty `replay.pkl` + `replay.json`.
- `EXTRA_SAMPLES_COMPREHENSIVE=4` → ego variants `default,s1,s2,s3,s4`.

## 3. Oracle selection

```bash
OUT=../data/yield/trajectories/trajectories_<ts>

python select_experts_coverage.py \
    --root "$OUT" \
    --catalog "$OUT/catalog.jsonl" \
    --signs 2.4 \
    --horizon 1500 \
    --out-dir "$OUT/experts"
```

Use `--signs 2.1` for main and `--signs 2.5` for stop collections.

## 4. Metrics table

```bash
SIGN=yield ./make_oracle_table.sh ../data/yield/trajectories/trajectories_<ts>
# → .../oracle_metrics/oracle_metrics_summary_top2.md
```

## Env → paths


| `SIGN`            | PDD | slug  | scenes                            | auto-manifest                                                      | default `OUT_BASE`                                        |
| ----------------- | --- | ----- | --------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------- |
| `yield`           | 2.4 | `2_4` | `data/yield/scenes`               | latest `data/yield/output/*/real_manifest.jsonl`                   | `data/yield/trajectories/trajectories_<ts>`               |
| `main`            | 2.1 | `2_1` | `data/main_road/scenes`           | latest `data/main_road/output/*/real_manifest.jsonl`               | `data/main_road/trajectories/trajectories_<ts>`           |
| `stop`            | 2.5 | `2_5` | `data/stop/scenes`                | latest `data/stop/output/*/real_manifest.jsonl`                    | `data/stop/trajectories/trajectories_<ts>`                |
| `secondary`       | 2.3 | `2_3` | `data/secondary_road/scenes`      | latest `data/secondary_road/output/*/real_manifest.jsonl`          | `data/secondary_road/trajectories/trajectories_<ts>`      |


Compatibility shims remain under `yield_sign/collect_trajectories` and
`main_sign/collect_trajectories` (forward here with the matching `SIGN`).
