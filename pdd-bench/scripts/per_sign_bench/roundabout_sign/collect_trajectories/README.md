# Roundabout (4.3) trajectory collection + oracle selection

Adapted copies of the colleague pipeline (`collect_trajectories.sh` →
`expert_replay.py` → `select_experts_coverage.py`), specialized for
`roundabout_sign/`:

- episodes go through `roundabout_sign/run_benchmark.run_one_episode`
- **auxiliary agents** on ring/spoke approaches are always enabled
- **no pedestrians**
- scenes come from `roundabout_sign/scenes/4_3` + roundabout manifests
- **`replay.pkl`** is written (RecordManager patch + sign guard) for PlanT2 training
- `aux_release_when_ego_within_m` default **20.0** (same as yield)

## Layout

```
collect_trajectories/
├── collect_trajectories.sh         # orchestrator (CPU / Carl / PlanT2 pools)
├── expert_replay_roundabout.py     # per-policy collector → all_runs + pkl + json
├── make_map_split.py               # map-level 80/20 train/test (seed=42)
├── select_experts_coverage.py      # oracle top-1 / top-2 / map selection
├── make_oracle_table.sh            # → oracle_metrics_summary_top2.md
└── README.md
```

Output of a run (aligned with colleague `traj_fv_train80_nodeA_*`):

```
output/trajectories_<ts>/
├── _logs/run_node<host>_<ts>/
├── _manifests/4_3/real_manifest.jsonl
├── _merged/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   └── var_0/<policy>[_<variant>]_replays.jsonl
├── catalog.jsonl
├── comprehensive_rule_expert/4_3/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   ├── gifs/*.gif                 # if SAVE_GIFS=1
│   └── by_sign/4_3/by_scene/<uid>/<policy>_<variant>/
│         replay.json
│         replay.pkl               # training trajectories
└── experts/                       # after select_experts_coverage.py
```

Same relative path as colleague:
`<policy>/<sign>/by_sign/<sign>/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}`

Oracle compliance uses `SIGN_CLASS_MAP["4.3"] = "RoundaboutYieldSign"`
(the invisible yield tracker that records violations).

## Train / test split

```bash
python make_map_split.py \
    --catalog ../benchmark_output/4_3/final_metrics_v1/real_manifest.jsonl
# → catalog_train80.jsonl  catalog_test20.jsonl  catalog_maps_split.json
```

Then collect with the train half:

```bash
MANIFEST=../benchmark_output/4_3/final_metrics_v1/catalog_train80.jsonl \
bash collect_trajectories.sh
```

## 1. Smoke / visual check (recommended first)

```bash
cd pdd-bench/scripts/per_sign_bench/roundabout_sign/collect_trajectories
conda activate zinkovich-plant2   # or your env

COUNT=1 SKIP_CARL=1 SKIP_PLANT2=1 \
POLICIES_CPU="comprehensive_rule_expert" \
SMOKE_EXTRA_SAMPLES=0 \
./collect_trajectories.sh
```

Check that `replay.pkl` exists next to `replay.json`:

```bash
find output/trajectories_* -path '*/4_3/by_sign/4_3/*/replay.pkl' | head
```

## 2. Full collection

Prefer `MANIFEST=.../catalog_train80.jsonl` after `make_map_split.py`.

```bash
cd pdd-bench/scripts/per_sign_bench/roundabout_sign/collect_trajectories

PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles EGO_CURVE_AWARE=1 \
EGO_HOLD_V0=1 CARL_LONGITUDINAL=tracking \
MANIFEST=../benchmark_output/4_3/final_metrics_v1/catalog_train80.jsonl \
SCENES_ROOT=../scenes/4_3 \
SIGNS_FILTER=4_3 \
POLICIES_CPU="comprehensive_rule_expert rule_compliant" \
POLICIES_CARL="carl_rule" POLICIES_PLANT2="plant2_rule" \
CARL_CKPT=/path/to/carl/model_best.pth \
PLANT2_CKPT=/path/to/epoch%3D029_final_3.ckpt \
PLANT2_ACTION_MODE=pid \
GPU_IDS=0,1,2,3 \
GPUS_CARL=0,1 GPUS_PLANT2=2,3 \
JOBS_PER_GPU=2 \
N_WORKERS=16 \
EXTRA_SAMPLES_COMPREHENSIVE=4 IDM_SEED_BASE=42 \
MAX_STEPS=1500 RESUME=1 \
OUT_BASE=/path/to/traj_roundabout_4_3 \
bash collect_trajectories.sh
```

Notes:
- `SIGNS_FILTER` is accepted for parity; roundabout is **4.3-only**.
- Roundabout traffic is mainly **aux agents**, not SUMO NPCs / pedestrians.
- `RESUME=1` skips episodes that already have non-empty `replay.pkl` + `replay.json`.

## 3. Oracle selection

```bash
OUT=output/trajectories_<ts>

python select_experts_coverage.py \
    --root "$OUT" \
    --catalog "$OUT/catalog.jsonl" \
    --signs 4.3 \
    --horizon 1500 \
    --out-dir "$OUT/experts"
```

## 4. Metrics table

```bash
./make_oracle_table.sh output/trajectories_<ts>
# → output/trajectories_<ts>/oracle_metrics/oracle_metrics_summary_top2.md
```
