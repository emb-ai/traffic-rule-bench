# Crosswalk (5.19) trajectory collection + oracle selection

Adapted copies of the colleague pipeline (`collect_trajectories.sh` →
`expert_replay.py` → `select_experts_coverage.py`), specialized for
`crosswalk_sign/`:

- episodes go through `crosswalk_sign/run_benchmark.run_one_episode`
- **auxiliary agents OFF by default** (manifest `auxiliary_agent` honored if True)
- **pedestrians** via manifest `use_pedestrian_manager`, `use_pedestrian_yield_rule`,
  `pedestrian_manager` presets (often ego_proximity mid-episode spawn)
- SUMO NPC density from manifest `traffic_density`
- scenes come from `crosswalk_sign/scenes/5_19` + crosswalk manifests
- **`replay.pkl`** is written (RecordManager patch for mid-episode ped spawns)

Oracle class mapping: `5.19` → `PedestrianYieldRule` in `select_experts.SIGN_CLASS_MAP`.

## Layout

```
collect_trajectories/
├── collect_trajectories.sh      # orchestrator (CPU / Carl / PlanT2 pools)
├── expert_replay_crosswalk.py   # per-policy collector → all_runs + pkl + json
├── make_map_split.py            # map-level 80/20 train/test (seed=42)
├── select_experts_coverage.py   # oracle top-1 / top-2 / map selection
├── make_oracle_table.sh         # → oracle_metrics_summary_top2.md
└── README.md
```

Output of a run (aligned with colleague `traj_fv_train80_nodeA_*`):

```
output/trajectories_<ts>/
├── _logs/run_node<host>_<ts>/
├── _manifests/5_19/real_manifest.jsonl
├── _merged/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   └── var_0/<policy>[_<variant>]_replays.jsonl
├── catalog.jsonl
├── comprehensive_rule_expert/5_19/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   ├── gifs/*.gif                 # if SAVE_GIFS=1
│   └── by_sign/5_19/by_scene/<uid>/<policy>_<variant>/
│         replay.json
│         replay.pkl               # training trajectories
└── experts/                       # after select_experts_coverage.py
```

Same relative path as colleague:
`<policy>/<sign>/by_sign/<sign>/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}`

## Train / test split (colleague contract)

```bash
python make_map_split.py \
    --catalog ../benchmark_output/5_19/final_metrics_v1/real_manifest.jsonl
# → catalog_train80.jsonl  catalog_test20.jsonl  catalog_maps_split.json
```

Then collect with the train half:

```bash
MANIFEST=../benchmark_output/5_19/final_metrics_v1/catalog_train80.jsonl \
bash collect_trajectories.sh
```

Keep `catalog_test20.jsonl` for later eval slicing. A map never appears in both
halves.

## 1. Smoke / visual check (recommended first)

```bash
cd pdd-bench/scripts/per_sign_bench/crosswalk_sign/collect_trajectories
conda activate zinkovich-plant2   # or your crosswalk env

COUNT=1 SMOKE_EXTRA_SAMPLES=0 SAVE_GIFS=1 SKIP_CARL=1 SKIP_PLANT2=1 \
POLICIES_CPU="comprehensive_rule_expert" \
./collect_trajectories.sh
```

Check that `replay.pkl` exists next to `replay.json`:

```bash
find output/trajectories_* -name replay.pkl | head
```

`SMOKE=1` defaults to COUNT=3 and EXTRA_SAMPLES=4; override with
`SMOKE_EXTRA_SAMPLES=0` for a single-variant quick check.

## 2. Full collection (colleague-style invocation)

Prefer `MANIFEST=.../catalog_train80.jsonl` after `make_map_split.py`.

```bash
cd pdd-bench/scripts/per_sign_bench/crosswalk_sign/collect_trajectories

PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles EGO_CURVE_AWARE=1 \
EGO_HOLD_V0=1 CARL_LONGITUDINAL=tracking \
MANIFEST=../benchmark_output/5_19/final_metrics_v1/catalog_train80.jsonl \
SCENES_ROOT=../scenes/5_19 \
SIGNS_FILTER=5_19 \
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
OUT_BASE=/path/to/traj_crosswalk_5_19 \
bash collect_trajectories.sh
```

Notes vs the general bench / yield:
- `SIGNS_FILTER` is accepted for parity; crosswalk is **5.19-only**.
- **Aux is OFF**; traffic is pedestrians + optional SUMO density from the row.
- Setting `CARL_CKPT` / `PLANT2_CKPT` auto-enables those pools.
- `RESUME=1` skips episodes that already have non-empty `replay.pkl` + `replay.json`.

`EXTRA_SAMPLES_COMPREHENSIVE=4` → ego variants `default,s1,s2,s3,s4`.

## 3. Oracle selection

```bash
OUT=output/trajectories_<ts>

python select_experts_coverage.py \
    --root "$OUT" \
    --catalog "$OUT/catalog.jsonl" \
    --signs 5.19 \
    --horizon 1500 \
    --out-dir "$OUT/experts"
```

Expert rows include `pkl_path` for downstream PlanT2 repack /
`expert_replay_inenv.py`.

## 4. Metrics table (oracle_metrics_summary_top2.md)

```bash
./make_oracle_table.sh output/trajectories_<ts>
# → output/trajectories_<ts>/oracle_metrics/oracle_metrics_summary_top2.md
```

## Differences vs yield / general collector

| | Yield collector | Crosswalk collector |
|---|---|---|
| Env / NPC | Auxiliary agents on main road | **Pedestrians** + SUMO density from manifest |
| Aux default | ON | **OFF** |
| Runner | `expert_replay_yield.py` | `expert_replay_crosswalk.py` |
| Sign | 2.4 | **5.19** |
| Oracle class | YieldSign | **PedestrianYieldRule** |
| `replay.pkl` | yes (sign/aux patch) | **yes** (mid-episode ped patch) |

Oracle selection needs `all_runs.jsonl`. Training needs `replay.pkl` (+
`expert_actions` in the sidecar). GIFs are for visual validation before a long run.
