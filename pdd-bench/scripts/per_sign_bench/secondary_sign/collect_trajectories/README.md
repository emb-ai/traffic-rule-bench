# Secondary (2.3.x) trajectory collection + oracle selection

Adapted copies of the colleague pipeline (`collect_trajectories.sh` →
`expert_replay.py` → `select_experts_coverage.py`), specialized for
`secondary_sign/`:

- episodes go through `secondary_sign/run_benchmark.run_one_episode`
- **auxiliary agents** on the main road are always enabled
- scenes come from `secondary_sign/scenes` + yield manifests
- **`replay.pkl`** is written (RecordManager patch + sign guard) for PlanT2 training

## Layout

```
collect_trajectories/
├── collect_trajectories.sh      # orchestrator (CPU / Carl / PlanT2 pools)
├── expert_replay_secondary.py       # per-policy collector → all_runs + pkl + json
├── make_map_split.py            # map-level 80/20 train/test (seed=42)
├── select_experts_coverage.py   # oracle top-1 / top-2 / map selection
├── make_oracle_table.sh         # → oracle_metrics_summary_top2.md
└── README.md
```

Output of a run (aligned with colleague `traj_fv_train80_nodeA_*`):

```
output/trajectories_<ts>/
├── _logs/run_node<host>_<ts>/
├── _manifests/2_3/real_manifest.jsonl
├── _merged/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl              # yield extra (for select_experts join)
│   └── var_0/<policy>[_<variant>]_replays.jsonl
├── catalog.jsonl                  # yield extra
├── comprehensive_rule_expert/2_3/
│   ├── all_runs.jsonl
│   ├── catalog.jsonl
│   ├── gifs/*.gif                 # if SAVE_GIFS=1 (yield QA extra)
│   └── by_sign/2_3/by_scene/<uid>/<policy>_<variant>/
│         replay.json
│         replay.pkl               # training trajectories
└── experts/                       # after select_experts_coverage.py
```

Same relative path as colleague:
`<policy>/<sign>/by_sign/<sign>/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}`

## Train / test split (colleague contract)

Colleague flow: A6 filter → split unique `net_path` 80/20 (`seed=42`) → collect
**only train80** → oracle → PlanT2 `.pt`. Test maps are held out for eval reports.

For secondary (until A6 metrics CSV exists):

```bash
python make_map_split.py \
    --catalog ../benchmark_output/2_3/<ts>/real_manifest.jsonl
# → catalog_train80.jsonl  catalog_test20.jsonl  catalog_maps_split.json
```

Then collect with the train half:

```bash
MANIFEST=../benchmark_output/2_3/<ts>/catalog_train80.jsonl \
bash collect_trajectories.sh
```

Keep `catalog_test20.jsonl` for later eval slicing. A map never appears in both
halves.

## 1. Smoke / visual check (recommended first)

```bash
cd pdd-bench/scripts/per_sign_bench/secondary_sign/collect_trajectories
conda activate zinkovich-plant2   # or your yield env

SMOKE=1 ./collect_trajectories.sh
```

Check GIFs and that `replay.pkl` exists next to `replay.json`:

```bash
ls output/trajectories_*/comprehensive_rule_expert/2_3/gifs/
find output/trajectories_* -name replay.pkl | head
```

Custom tiny run:

```bash
COUNT=5 SAVE_GIFS=1 SKIP_CARL=1 SKIP_PLANT2=1 \
POLICIES_CPU="comprehensive_rule_expert" \
SMOKE_EXTRA_SAMPLES=0 \
./collect_trajectories.sh
```

## 2. Full collection (colleague-style invocation)

Prefer `MANIFEST=.../catalog_train80.jsonl` after `make_map_split.py`.

```bash
cd pdd-bench/scripts/per_sign_bench/secondary_sign/collect_trajectories

PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles EGO_CURVE_AWARE=1 \
EGO_HOLD_V0=1 CARL_LONGITUDINAL=tracking \
MANIFEST=../benchmark_output/2_3/<ts>/catalog_train80.jsonl \
SCENES_ROOT=../scenes/2_3 \
SIGNS_FILTER=2_3 \
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
OUT_BASE=/path/to/traj_secondary_2_3 \
bash collect_trajectories.sh
```

Notes vs the general bench:
- `SIGNS_FILTER` is accepted for parity; yield is **2.3-only**.
- Setting `CARL_CKPT` / `PLANT2_CKPT` auto-enables those pools.
- Secondary (2.3.x) traffic is mainly **aux agents**, not SUMO NPCs.
- `RESUME=1` skips episodes that already have non-empty `replay.pkl` + `replay.json`.

`EXTRA_SAMPLES_COMPREHENSIVE=4` → ego variants `default,s1,s2,s3,s4`.

## 3. Oracle selection (like colleague)

```bash
OUT=output/trajectories_<ts>

python select_experts_coverage.py \
    --root "$OUT" \
    --catalog "$OUT/catalog.jsonl" \
    --signs 2.3.1 2.3.2 2.3.3 \
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

## Differences vs general `expert_replay.py`

| | General bench | Secondary (2.3.x) collector |
|---|---|---|
| Env / NPC | MetaDrive IDM traffic | **Auxiliary agents** on main road |
| Runner | `expert_replay.py` | `expert_replay_secondary.py` → yield `run_benchmark` |
| `replay.pkl` | yes | **yes** (shared RecordManager patch) |
| `replay.json` / `all_runs.jsonl` | yes | yes |
| Train/test | `make_fv_map_split.py` (+A6) | `make_map_split.py` (map 80/20; A6 later) |
| GIF QA | optional | `--save-gifs` / `SMOKE=1` |

Oracle selection needs `all_runs.jsonl`. Training needs `replay.pkl` (+
`expert_actions` in the sidecar). GIFs are for visual validation before a long run.
