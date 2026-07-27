# lane_direction (5.15.1) trajectory collection + oracle selection

Collector for PDD **5.15.1** (directions of movement by lanes).

- episodes go through `lane_direction_signs/run_benchmark.run_one_episode`
- **no auxiliary agents**, **no pedestrians**; SUMO density from the manifest
- scene root: `scenes/5_15_1` (pass parent `scenes/` or the slug dir)
- **`replay.pkl`** via shared `bench/record_manager_patch.py` + sign guard
- ego starts on the **wrong** lane and must peer-LC onto `target_lane_num`

## Catalog

```bash
python build_combined_catalog.py
# optional: pin a specific run
# python build_combined_catalog.py --manifest ../benchmark_output/5_15_1/<ts>/real_manifest.jsonl

python make_map_split.py \
    --catalog ../benchmark_output/combined/real_manifest_balanced.jsonl
```

## Layout

```
collect_trajectories/
├── build_combined_catalog.py
├── collect_trajectories.sh
├── expert_replay_lane_direction.py
├── make_map_split.py
├── select_experts_coverage.py
├── make_oracle_table.sh
└── README.md
```

```
output/trajectories_<ts>/
├── comprehensive_rule_expert/
│   ├── catalog.jsonl
│   └── 5_15_1/
│       ├── all_runs.jsonl
│       └── by_sign/5_15_1/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
```

## Smoke

```bash
COUNT=2 SAVE_GIFS=1 SKIP_CARL=1 SKIP_PLANT2=1 SMOKE_EXTRA_SAMPLES=0 \
  bash collect_trajectories.sh
```

Auto-picks newest `benchmark_output/5_15_1/*/real_manifest.jsonl` if no combined catalog yet.

## Full train80

```bash
MANIFEST=../benchmark_output/combined/catalog_train80.jsonl \
  SCENES_ROOT=../scenes SIGNS_FILTER=5_15_1 \
  POLICIES_CPU="comprehensive_rule_expert rule_compliant" \
  EXTRA_SAMPLES_COMPREHENSIVE=4 MAX_STEPS=1500 RESUME=1 \
  OUT_BASE=./output/traj_lane_direction_5_15_1_train80 \
  bash collect_trajectories.sh
```

## Oracle

```bash
python select_experts_coverage.py --root $OUT_BASE --catalog $OUT_BASE/catalog.jsonl \
    --signs 5.15.1 --horizon 1500 --out-dir $OUT_BASE/experts
./make_oracle_table.sh $OUT_BASE
```
