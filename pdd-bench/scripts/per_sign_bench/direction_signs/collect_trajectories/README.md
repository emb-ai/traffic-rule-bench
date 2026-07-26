# direction (4.1.1–4.1.6) combined trajectory collection + oracle selection

One collector for **all** listed signs with **equal map contribution**.

- episodes go through `direction_signs/run_benchmark.run_one_episode`
- **no auxiliary agents**, **no pedestrians**; SUMO density from the manifest
- scene roots: `scenes/{4_1_1,4_1_2,4_1_3,4_1_4,4_1_5,4_1_6}` (pass parent `scenes/`)
- **`replay.pkl`** via shared `bench/record_manager_patch.py` + sign guard

## Equal map contribution

```bash
python build_combined_catalog.py
python make_map_split.py \
    --catalog ../benchmark_output/combined/real_manifest_balanced.jsonl
```

## Layout

```
collect_trajectories/
├── build_combined_catalog.py
├── collect_trajectories.sh
├── expert_replay_direction.py
├── make_map_split.py
├── select_experts_coverage.py
├── make_oracle_table.sh
└── README.md
```

```
output/trajectories_<ts>/
├── comprehensive_rule_expert/
│   ├── catalog.jsonl
│   ├── 4_1_1/
│   │   ├── all_runs.jsonl
│   │   └── by_sign/4_1_1/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
│   ├── 4_1_2/
│   │   ├── all_runs.jsonl
│   │   └── by_sign/4_1_2/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
│   ├── 4_1_3/
│   │   ├── all_runs.jsonl
│   │   └── by_sign/4_1_3/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
│   ├── 4_1_4/
│   │   ├── all_runs.jsonl
│   │   └── by_sign/4_1_4/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
│   ├── 4_1_5/
│   │   ├── all_runs.jsonl
│   │   └── by_sign/4_1_5/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
│   ├── 4_1_6/
│   │   ├── all_runs.jsonl
│   │   └── by_sign/4_1_6/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
```

## Smoke

```bash
COUNT=6 SAVE_GIFS=1 SKIP_CARL=1 SKIP_PLANT2=1 SMOKE_EXTRA_SAMPLES=0 \
  bash collect_trajectories.sh
```

## Full train80

```bash
MANIFEST=../benchmark_output/combined/catalog_train80.jsonl \
  SCENES_ROOT=../scenes SIGNS_FILTER=4_1_1,4_1_2,4_1_3,4_1_4,4_1_5,4_1_6 \
  POLICIES_CPU="comprehensive_rule_expert rule_compliant" \
  EXTRA_SAMPLES_COMPREHENSIVE=4 MAX_STEPS=1500 RESUME=1 \
  OUT_BASE=./output/traj_4_1_train80 \
  bash collect_trajectories.sh
```

## Oracle

```bash
python select_experts_coverage.py --root $OUT_BASE --catalog $OUT_BASE/catalog.jsonl \
    --signs 4.1.1 4.1.2 4.1.3 4.1.4 4.1.5 4.1.6 --horizon 1500 --out-dir $OUT_BASE/experts
./make_oracle_table.sh $OUT_BASE
```
