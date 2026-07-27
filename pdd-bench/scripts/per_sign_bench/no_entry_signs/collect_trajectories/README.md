# No-entry (3.1 + 3.2) combined trajectory collection + oracle selection

One collector for **both** no-entry signs with **equal map contribution**.

- episodes go through `no_entry_signs/run_benchmark.run_one_episode`
- **no auxiliary agents**, **no pedestrians**; SUMO density from the manifest
- dual scene roots: `scenes/3_1` and `scenes/3_2` (pass parent `scenes/`)
- **`replay.pkl`** via shared `bench/record_manager_patch.py` + sign guard

## Equal map contribution

```bash
python build_combined_catalog.py
# → ../benchmark_output/combined/real_manifest_balanced.jsonl
#    n = min(n_maps_3.1, n_maps_3.2) maps from each (seed=42);
#    keep ALL rows for selected maps; interleave signs for smoke.

python make_map_split.py \
    --catalog ../benchmark_output/combined/real_manifest_balanced.jsonl
# → catalog_train80.jsonl / catalog_test20.jsonl  (stratified by sign_code)
```

Raw sources (no map overlap):

| sign | manifest | rows | maps |
|------|----------|------|------|
| 3.1 | `benchmark_output/3_1/final_metrics_v1/real_manifest.jsonl` | 720 | 194 |
| 3.2 | `benchmark_output/3_2/final_metrics_v1/real_manifest.jsonl` | 744 | 200 |

## Layout

```
collect_trajectories/
├── build_combined_catalog.py    # equal-map 3.1+3.2 catalog
├── collect_trajectories.sh      # orchestrator (CPU / Carl / PlanT2 pools)
├── expert_replay_no_entry.py    # multi-sign collector → all_runs + pkl + json
├── make_map_split.py            # map-level 80/20 stratified by sign_code
├── select_experts_coverage.py   # oracle top-1 / top-2 / map (--signs 3.1 3.2)
├── make_oracle_table.sh
└── README.md
```

Output of a run:

```
output/trajectories_<ts>/
├── _logs/run_node<host>_<ts>/
├── _manifests/3_1_3_2/real_manifest.jsonl
├── _merged/all_runs.jsonl
├── catalog.jsonl
├── comprehensive_rule_expert/
│   ├── catalog.jsonl
│   ├── 3_1/
│   │   ├── all_runs.jsonl
│   │   └── by_sign/3_1/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
│   └── 3_2/
│       ├── all_runs.jsonl
│       └── by_sign/3_2/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
└── experts/                       # after select_experts_coverage.py
```

**Output-dir contract:** shell passes `--output-dir $OUT_BASE/$policy`
(parent **without** slug). One process reads the combined catalog and writes
both `3_1/` and `3_2/` trees. Per-row `scenes_root` resolves to
`SCENES_ROOT/<slug>/`.

Legacy: `--output-dir $OUT_BASE/$policy/3_1` still works (replay_root = parent).

## 1. Build catalogs (required once)

```bash
cd pdd-bench/scripts/per_sign_bench/no_entry_signs/collect_trajectories
python build_combined_catalog.py
python make_map_split.py \
    --catalog ../benchmark_output/combined/real_manifest_balanced.jsonl
```

Train/test row counts per `sign_code` should be roughly equal between 3.1 and 3.2.

## 2. Smoke

```bash
cd pdd-bench/scripts/per_sign_bench/no_entry_signs/collect_trajectories
conda activate zinkovich-plant2   # or your env

COUNT=2 SAVE_GIFS=0 SKIP_CARL=1 SKIP_PLANT2=1 \
POLICIES_CPU="comprehensive_rule_expert" \
EXTRA_SAMPLES_COMPREHENSIVE=0 \
./collect_trajectories.sh
```

Balanced catalog / train80 are interleaved, so `COUNT=2` yields one 3.1 + one 3.2.
Verify:

```bash
find output/trajectories_* -path '*/3_1/*/replay.pkl' | head
find output/trajectories_* -path '*/3_2/*/replay.pkl' | head
```

## 3. Full collection

```bash
MANIFEST=../benchmark_output/combined/catalog_train80.jsonl \
SCENES_ROOT=../scenes \
SIGNS_FILTER=3_1,3_2 \
POLICIES_CPU="comprehensive_rule_expert rule_compliant" \
EXTRA_SAMPLES_COMPREHENSIVE=4 IDM_SEED_BASE=42 \
MAX_STEPS=1500 RESUME=1 \
OUT_BASE=./output/traj_no_entry_3_1_3_2_train80 \
bash collect_trajectories.sh
```

`RESUME=1` skips episodes that already have non-empty `replay.pkl` + `replay.json`.

## 4. Oracle selection

```bash
OUT=output/trajectories_<ts>

python select_experts_coverage.py \
    --root "$OUT" \
    --catalog "$OUT/catalog.jsonl" \
    --signs 3.1 3.2 \
    --horizon 1500 \
    --out-dir "$OUT/experts"

./make_oracle_table.sh "$OUT"
```

Oracle classes (from `select_experts.SIGN_CLASS_MAP`): `3.1 → NoEntrySign`,
`3.2 → NoTrafficSign`.
