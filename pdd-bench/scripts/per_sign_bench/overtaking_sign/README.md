# No-Overtaking (3.20)

Straight **1+1** roads: one ego lane + one opposite lane.

| Code | Title         | Sign class         | Scenes folder  |
| ---- | ------------- | ------------------ | -------------- |
| 3.20 | No overtaking | `NoOvertakingSign` | `scenes/3_20/` |

Incentive:

* **Experts** (`comprehensive_rule_expert`, …) stay behind the mid-lane blocker
  → `compliant_wait_success` / `arrive_dest`.
* **Plain MetaDrive IDM** also tends to stop (it does not lane-change onto the
  opposite directed edge on 1+1). Agents that steer onto the opposite edge
  (some NN / rule policies) accumulate `NoOvertakingSign` violations.

Layout: sign + ego near lane start; stationary aux at ~half length; destination
further along the **same** directed edge past the aux. If ego nearly stops
behind the aux (`wait_behind_*` knobs) with 0 violations for
`wait_behind_success_seconds` (default 2 s), the episode ends early as
`arrive_dest` / `compliant_wait_success`. Horizon timeout is **not** success.


## Quick start

```bash
cd pdd-bench/scripts/per_sign_bench/overtaking_sign
conda activate zinkovich-plant2   # or your env

# Import catalog → core → crop sign_*_s0
python tools/filter_scenes/import_catalog_scenes.py --limit 20
python tools/filter_scenes/crop_straight_scene.py --limit 20 --overwrite

# Manifest + smoke GIFs
python generate_manifest.py
python generate_manifest.py gif.enabled=true gif.policy=idm gif.max_scenes=3
python generate_manifest.py gif.enabled=true gif.policy=comprehensive_rule_expert gif.max_scenes=3
# NN policies need a checkpoint (defaults under pdd-bench/checkpoints/):
python generate_manifest.py gif.enabled=true gif.policy=carl gif.max_scenes=3

# Smoke (CRE vs plain IDM)
python run_benchmark.py \
  --policy comprehensive_rule_expert --run-name smoke_cre \
  --manifest benchmark_output/3_20/<run_dir>/real_manifest.jsonl \
  --scenes-root scenes/3_20 --max-scenes-per-sign 3

python run_benchmark.py \
  --policy idm --run-name smoke_idm \
  --manifest benchmark_output/3_20/<run_dir>/real_manifest.jsonl \
  --scenes-root scenes/3_20 --max-scenes-per-sign 3
```

Eval:

```bash
python eval_pipeline.py \
  --policies idm,comprehensive_rule_expert \
  --manifest benchmark_output/3_20/<run_dir> \
  --scenes-root scenes/3_20 \
  --max-scenes 3
```

Expected smoke: CRE → `compliant_wait_success` / 0 `NoOvertakingSign` events.
`select_experts` treats `3.20` like no-entry for `recompute_dest` when clean.
