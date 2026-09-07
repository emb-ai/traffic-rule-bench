# Reference model: `joint_k3_augFIX_lr3e4_ep24` (checkpoint `last`)

PlanT2 fine-tuned **jointly** on stop-sign (2.5) and detour (4.2.1 / 4.2.2 / 4.2.3)
expert replays. This is the checkpoint behind the sign-compliance gif page and the
numbers quoted in the paper draft. Everything here is copied verbatim from the run
so the numbers can be re-derived without access to the NFS share.

## Checkpoint (not in git, 446 MB)

```
plant2/PlanT/checkpoints_ft/joint_k3_augFIX_lr3e4_ep24/last_ft_joint_k3_augFIX_lr3e4_ep24_1.ckpt
sha256  eb62ae028fe774c837218036f7a8790599097de3c3df4c50056601392c4f54c2
```

On `antonov` the repo lives at `/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench`.
Use `last`, not `best_007`: `val/loss_all` is dominated by `val/loss_egospeed`, which
grows monotonically while `val/loss_path` keeps falling, so "best" by that criterion
picks a barely fine-tuned epoch (see *Caveats*).

## Results

**Detour benchmark** — 122-scene catalog `detour_v1/catalog_fv_test20.jsonl`, 366 runs
(`eval_detour/`). *strict* = no violation **and** the car reached the sign's zone
(`plant2_pipeline/metrics.py`); the benchmark's own `sign_compliance` also credits runs
that never arrive.

| sign | runs | strict compliance | benchmark `sign_compliance` | reached zone | success | off_road | crash | mean dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 366 | **0.893** | 0.918 | 0.975 | 0.683 | 0.137 | 0.311 | 139.6 m |
| 4.2.1 right | 129 | 0.922 | 0.953 | | | | | |
| 4.2.2 left | 99 | 0.859 | 0.859 | | | | | |
| 4.2.3 either | 138 | 0.891 | 0.928 | | | | | |

**Stop benchmark (2.5)** — 42 held-out runs (`eval_stop/`):

| runs | sign compliance | success | off_road | crash | mean dist |
|---:|---:|---:|---:|---:|---:|
| 42 | **0.738** | 0.857 | 0.024 | 0.143 | 94.4 m |

Reproduce the tables from the per-episode files:
`plant2_detour_pipeline/summarize_detour.py joint_k3_augFIX_lr3e4_ep24_last`
(needs the original `eval/` layout) or any script over `*/episodes_plant2.jsonl`.

## Independent audits (`audit/`)

Run on this checkpoint after the eval, without trusting the benchmark flag.

* **Detour** (`detour_4_2_?_episodes.jsonl`, 40 runs per sign, 36 / 36 / 25 of them reached the obstacle, field `pass_geometry`):
  lateral offset at the moment of passing the obstacle is +2.98 m for 4.2.1,
  −3.14 m for 4.2.2 and +3.10 m for 4.2.3 — the sign of the offset follows the sign's
  required side without exception, and the lane index at the pass agrees with
  `sign_violations` in 97 / 97 episodes that reached the obstacle. Scenes without cones
  score the same as scenes with cones (0.895 vs 0.892): the model reads the sign, not the
  obstacle. Spawn is always in the sign's lane (366 / 366), so nothing is credited for free.
* **Stop** (`stop_episodes.jsonl`, field `stop_audit`): all 31 compliant runs braked to
  ≤ 0.24 m/s before the stop line (median 0.001 m/s). All 11 violations are **yield**
  violations (`yield_steps` = 33, `stopline_steps` = 0): `StopSign` inherits `YieldSign`
  and checks priority first, so **0.738 means "no priority violation"**, not "fraction of
  stops". The model does stop; the misses are about giving way to main-road traffic.

## Training recipe (`train/`)

* init: CARLA pretrain `stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`
  (scores 0.000 on detour and crashes in 100 % of runs — the whole skill comes from FT)
* data: joint split (`/tmp/joint_split_fx` on antonov, rebuildable with the pipeline):
  1544 train routes (314 stop + 1230 detour, 355 005 samples) / 261 val (30 + 231),
  split **by scene** (0 of 1010 scenes leaked); dumped with `PLANT2_SIGN_RANGE_M=90`,
  `PLANT2_YLEFT=1`, pose jitter ±1 m / ±5° (`PLANT2_AUG_TRANSLATION_M=1`,
  `PLANT2_AUG_ROTATION_DEG=5`); path target rebuilt offline with
  `scripts/plant2_ft_pipeline/data/fix_route_target.py`
* hydra (`hydra_config.yaml`, `hydra_overrides.yaml`): lr 3e-4, cosine with 10 % warm-up,
  batch 512, 24 epochs, AdamW wd 0.1, grad-clip 1.0, `augment=True`, loss weights
  path 5 / wp 1 / speed 1, speed-class weights `[15, 15, 10, 1, …]`,
  `sign_range_m=30` in the dataset (the dump holds 90 m), no oversampling, no route dropout
* launcher: `plant2_detour_pipeline/launch_joint_one.sh k3_augFIX_lr3e4_ep24 <gpu> 3e-4 24`
* ≈ 80 min on one A100, started 2026-08-29 22:31; curve in `metrics.csv`
  (`val/loss_path` 0.463 → 0.303, `val/loss_egospeed` 2.3 → 11.0)
* code state at train time: plant2 submodule `90d0c09` plus the working-tree diff recorded
  in `git_info.txt` (committed to the submodule together with this directory)

## Evaluation

* detour: `plant2_detour_pipeline/run_eval.sh` → `eval/<tag>/fv_fast`, table with
  `plant2_detour_pipeline/summarize_detour.py <tag>`
* stop: `plant2_stop_pipeline_hyp5/run_eval5.sh` (the `priority_bench` runner)
* inference must run with `PLANT2_SIGN_RANGE_M=90` and `PLANT2_YLEFT=1`, i.e. as the data
  was dumped; the adapter renders the BEV at dump resolution (256 px / 64 m, centre-cropped
  to 128 px)

## Caveats

* selecting `best` by `val/loss_all` is unusable here: the `best` checkpoints of the sibling
  runs k4–k6 reach stop compliance 1.000 with success 0.000 (the car barely moves)
* detour `success` is low even for the privileged expert on this catalog; read distance,
  reached-zone and off_road next to compliance
* the stop runner increments `sign_violations` twice per step
  (`priority_bench/run_benchmark.py`); compliance (zero vs non-zero) is unaffected
* the later 7-family run `joint7` (oracle scenes, sign radius 30 m at inference) regressed
  to stop 0.107 / detour 0.014 — see `plant2_pipeline/runs/joint7/eval/summary.txt`;
  `joint8` (90 m on both sides, sign-keyed oversampling) got as far as the cache prefill
  (`plant2_pipeline/runs/joint8/REUSED.md`)
