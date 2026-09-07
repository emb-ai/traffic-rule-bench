# Best experiment so far: `joint_k3_augFIX_lr3e4_ep24`, checkpoint `last`

One PlanT2 model fine-tuned jointly on stop-sign (2.5) and detour (4.2.1 / 4.2.2 / 4.2.3)
expert replays. Trained 2026-08-29, ~80 min on one A100.

| benchmark | runs | compliance | success | note |
|---|---:|---:|---:|---|
| detour 4.2.x (122-scene catalog) | 366 | **0.893** strict (0.918 raw) | 0.683 | strict = no violation *and* the zone was reached |
| stop 2.5 | 42 | **0.738** | 0.857 | all 11 misses are yield (priority) violations; the car does stop |

Everything below is detailed in
[`plant2_detour_pipeline/results/joint_k3_augFIX_lr3e4_ep24_last/README.md`](plant2_detour_pipeline/results/joint_k3_augFIX_lr3e4_ep24_last/README.md).

## What it consists of

| piece | where |
|---|---|
| checkpoint (446 MB, not in git) | `plant2/PlanT/checkpoints_ft/joint_k3_augFIX_lr3e4_ep24/last_ft_joint_k3_augFIX_lr3e4_ep24_1.ckpt` on antonov, sha256 `eb62ae02…54c2` |
| starting point | CARLA pretrain `stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt` |
| training data | joint split: 1544 train / 261 val routes (stop 314 + 30, detour 1230 + 231), split by scene, dumped with sign radius 90 m, y-left, pose jitter ±1 m / ±5°, path target rebuilt by `scripts/plant2_ft_pipeline/data/fix_route_target.py` |
| hyper-parameters | lr 3e-4 cosine + 10 % warm-up, batch 512, 24 epochs, AdamW wd 0.1, augmentation on, `sign_range_m=30` in the dataset — `results/…/train/hydra_config.yaml` |
| code | this commit: `plant2` submodule (dataset / model knobs), `metadrive` submodule (BEV pose jitter), `pdd-bench` adapter (`PLANT2_YLEFT`, BEV at dump resolution), `scripts/plant2_ft_pipeline` |
| launcher | `plant2_detour_pipeline/launch_joint_one.sh` |
| eval outputs, audits, training curve | `plant2_detour_pipeline/results/joint_k3_augFIX_lr3e4_ep24_last/{eval_detour,eval_stop,audit,train}` |

## How to run

On antonov, from the repo root. Interpreter and offline flags for every command:

```bash
PY=/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT=$PWD/plant2/PlanT/checkpoints_ft/joint_k3_augFIX_lr3e4_ep24/last_ft_joint_k3_augFIX_lr3e4_ep24_1.ckpt
```

Inference must see signs the way the data was dumped: `PLANT2_SIGN_RANGE_M=90`
(and `PLANT2_YLEFT=1`, which is the default). Checkpoint paths must be absolute.

### Evaluate the checkpoint

```bash
# detour: 366 runs on the 122-scene catalog + 5 sample gifs, ~1.5 h on one GPU
PLANT2_SIGN_RANGE_M=90 bash plant2_detour_pipeline/run_eval.sh k3_last "$CKPT" 7
$PY plant2_detour_pipeline/summarize_detour.py k3_last      # strict compliance table

# stop: 42 held-out runs, the harness behind the 0.738 number
PLANT2_SIGN_RANGE_M=90 PLANT2_DUMP_SIGN_CLASSES=2.5 CUDA_VISIBLE_DEVICES=7 $PY -u \
  pdd-bench/scripts/per_sign_bench/priority_bench/eval_pipeline.py \
  --policies plant2 --model-paths "plant2:$CKPT" \
  --manifest stop_data/output/ts_test/real_manifest.jsonl --scenes-root stop_data/scenes \
  --out-dir plant2_stop_pipeline_hyp5/eval/k3_last --plant2-action-mode pid --jobs 8 --backends sumo
# -> plant2_stop_pipeline_hyp5/eval/k3_last/reports/report_cumulative.md

# gifs on either benchmark
bash plant2_detour_pipeline/make_gifs.sh detour "$CKPT" plant2_detour_pipeline/gifs_test/k3 7 5
```

### Retrain the same recipe

The joint split lived in `/tmp/joint_split_fx` and was lost when `/tmp` was wiped, so the
data has to be rebuilt first (a few hours of dumping):

1. dump stop and detour expert replays with
   `pdd-bench/scripts/per_sign_bench/expert_replay_for_plant2.py` under
   `PLANT2_DUMP_SIGN_CLASSES=4.2.1,4.2.2,4.2.3,2.5 PLANT2_SIGN_RANGE_M=90 PLANT2_YLEFT=1
   PLANT2_AUG_TRANSLATION_M=1 PLANT2_AUG_ROTATION_DEG=5` — `plant2_pipeline/stages.py`
   (`dump`) shows the exact call, sharded over CPUs;
2. split by scene (`plant2_pipeline` `split` stage, or
   `scripts/plant2_ft_pipeline/data/make_train_val_split_fv_experts_signs.py`), then
   `$PY scripts/plant2_ft_pipeline/data/fix_route_target.py --src <split> --out <split_fx>`;
3. warm the cache: `$PY scripts/plant2_ft_pipeline/data/prefill_diskcache.py` (otherwise the
   first epoch reads everything from NFS);
4. train: `JOINT_SPLIT=<split_fx> bash plant2_detour_pipeline/launch_joint_one.sh
   k3_augFIX_lr3e4_ep24 <gpu> 3e-4 24` — checkpoints land in
   `plant2/PlanT/checkpoints_ft/joint_k3_augFIX_lr3e4_ep24/`; take `last`, not `best`.

For a new scene set the whole loop is one command:
`$PY -m plant2_pipeline all --name <run> --gpu <gpu>` (see `plant2_pipeline/README.md`);
that is what `joint7` / `joint8` used, on the 7-family oracle collection.
