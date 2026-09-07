#!/usr/bin/env bash
# 5 stop-sign (2.5) hypothesis FT runs, one per free GPU (3-7), same
# split/ckpt/LR/epochs as the classemb baseline (0.548 @ ep10 lr3e4) so deltas
# are attributable to each hypothesis's mechanism, not confounds.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
PY="/home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python"
SPLIT="$TRB_ROOT/plant2_stop_pipeline_signfix/plant2_l1_stop_split"
CKPT0="$TRB_ROOT/stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt"
WORK="$TRB_ROOT/plant2_stop_pipeline_hyp5"
LOGDIR="$WORK/logs"

LR="3e-4"
EPOCHS=10
BATCH_SIZE=512
NUM_WORKERS=4
CKPT_EVERY=5
CACHE_GB=50

mkdir -p "$LOGDIR"

launch_job() {
  local name="$1" gpu="$2" addon="$3"; shift 3
  local extra_hydra=("$@")
  local ds_local="/tmp/plant2_ds_cache_hyp_${name}"
  local log="$LOGDIR/train_${name}.log"
  # Explicit short run-dir: Hydra's default dir name concatenates every
  # override key=value pair and overflows the 255-char filename limit once
  # a job has more than ~2 extra overrides.
  local hydra_run_dir="$TRB_ROOT/plant2/PlanT/outputs/PlanT2_train/${addon}"
  mkdir -p "$ds_local"

  local hydra_args=()
  for o in "${extra_hydra[@]}"; do
    hydra_args+=(--hydra-override "$o")
  done

  echo "[launch] $name gpu=$gpu addon=$addon extra=${extra_hydra[*]:-none}"
  (
    cd "$PIPELINE"
    "$PY" -u train/run_plant2_finetune.py \
      --split "$SPLIT" \
      --learning-rate "$LR" \
      --checkpoint-addon "$addon" \
      --cuda-device "$gpu" \
      --ds-local "$ds_local" \
      --cache-size-gb "$CACHE_GB" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-epochs "$EPOCHS" \
      --ckpt-every-n-epochs "$CKPT_EVERY" \
      --augment --no-filter-routes \
      --resume-ckpt "$CKPT0" \
      --wandb-mode offline \
      --python "$PY" \
      --hydra-override "user.working_dir=$TRB_ROOT/plant2" \
      --hydra-run-dir "$hydra_run_dir" \
      "${hydra_args[@]}" \
      --log "$log" \
      >>"$log" 2>&1
  ) &
  echo $! > "$LOGDIR/train_${name}.pid"
}

launch_job h1_persist 3 stop_hyp_h1_persist \
  "model.training.sign_ignore_affects_ego=True" "model.training.sign_range_m=45"

launch_job h2_reweight 4 stop_hyp_h2_reweight \
  "model.pre_training.forecastLoss_weight=0" "model.waypoints.path_weight=0" "model.waypoints.speed_weight=5"

launch_job h3_memory 5 stop_hyp_h3_memory \
  "model.training.sign_memory_frames=10"

launch_job h4_auxloss 6 stop_hyp_h4_auxloss \
  "model.training.aux_sign_presence_weight=1.0"

launch_job h5_pool 7 stop_hyp_h5_pool \
  "model.training.sign_pool_token=True"

echo "All 5 jobs launched. PIDs:"
cat "$LOGDIR"/train_*.pid
