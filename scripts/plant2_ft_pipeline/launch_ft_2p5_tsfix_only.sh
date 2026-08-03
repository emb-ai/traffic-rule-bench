#!/usr/bin/env bash
# Launch two PlanT2 FT jobs on the extracted 2.5 tsfix diskcache.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

CT="$PIPELINE_DIR"
PLAN_T="$TRB_ROOT/plant2/PlanT"
SPLIT="$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5"
CKPT0="$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt"
SHIM="$PIPELINE_DIR/plant2_py_shims/run_lit_finetune.py"
PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
LOGDIR="$CT/logs_pipeline_2p5_tsfix"
mkdir -p "$LOGDIR" "$PLAN_T/checkpoints_ft"

hydra_esc() { printf '%s' "$1" | sed 's/=/\\=/g'; }

export DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_2p5_tsfix}"
export CACHE_SIZE_GB="${CACHE_SIZE_GB:-400}"
export DS="$SPLIT/train"
export DS_VAL="$SPLIT/val"
export SPLIT
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export MAX_EPOCHS="${MAX_EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-1344}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CKPT_EVERY_N_EPOCHS="${CKPT_EVERY_N_EPOCHS:-5}"
export LR_SCHEDULER=cosine_warmup
export WARMUP_RATIO=0.1
export SEED=1

test -d "$DS_LOCAL"
test -f "$CKPT0"
test -f "$SHIM"

declare -a JOBS=(
  "0|1e-4|fvexp30_spatial_2p5_tsfix_lr1e4"
  "1|1e-5|fvexp30_spatial_2p5_tsfix_lr1e5"
)

pids=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r GPU LR ADDON <<<"$spec"
  LR_TAG=$(echo "$LR" | tr -d '-')
  LOG="/tmp/plant2_ft_2p5_tsfix_lr${LR_TAG}.log"
  PIPE_LOG="$LOGDIR/ft_lr${LR_TAG}.log"
  mkdir -p "$PLAN_T/checkpoints_ft/$ADDON"
  (
    set -euo pipefail
    cd "$PLAN_T"
    export CUDA_VISIBLE_DEVICES=$GPU
    export LEARNING_RATE=$LR
    export CHECKPOINT_ADDON=$ADDON
    export DS_LOCAL CACHE_SIZE_GB MAX_EPOCHS BATCH_SIZE NUM_WORKERS
    export CKPT_EVERY_N_EPOCHS LR_SCHEDULER WARMUP_RATIO SEED WANDB_MODE
    export PYTHONNOUSERSITE=1
    echo "FT_START $(date -Is) gpu=$GPU lr=$LR addon=$ADDON ds_local=$DS_LOCAL" | tee "$LOG" | tee "$PIPE_LOG"
    "$PY" -u "$SHIM" \
      resume=True \
      "resume_path=$(hydra_esc "$CKPT0")" \
      gpus=1 \
      use_caching=True \
      "lr_scheduler=$LR_SCHEDULER" \
      "warmup_ratio=$WARMUP_RATIO" \
      "model.training.learning_rate=$LR" \
      "model.training.max_epochs=$MAX_EPOCHS" \
      "model.training.batch_size=$BATCH_SIZE" \
      "model.training.num_workers=$NUM_WORKERS" \
      model.training.augment=True \
      model.training.augment_parked=False \
      '+model.training.filter_routes=False' \
      "model.training.log_path=$(hydra_esc "$PLAN_T/log/ft_${ADDON}_${SEED}")" \
      "expname=ft_${ADDON}" \
      "wandb_name=ft_${ADDON}_${SEED}" \
      2>&1 | tee -a "$LOG" | tee -a "$PIPE_LOG"
    echo "FT_EXIT=${PIPESTATUS[0]} $(date -Is)" | tee -a "$LOG" | tee -a "$PIPE_LOG"
  ) &
  pids+=($!)
  echo "started FT gpu=$GPU lr=$LR addon=$ADDON pid=${pids[-1]} log=$LOG"
done

printf '%s\n' "${pids[@]}" > "$LOGDIR/ft_pids.txt"
echo "FT_PIDS=${pids[*]}"
# Do not wait — caller/watcher handles eval.
wait || true
