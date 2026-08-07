#!/usr/bin/env bash
# Launch 7 PlanT2 fine-tunes on GPUs 0-6 (spatial PDD signs dump).
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

PLAN_T="$TRB_ROOT/plant2/PlanT"
RUN="$SHEPELEV/collected_trajectories/run_plant2_finetune.sh"
SPLIT="${SPLIT:-$SHEPELEV/plant2_l1_fv_experts_split_signs}"
PY="${PYTHON:-$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python}"

export SPLIT
export DS="${DS:-$SPLIT/train}"
export DS_VAL="${DS_VAL:-$SPLIT/val}"
export DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_spatial_aug}"
export CACHE_SIZE_GB="${CACHE_SIZE_GB:-1800}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine_warmup}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export BATCH_SIZE="${BATCH_SIZE:-1344}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export MAX_EPOCHS="${MAX_EPOCHS:-30}"
export CKPT_EVERY_N_EPOCHS="${CKPT_EVERY_N_EPOCHS:-5}"
export SEED="${SEED:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHON="$PY"

mkdir -p "$PLAN_T/log" "$PLAN_T/checkpoints_ft" /tmp

declare -a JOBS=(
  "0|1e-6|fvexp30_spatial_lr1e6"
  "1|5e-6|fvexp30_spatial_lr5e6"
  "2|1e-5|fvexp30_spatial_lr1e5"
  "3|3e-5|fvexp30_spatial_lr3e5"
  "4|5e-5|fvexp30_spatial_lr5e5"
  "5|7e-5|fvexp30_spatial_lr7e5"
  "6|1e-4|fvexp30_spatial_lr1e4"
)

echo "============================================================"
echo "PlanT2 spatial LR sweep  $(date -Is)"
echo "  SPLIT    = $SPLIT"
echo "  DS_LOCAL = $DS_LOCAL"
echo "  CACHE_GB = $CACHE_SIZE_GB"
echo "  BS       = $BATCH_SIZE  epochs=$MAX_EPOCHS  workers=$NUM_WORKERS"
echo "============================================================"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r GPU LR ADDON <<<"$spec"
  LR_TAG=$(echo "$LR" | tr -d '-')
  SESSION="arbelyaev-ft-spatial-lr${LR_TAG}"
  LOG="/tmp/plant2_ft_spatial_lr${LR_TAG}.log"
  CKPT_DIR="$PLAN_T/checkpoints_ft/$ADDON"
  mkdir -p "$CKPT_DIR"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "WARN: session $SESSION already exists — leaving it"
    continue
  fi

  tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail
cd '$PLAN_T'
export CUDA_VISIBLE_DEVICES=$GPU
export LEARNING_RATE=$LR
export CHECKPOINT_ADDON=$ADDON
export SPLIT='$SPLIT'
export DS='$DS'
export DS_VAL='$DS_VAL'
export DS_LOCAL='$DS_LOCAL'
export CACHE_SIZE_GB=$CACHE_SIZE_GB
export MAX_EPOCHS=$MAX_EPOCHS
export BATCH_SIZE=$BATCH_SIZE
export NUM_WORKERS=$NUM_WORKERS
export CKPT_EVERY_N_EPOCHS=$CKPT_EVERY_N_EPOCHS
export LR_SCHEDULER=$LR_SCHEDULER
export WARMUP_RATIO=$WARMUP_RATIO
export SEED=$SEED
export WANDB_MODE=$WANDB_MODE
export PYTHONNOUSERSITE=1
export PYTHON='$PY'
echo \"FT_START \$(date -Is) gpu=$GPU lr=$LR addon=$ADDON\" | tee -a '$LOG'
bash '$RUN' 2>&1 | tee -a '$LOG'
echo \"FT_EXIT=\$? \$(date -Is)\" | tee -a '$LOG'
exec bash
"
  echo "started session=$SESSION gpu=$GPU lr=$LR addon=$ADDON log=$LOG ckpt=$CKPT_DIR"
done

echo "All spatial LR sweep jobs launched (or already running)."
tmux ls | grep -E 'arbelyaev-ft-spatial-lr' || true
