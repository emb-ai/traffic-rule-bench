#!/usr/bin/env bash
# Fine-tune PlanT2 on 2.5-only subset-split, lr=1e-4 and 1e-5, 30 epochs.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

PLAN_T="$TRB_ROOT/plant2/PlanT"
CKPT="$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt"
SPLIT="${SPLIT:-$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5}"
PY="${PYTHON:-$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python}"
SHIM="$SHEPELEV/collected_trajectories/plant2_py_shims/run_lit_finetune.py"

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

hydra_esc() { printf '%s' "$1" | sed 's/=/\\=/g'; }

test -f "$SPLIT/split_meta.json"
test -d "$DS/data"
test -d "$DS_VAL/data"
test -f "$CKPT"
test -f "$SHIM"
mkdir -p "$PLAN_T/log" "$PLAN_T/checkpoints_ft" /tmp

declare -a JOBS=(
  "0|1e-4|fvexp30_spatial_2p5_lr1e4"
  "1|1e-5|fvexp30_spatial_2p5_lr1e5"
)

echo "============================================================"
echo "PlanT2 2.5-only FT  $(date -Is)"
echo "  SPLIT    = $SPLIT"
echo "  DS_LOCAL = $DS_LOCAL"
echo "  CACHE_GB = $CACHE_SIZE_GB"
echo "  BS       = $BATCH_SIZE  epochs=$MAX_EPOCHS  workers=$NUM_WORKERS"
echo "============================================================"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r GPU LR ADDON <<<"$spec"
  LR_TAG=$(echo "$LR" | tr -d '-')
  SESSION="arbelyaev-ft-2p5-lr${LR_TAG}"
  LOG="/tmp/plant2_ft_2p5_lr${LR_TAG}.log"
  CKPT_DIR="$PLAN_T/checkpoints_ft/$ADDON"
  mkdir -p "$CKPT_DIR"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "WARN: session $SESSION already exists — leaving it"
    continue
  fi

  # Match spatial FT: augment on, parked off. Split is pre-filtered → filter_routes=False.
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
'$PY' -u '$SHIM' \\
  resume=True \\
  'resume_path=$(hydra_esc "$CKPT")' \\
  gpus=1 \\
  use_caching=True \\
  'lr_scheduler=$LR_SCHEDULER' \\
  'warmup_ratio=$WARMUP_RATIO' \\
  'model.training.learning_rate=$LR' \\
  'model.training.max_epochs=$MAX_EPOCHS' \\
  'model.training.batch_size=$BATCH_SIZE' \\
  'model.training.num_workers=$NUM_WORKERS' \\
  model.training.augment=True \\
  model.training.augment_parked=False \\
  '+model.training.filter_routes=False' \\
  'model.training.log_path=$(hydra_esc "$PLAN_T/log/ft_${ADDON}_${SEED}")' \\
  'expname=ft_${ADDON}' \\
  'wandb_name=ft_${ADDON}_${SEED}' \\
  2>&1 | tee -a '$LOG'
echo \"FT_EXIT=\$? \$(date -Is)\" | tee -a '$LOG'
exec bash
"
  echo "started session=$SESSION gpu=$GPU lr=$LR addon=$ADDON log=$LOG ckpt=$CKPT_DIR"
done

echo "Done launching."
tmux ls | grep -E 'arbelyaev-ft-2p5-lr' || true
