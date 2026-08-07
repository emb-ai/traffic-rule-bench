#!/usr/bin/env bash
# 7× PlanT2 2.5-only FT: H1 / H2 / H1+H2 / H5 matrix, 30 epochs.
# Durable: call via nohup/flock; each job is a background subshell with its own GPU.
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
LOGDIR="$CT/logs_pipeline_2p5_hyp"
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
test -f "$PLAN_T/lit_module.py"

# GPU|lr|hyp|addon|extra hydra overrides (space-separated, quoted later)
# H1: path_weight=0 forecastLoss=0 speed_weight=5
# H2: speed_class_weights=[15,1×7], default path/forecast/speed weights
# H1+H2: H1 loss weights + class weights
# H5: augment=False (uses unaugmented cache keys — no rebuild)
declare -a JOBS=(
  "0|1e-4|h1|fvexp30_2p5_h1_path0_sw5_lr1e4|model.waypoints.path_weight=0 model.pre_training.forecastLoss_weight=0 model.waypoints.speed_weight=5 model.training.augment=True"
  "1|1e-5|h1|fvexp30_2p5_h1_path0_sw5_lr1e5|model.waypoints.path_weight=0 model.pre_training.forecastLoss_weight=0 model.waypoints.speed_weight=5 model.training.augment=True"
  "2|1e-4|h2|fvexp30_2p5_h2_cw15_lr1e4|model.training.speed_class_weights=[15,1,1,1,1,1,1,1] model.training.augment=True"
  "3|1e-5|h2|fvexp30_2p5_h2_cw15_lr1e5|model.training.speed_class_weights=[15,1,1,1,1,1,1,1] model.training.augment=True"
  "4|1e-4|h1h2|fvexp30_2p5_h1h2_path0_sw5_cw15_lr1e4|model.waypoints.path_weight=0 model.pre_training.forecastLoss_weight=0 model.waypoints.speed_weight=5 model.training.speed_class_weights=[15,1,1,1,1,1,1,1] model.training.augment=True"
  "5|1e-5|h1h2|fvexp30_2p5_h1h2_path0_sw5_cw15_lr1e5|model.waypoints.path_weight=0 model.pre_training.forecastLoss_weight=0 model.waypoints.speed_weight=5 model.training.speed_class_weights=[15,1,1,1,1,1,1,1] model.training.augment=True"
  "6|1e-5|h5|fvexp30_2p5_h5_noaug_lr1e5|model.training.augment=False"
)

{
  echo "============================================================"
  echo "PlanT2 2.5 hyp FT sweep (H1/H2/H1+H2/H5)  $(date -Is)"
  echo "  SPLIT    = $SPLIT"
  echo "  DS_LOCAL = $DS_LOCAL"
  echo "  CACHE_GB = $CACHE_SIZE_GB"
  echo "  BS       = $BATCH_SIZE  epochs=$MAX_EPOCHS  workers=$NUM_WORKERS"
  echo "  jobs     = ${#JOBS[@]}"
  echo "============================================================"
} | tee -a "$LOGDIR/launch_ft.log"

pids=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r GPU LR HYP ADDON EXTRA <<<"$spec"
  LR_TAG=$(echo "$LR" | tr -d '-')
  LOG="$LOGDIR/ft_${HYP}_lr${LR_TAG}.log"
  # disambiguate when hyp repeats across LRs already encoded in filename
  if [[ "$HYP" == "h1h2" ]]; then
    LOG="$LOGDIR/ft_h1h2_lr${LR_TAG}.log"
  elif [[ "$HYP" == "h5" ]]; then
    LOG="$LOGDIR/ft_h5_noaug_lr${LR_TAG}.log"
  fi
  mkdir -p "$PLAN_T/checkpoints_ft/$ADDON"
  # shellcheck disable=SC2086
  (
    set -euo pipefail
    cd "$PLAN_T"
    export CUDA_VISIBLE_DEVICES=$GPU
    export LEARNING_RATE=$LR
    export CHECKPOINT_ADDON=$ADDON
    export DS_LOCAL CACHE_SIZE_GB MAX_EPOCHS BATCH_SIZE NUM_WORKERS
    export CKPT_EVERY_N_EPOCHS LR_SCHEDULER WARMUP_RATIO SEED WANDB_MODE
    export PYTHONNOUSERSITE=1
    RUN_DIR="outputs/PlanT2_train/${ADDON}/$(date -u +%Y%m%d_%H%M%S)_g${GPU}"
    echo "FT_START $(date -Is) gpu=$GPU lr=$LR hyp=$HYP addon=$ADDON extra=$EXTRA ds_local=$DS_LOCAL run_dir=$RUN_DIR" | tee "$LOG"
    # Short hydra.run.dir: override_dirname with many loss-weight keys exceeds NAME_MAX.
    # EXTRA is a space-separated list of hydra overrides
    # shellcheck disable=SC2086
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
      model.training.augment_parked=False \
      '+model.training.filter_routes=False' \
      $EXTRA \
      "model.training.log_path=$(hydra_esc "$PLAN_T/log/ft_${ADDON}_${SEED}")" \
      "expname=ft_${ADDON}" \
      "wandb_name=ft_${ADDON}_${SEED}" \
      "hydra.run.dir=$(hydra_esc "$RUN_DIR")" \
      2>&1 | tee -a "$LOG"
    echo "FT_EXIT=${PIPESTATUS[0]} $(date -Is)" | tee -a "$LOG"
  ) &
  pids+=($!)
  echo "started FT gpu=$GPU lr=$LR hyp=$HYP addon=$ADDON pid=${pids[-1]} log=$LOG" | tee -a "$LOGDIR/launch_ft.log"
done

printf '%s\n' "${pids[@]}" > "$LOGDIR/ft_pids.txt"
{
  echo "FT_PIDS=${pids[*]}"
  echo "FT_LAUNCHED $(date -Is)"
} | tee -a "$LOGDIR/launch_ft.log"

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "FT_ALL_DONE fail=$fail $(date -Is)" | tee -a "$LOGDIR/launch_ft.log"
exit "$fail"
