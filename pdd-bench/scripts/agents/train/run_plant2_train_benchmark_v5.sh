#!/usr/bin/env bash
# Supervised Plant2 on benchmark_sign_trajectories_v5.
#
# Episode .pt files are produced under benchmark_sign_trajectories_v* (collection).
# This script writes checkpoints/logs under OUT_DIR (default plant2_supervised_benchmark_v5).
#
# Hyperparams (defaults match common v5 request):
#   - 4 GPUs via DataParallel: pass --devices 0,1,2,3 (logical IDs after CUDA_VISIBLE_DEVICES).
#   - Global batch 768: BATCH_PER_GPU=192 (768/4); lr = 1e-5 * sqrt(768/128) = 1e-5 * sqrt(6).
#
# Override: PY, DATA_DIR, CKPT, OUT_DIR, EPOCHS, BATCH_PER_GPU, LR, CUDA_VISIBLE_DEVICES.
set -euo pipefail

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-pdd-bench/outputs/benchmark_sign_trajectories_v5}"
CKPT="${CKPT:-epoch%3D029_final_3.ckpt}"
OUT_DIR="${OUT_DIR:-pdd-bench/outputs/plant2_supervised_benchmark_v5}"
SCRIPT="${SCRIPT:-pdd-bench/scripts/agents/train/train_plant2_from_carl_trajectories.py}"
LOG="${LOG:-$OUT_DIR/train_console.log}"

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "$OUT_DIR"
# script assumes CWD = repo root

EPOCHS="${EPOCHS:-30}"
BATCH_PER_GPU="${BATCH_PER_GPU:-192}"
if [[ -n "${LR-}" ]]; then
  EFF_LR="$LR"
else
  EFF_LR="$("$PY" -c "import math; print(1e-5 * math.sqrt(6.0))")"
fi

echo "======== $(date -Is) plant2 v5 train  epochs=${EPOCHS} batch_per_gpu=${BATCH_PER_GPU} eff_bs=$((BATCH_PER_GPU * 4)) lr=${EFF_LR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ========" \
  | tee -a "$LOG"

exec "$PY" -u "$SCRIPT" \
  --data-dir "$DATA_DIR" \
  --checkpoint_file "$CKPT" \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_PER_GPU" \
  --lr "$EFF_LR" \
  --devices 0,1,2,3 \
  2>&1 | tee -a "$LOG"
