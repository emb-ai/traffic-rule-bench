#!/usr/bin/env bash
# Supervised Plant2 on benchmark_sign_trajectories_v4.
# - GPUs 0,1 unavailable: use only physical 2..7 via CUDA_VISIBLE_DEVICES.
# - Default batch 768 (6×128). AdamW: if LR is unset, use sqrt rule
#     lr(bs) = BASE_LR * sqrt(bs / BASE_LR_REF_BS)   (not linear in batch).
#   Example: BASE_LR=1e-5 @ 128 → 768 gives 1e-5*sqrt(6) ≈ 2.45e-5.
# - On OOM, retry BATCH_FALLBACK (default 384); LR is recomputed for that bs unless LR is set.
# - Override: BATCH_FIRST, LR (fixed for all attempts), BASE_LR, BASE_LR_REF_BS, BATCH_FALLBACK.
set -euo pipefail

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-pdd-bench/outputs/benchmark_sign_trajectories_v4}"
CKPT="${CKPT:-epoch%3D029_final_3.ckpt}"
OUT_DIR="${OUT_DIR:-pdd-bench/outputs/plant2_supervised_benchmark_v4}"
SCRIPT="${SCRIPT:-pdd-bench/scripts/agents/train/train_plant2_from_carl_trajectories.py}"
LOG="${LOG:-$OUT_DIR/train_console.log}"

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"

mkdir -p "$OUT_DIR"
# script assumes CWD = repo root

BASE_LR="${BASE_LR:-1e-5}"
BASE_LR_REF_BS="${BASE_LR_REF_BS:-128}"

run_train() {
  local bs="$1"
  local eff_lr
  if [[ -n "${LR+x}" && -n "$LR" ]]; then
    eff_lr="$LR"
  else
    eff_lr="$("$PY" -c "import math; bs=float('$bs'); lr0=float('$BASE_LR'); b0=float('$BASE_LR_REF_BS'); print(lr0 * math.sqrt(max(bs, 1.0) / b0))")"
  fi
  echo "======== $(date -Is) batch_size=${bs} lr=${eff_lr} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ========" \
    | tee -a "$LOG"
  "$PY" -u "$SCRIPT" \
    --data-dir "$DATA_DIR" \
    --checkpoint_file "$CKPT" \
    --output_dir "$OUT_DIR" \
    --epochs "${EPOCHS:-10}" \
    --batch_size "$bs" \
    --lr "$eff_lr" \
    --log-every-n-steps "${LOG_EVERY:-100}" \
    --ckpt-every-n-epochs "${CKPT_EVERY:-1}" \
    --device cuda \
    2>&1 | tee -a "$LOG"
}

set +e
run_train "${BATCH_FIRST:-768}"
code=$?
set -e
if [[ "${SKIP_BATCH_FALLBACK:-0}" != "1" && "$code" -ne 0 ]]; then
  echo "======== $(date -Is) first run failed exit=$code; retry batch ${BATCH_FALLBACK:-384} ========" \
    | tee -a "$LOG"
  run_train "${BATCH_FALLBACK:-384}"
  code=$?
fi
echo "======== $(date -Is) finished exit=$code ========" | tee -a "$LOG"
exit "$code"
