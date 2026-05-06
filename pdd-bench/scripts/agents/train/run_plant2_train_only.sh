#!/usr/bin/env bash
# Plant2 supervised training only (repack data must exist). Logs to TRAIN_OUT/train_console.log
#
#   conda activate plant2   # optional if PLANT2_PY points to that env
#   bash run_plant2_train_only.sh
#
set -euo pipefail

PLANT2_PY="${PLANT2_PY:-python}"
DATA_DIR="${DATA_DIR:-pdd-bench/outputs/plant2_repack_mini}"
INIT_CKPT="${INIT_CKPT:-plant2/models/epoch%3D029_final_3.ckpt}"
TRAIN_OUT="${TRAIN_OUT:-pdd-bench/outputs/plant2_supervised_from_mini}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# Default LR as decimal — safe for bash; override with export LR=1e-5 if desired
LR="${LR:-0.00001}"

TRAIN_SCRIPT="pdd-bench/scripts/agents/train/train_plant2_from_carl_trajectories.py"
LOG_FILE="${LOG_FILE:-$TRAIN_OUT/train_console.log}"

mkdir -p "$TRAIN_OUT"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
# So `tee` log files get lines immediately (Python block-buffers pipes otherwise).
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

{
  echo "=== $(date -Is) Plant2 supervised train ==="
  echo "PLANT2_PY=$PLANT2_PY"
  echo "DATA_DIR=$DATA_DIR"
  echo "INIT_CKPT=$INIT_CKPT"
  echo "TRAIN_OUT=$TRAIN_OUT"
  echo "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE LR=$LR"
  echo "Metrics will be under: $TRAIN_OUT/metrics_csv/ and $TRAIN_OUT/tensorboard/"
  echo "---"
} | tee -a "$LOG_FILE"

set +e
"$PLANT2_PY" "$TRAIN_SCRIPT" \
  --data-dir "$DATA_DIR" \
  --checkpoint_file "$INIT_CKPT" \
  --output_dir "$TRAIN_OUT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --log-every-n-steps 100 \
  --ckpt-every-n-epochs 1 \
  2>&1 | tee -a "$LOG_FILE"
EXIT=$?
set -e

echo "Training exit code: $EXIT" | tee -a "$LOG_FILE"
exit "$EXIT"
