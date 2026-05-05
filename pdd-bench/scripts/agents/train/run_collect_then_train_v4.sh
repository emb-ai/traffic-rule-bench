#!/usr/bin/env bash
# run_collect_then_train_v4.sh — wait for run_collect_v4_parallel.sh to finish,
# run validate_v4.py, and start training. Use this as the overnight wrapper:
#   nohup bash run_collect_then_train_v4.sh > .../auto_pipeline.log 2>&1 &
#
# Env vars (all optional):
#   PY                       (default plant2 python)
#   OUTPUT_DIR               (default pdd-bench/outputs/benchmark_sign_trajectories_v4)
#   CHECK_INTERVAL_SECS      (default 60)
#   MIN_EXPECTED_PT          (default 9000) — collection considered done if
#                            no collectors are alive AND >=this many .pt files.
#   EPOCHS                   (default 20)   — passed through to training.
#   BATCH_FIRST              (default 768)
#   AUTO_TRAIN               (default 1) — if 0, skip training and stop after
#                            validate.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARBELYAEV_SDC="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PY="${PY:-/home/jovyan/.mlspace/envs/plant2/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARBELYAEV_SDC/outputs/benchmark_sign_trajectories_v4}"
CHECK_INTERVAL_SECS="${CHECK_INTERVAL_SECS:-60}"
MIN_EXPECTED_PT="${MIN_EXPECTED_PT:-9000}"
EPOCHS="${EPOCHS:-20}"
BATCH_FIRST="${BATCH_FIRST:-768}"
AUTO_TRAIN="${AUTO_TRAIN:-1}"

LOG_DIR="$OUTPUT_DIR/_pipeline_logs"
mkdir -p "$LOG_DIR"

PIPE_LOG="$LOG_DIR/pipeline.log"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE_LOG"; }

log "OUTPUT_DIR=$OUTPUT_DIR"
log "Waiting for collection workers to finish (poll every ${CHECK_INTERVAL_SECS}s)..."

# Phase 1: poll until no collector workers alive AND .pt count >= threshold.
while :; do
  alive=$(pgrep -f "collect_benchmark_sign_trajectories.py" | wc -l)
  pt_count=$(ls "$OUTPUT_DIR"/*.pt 2>/dev/null | wc -l)
  log "alive=$alive  pt_count=$pt_count"
  if [[ "$alive" -eq 0 && "$pt_count" -ge "$MIN_EXPECTED_PT" ]]; then
    log "Collection complete (no live workers, $pt_count .pt files)."
    break
  fi
  if [[ "$alive" -eq 0 && "$pt_count" -lt "$MIN_EXPECTED_PT" ]]; then
    log "WARN: workers stopped but only $pt_count<$MIN_EXPECTED_PT .pt files. Sleeping anyway in case retry started later..."
    # Wait once more and re-check; if still zero alive, give up.
    sleep "$CHECK_INTERVAL_SECS"
    alive2=$(pgrep -f "collect_benchmark_sign_trajectories.py" | wc -l)
    if [[ "$alive2" -eq 0 ]]; then
      log "ERROR: collection appears stuck/dead at $pt_count .pt files. Exiting before train."
      exit 2
    fi
  fi
  sleep "$CHECK_INTERVAL_SECS"
done

# Phase 2: validate.
log "Running validate_v4.py ..."
VAL_LOG="$LOG_DIR/validate.log"
"$PY" "$SCRIPT_DIR/validate_v4.py" --data-dir "$OUTPUT_DIR" \
  > "$VAL_LOG" 2>&1 \
  && log "validate_v4.py PASS  (full log: $VAL_LOG)" \
  || log "validate_v4.py NON-ZERO exit (full log: $VAL_LOG)"

# Phase 3: training.
if [[ "$AUTO_TRAIN" != "1" ]]; then
  log "AUTO_TRAIN=0 -> skipping training. Pipeline done."
  exit 0
fi

log "Launching training (EPOCHS=$EPOCHS BATCH_FIRST=$BATCH_FIRST) ..."
TRAIN_LOG="$LOG_DIR/train.log"
EPOCHS="$EPOCHS" BATCH_FIRST="$BATCH_FIRST" \
  bash "$SCRIPT_DIR/run_plant2_train_benchmark_v4.sh" \
  > "$TRAIN_LOG" 2>&1
ec=$?
log "Training finished with exit=$ec  (full log: $TRAIN_LOG)"
exit "$ec"
