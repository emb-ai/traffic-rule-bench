#!/usr/bin/env bash
# run_collect_v5_parallel.sh — shard collect_benchmark_sign_trajectories.py
# across N worker processes by --only-codes, in REJECTION-SAMPLING mode.
#
# Differences vs v4:
#   * --success-target N: each worker collects until N successful (arrived)
#     trajectories per code, not a fixed manifest pass.
#   * --min-sign-distance 70: scenes whose ego→sign Euclidean distance at
#     spawn is < 70 m are rejected before the step loop begins.
#   * Failed-attempt .pt files go to $OUTPUT_DIR/_failed for offline diag.
#   * Each worker uses a distinct --shuffle-seed so workers exploring the
#     same code see independent orderings (only matters if codes are not
#     fully sharded).
#
# Usage:
#   bash run_collect_v5_parallel.sh                   # default 8 workers, 30 PGMap codes
#   N_WORKERS=10 SUCCESS_TARGET=300 bash run_collect_v5_parallel.sh
#   ONLY_CODES=4.2.1,4.2.2,4.2.3 bash run_collect_v5_parallel.sh
#
# Tunables:
#   N_WORKERS         (default 8)
#   PY                (default: python on PATH)
#   SAMPLED_DIR       (default benchmark_output/sampled_for_expert_v2)
#   OUTPUT_DIR        (default pdd-bench/outputs/benchmark_sign_trajectories_v5)
#   MAX_STEPS         (default 600)
#   GIFS_PER_SIGN     (default 3) — only render this many GIFs per sign
#   TRAFFIC_DENSITY   (default 0.1)
#   SUCCESS_TARGET    (default 300)
#   MIN_SIGN_DISTANCE (default 70)
#   MAX_ATTEMPTS_PER_SIGN (default 0 = auto-cap to manifest_rows*20)
#   ONLY_CODES        (override the full list, comma-separated)
#   LOG_DIR           (default $OUTPUT_DIR/_collect_logs)

set -euo pipefail

PY="${PY:-python}"
N_WORKERS="${N_WORKERS:-8}"

SAMPLED_DIR="${SAMPLED_DIR:-pdd-bench/scripts/per_sign_bench/benchmark_output/sampled_for_expert_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-pdd-bench/outputs/benchmark_sign_trajectories_v5}"
MAX_STEPS="${MAX_STEPS:-600}"
GIFS_PER_SIGN="${GIFS_PER_SIGN:-3}"
TRAFFIC_DENSITY="${TRAFFIC_DENSITY:-0.1}"
SUCCESS_TARGET="${SUCCESS_TARGET:-300}"
MIN_SIGN_DISTANCE="${MIN_SIGN_DISTANCE:-70}"
MAX_ATTEMPTS_PER_SIGN="${MAX_ATTEMPTS_PER_SIGN:-0}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/_collect_logs}"
FAILED_DIR="$OUTPUT_DIR/_failed"

# Default: 30 PGMap codes (skip 4 citymap codes 3.1/3.2/3.18.2/3.19 which
# still have the routing-fidelity M1 bug under investigation).
DEFAULT_CODES=(
  "2.1" "2.2" "2.3.1" "2.3.2" "2.3.3" "2.4" "2.5"
  "3.20"
  "3.24" "3.25" "3.27" "3.31"
  "4.2.1" "4.2.2" "4.2.3" "4.6"
  "5.11.1" "5.11.2" "5.12.1" "5.12.2"
  "5.13.1" "5.13.2" "5.13.3" "5.13.4"
  "5.14.1" "5.14.2" "5.14.3"
  "5.16"
  "5.31" "5.32"
)

if [[ -n "${ONLY_CODES:-}" ]]; then
  IFS=',' read -r -a CODES <<< "$ONLY_CODES"
else
  CODES=("${DEFAULT_CODES[@]}")
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$FAILED_DIR"

echo "[collect_v5_parallel] PY=$PY"
echo "[collect_v5_parallel] N_WORKERS=$N_WORKERS"
echo "[collect_v5_parallel] SAMPLED_DIR=$SAMPLED_DIR"
echo "[collect_v5_parallel] OUTPUT_DIR=$OUTPUT_DIR"
echo "[collect_v5_parallel] MAX_STEPS=$MAX_STEPS  GIFS_PER_SIGN=$GIFS_PER_SIGN"
echo "[collect_v5_parallel] SUCCESS_TARGET=$SUCCESS_TARGET  MIN_SIGN_DISTANCE=$MIN_SIGN_DISTANCE"
echo "[collect_v5_parallel] codes(${#CODES[@]}): ${CODES[*]}"
echo "[collect_v5_parallel] LOG_DIR=$LOG_DIR"
echo "[collect_v5_parallel] FAILED_DIR=$FAILED_DIR"

# Round-robin shard codes across N_WORKERS.
declare -a SHARDS
for ((i=0; i<N_WORKERS; i++)); do SHARDS[$i]=""; done
for ((i=0; i<${#CODES[@]}; i++)); do
  w=$((i % N_WORKERS))
  if [[ -z "${SHARDS[$w]}" ]]; then
    SHARDS[$w]="${CODES[$i]}"
  else
    SHARDS[$w]="${SHARDS[$w]},${CODES[$i]}"
  fi
done

PIDS=()
for ((w=0; w<N_WORKERS; w++)); do
  shard="${SHARDS[$w]}"
  if [[ -z "$shard" ]]; then continue; fi
  log_path="$LOG_DIR/collect_worker_${w}.log"
  shuffle_seed=$(( 17 + 31 * w ))
  echo "[shard $w] codes=[$shard]  shuffle_seed=$shuffle_seed -> $log_path"
  (
    set -e
    # script assumes CWD = repo root
    echo "=== shard $w start $(date) codes=$shard ===" >> "$log_path"
    SDL_VIDEODRIVER=dummy CUDA_VISIBLE_DEVICES= \
      "$PY" pdd-bench/scripts/agents/train/collect_benchmark_sign_trajectories.py \
        --sampled-dir         "$SAMPLED_DIR" \
        --output-dir          "$OUTPUT_DIR" \
        --max-steps           "$MAX_STEPS" \
        --traffic-density     "$TRAFFIC_DENSITY" \
        --success-target      "$SUCCESS_TARGET" \
        --min-sign-distance   "$MIN_SIGN_DISTANCE" \
        --max-attempts-per-sign "$MAX_ATTEMPTS_PER_SIGN" \
        --save-failed-pt-dir  "$FAILED_DIR" \
        --shuffle-seed        "$shuffle_seed" \
        --save-gifs --gifs-per-sign "$GIFS_PER_SIGN" \
        --only-codes          "$shard" \
      >> "$log_path" 2>&1
    echo "=== shard $w end $(date) ===" >> "$log_path"
  ) </dev/null >/dev/null 2>&1 &
  PIDS+=($!)
done

echo "[collect_v5_parallel] launched ${#PIDS[@]} workers: ${PIDS[*]}"
echo "[collect_v5_parallel] tail -F $LOG_DIR/collect_worker_*.log to monitor"

wait "${PIDS[@]}"
echo "[collect_v5_parallel] all shards done $(date)"
