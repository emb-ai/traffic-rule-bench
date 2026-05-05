#!/usr/bin/env bash
# run_collect_v4_parallel.sh — shard collect_benchmark_sign_trajectories.py
# across N worker processes by --only-codes. Each worker writes into the
# shared OUTPUT_DIR. File names are <slug>_ep<i:03d>.pt so distinct codes
# never collide.
#
# Usage:
#   bash run_collect_v4_parallel.sh                                    # default 8 workers
#   N_WORKERS=4 SAMPLED_DIR=...sampled_for_expert_v2 bash run_collect_v4_parallel.sh
#
# Tunables:
#   N_WORKERS         (default 8)
#   PY                (default /home/jovyan/.mlspace/envs/plant2/bin/python)
#   SAMPLED_DIR       (default benchmark_output/sampled_for_expert_v2)
#   OUTPUT_DIR        (default pdd-bench/outputs/benchmark_sign_trajectories_v4)
#   MAX_STEPS         (default 600)
#   GIFS_PER_SIGN     (default 3) — only render this many GIFs per sign
#   TRAFFIC_DENSITY   (default 0.1)
#   START_INDEX       (default 0)
#   ONLY_CODES        (override the full list, comma-separated)
#   LOG_DIR           (default $OUTPUT_DIR/_collect_logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARBELYAEV_SDC="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # arbelyaev/sdc/pdd-bench

PY="${PY:-/home/jovyan/.mlspace/envs/plant2/bin/python}"
N_WORKERS="${N_WORKERS:-8}"

SAMPLED_DIR="${SAMPLED_DIR:-$ARBELYAEV_SDC/scripts/per_sign_bench/benchmark_output/sampled_for_expert_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARBELYAEV_SDC/outputs/benchmark_sign_trajectories_v4}"
MAX_STEPS="${MAX_STEPS:-600}"
GIFS_PER_SIGN="${GIFS_PER_SIGN:-3}"
TRAFFIC_DENSITY="${TRAFFIC_DENSITY:-0.1}"
START_INDEX="${START_INDEX:-0}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/_collect_logs}"

# Default: 34 PDD codes (the v2 set: original 32 + 2.5 + 5.16).
DEFAULT_CODES=(
  "2.1" "2.2" "2.3.1" "2.3.2" "2.3.3" "2.4" "2.5"
  "3.1" "3.2" "3.18.2" "3.19" "3.20"
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

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[collect_v4_parallel] PY=$PY"
echo "[collect_v4_parallel] N_WORKERS=$N_WORKERS"
echo "[collect_v4_parallel] SAMPLED_DIR=$SAMPLED_DIR"
echo "[collect_v4_parallel] OUTPUT_DIR=$OUTPUT_DIR"
echo "[collect_v4_parallel] MAX_STEPS=$MAX_STEPS GIFS_PER_SIGN=$GIFS_PER_SIGN"
echo "[collect_v4_parallel] codes(${#CODES[@]}): ${CODES[*]}"
echo "[collect_v4_parallel] LOG_DIR=$LOG_DIR"

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
  echo "[shard $w] codes=[$shard] -> $log_path"
  (
    set -e
    cd "$ARBELYAEV_SDC/.."   # /home/jovyan/.../arbelyaev/sdc
    echo "=== shard $w start $(date) codes=$shard ===" >> "$log_path"
    SDL_VIDEODRIVER=dummy CUDA_VISIBLE_DEVICES= \
      "$PY" pdd-bench/scripts/agents/train/collect_benchmark_sign_trajectories.py \
        --sampled-dir   "$SAMPLED_DIR" \
        --output-dir    "$OUTPUT_DIR" \
        --max-steps     "$MAX_STEPS" \
        --traffic-density "$TRAFFIC_DENSITY" \
        --start-index   "$START_INDEX" \
        --save-gifs --gifs-per-sign "$GIFS_PER_SIGN" \
        --only-codes    "$shard" \
      >> "$log_path" 2>&1
    echo "=== shard $w end $(date) ===" >> "$log_path"
  ) </dev/null >/dev/null 2>&1 &
  PIDS+=($!)
done

echo "[collect_v4_parallel] launched ${#PIDS[@]} workers: ${PIDS[*]}"
echo "[collect_v4_parallel] tail -F $LOG_DIR/collect_worker_*.log to monitor"

wait "${PIDS[@]}"
echo "[collect_v4_parallel] all shards done $(date)"
