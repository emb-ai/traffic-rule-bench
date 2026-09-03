#!/usr/bin/env bash
# Parallel eval across signs (collect.sh-style).
#
# CPU policies: queue of signs, up to CPU_WORKERS concurrent eval runs
#   (each run uses jobs= for scene parallelism inside the sign).
# GPU policies: up to one sign per GPU; each run uses jobs_nn on that single GPU.
#
# Usage (from repo root, conda env active):
#   bash traffic_bench/eval/run/run_signs_parallel.sh
#   CPU_WORKERS=4 GPUS=1,2,3,4,5,6,7 JOBS=16 JOBS_NN=16 bash ...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

: "${CPU_WORKERS:=4}"
: "${GPUS:=1,2,3,4,5,6,7}"
: "${JOBS:=16}"
: "${JOBS_NN:=16}"
: "${EGO_VARIANTS:=[default]}"
: "${CPU_POLICIES:=[idm,idm_rule,ppo_lidar,ppo_rule]}"
: "${GPU_POLICIES:=[carl,carl_rule]}"
: "${LOG_DIR:=data/eval_parallel_logs}"

mkdir -p "$LOG_DIR"

# Avoid BLAS/OpenMP thrash when many MetaDrive workers share the node
# (same trick as legacy run_eval_8gpu.sh).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

# Pin like collect.sh: numeric CUDA_VISIBLE_DEVICES only.
# Do NOT override NVIDIA_VISIBLE_DEVICES to a single UUID — on MLSpace that is
# ignored and CVD=0 then lands every worker on physical GPU0.
gpu_env() {
  echo "CUDA_VISIBLE_DEVICES=$1"
}

# hydra sign id → on-disk runs dir
sign_dir() {
  case "$1" in
    secondary) echo secondary_road ;;
    no_turn/right) echo no_turn_right ;;
    no_turn/left) echo no_turn_left ;;
    direction/*) echo "direction_${1#direction/}" ;;
    one_way/right) echo one_way_right ;;
    one_way/left) echo one_way_left ;;
    detour/*) echo "detour_${1#detour/}" ;;
    *) echo "$1" ;;
  esac
}

SIGNS=(
  main_road secondary yield stop roundabout blocked_road no_entry
  no_turn/right no_turn/left
  direction/straight direction/right direction/left
  direction/straight_right direction/straight_left direction/left_right
  one_way/right one_way/left
  detour/right detour/left detour/either
  speed_limit min_speed residential_zone zone_speed_limit crosswalk
)

IFS=',' read -ra GPU_LIST <<< "$GPUS"
echo "CPU_WORKERS=$CPU_WORKERS  GPUS=${GPU_LIST[*]}  JOBS=$JOBS  JOBS_NN=$JOBS_NN"
echo "logs: $LOG_DIR"
echo "GPU pin: CUDA_VISIBLE_DEVICES=<index> (cluster NVD left intact)"

run_cpu_sign() {
  local s="$1"
  local d; d="$(sign_dir "$s")"
  local man="data/runs/$d/train"
  if [[ ! -f "$man/real_manifest.jsonl" ]]; then
    echo "[cpu] SKIP $s — no manifest"
    return 0
  fi
  local log="$LOG_DIR/cpu_${d}.log"
  echo "[cpu] START $s → $log"
  # Hide GPUs from CPU-policy workers so they don't grab VRAM.
  CUDA_VISIBLE_DEVICES="" \
  python -m traffic_bench.eval run \
    "policies=$CPU_POLICIES" \
    "ego_variants=$EGO_VARIANTS" \
    "sign=$s" \
    "manifest=$man" \
    "jobs=$JOBS" \
    jobs_nn=1 \
    >"$log" 2>&1
  echo "[cpu] DONE  $s"
}

run_gpu_sign() {
  local s="$1"
  local gpu="$2"
  local d; d="$(sign_dir "$s")"
  local man="data/runs/$d/train"
  if [[ ! -f "$man/real_manifest.jsonl" ]]; then
    echo "[gpu$gpu] SKIP $s — no manifest"
    return 0
  fi
  local log="$LOG_DIR/gpu${gpu}_${d}.log"
  local pin; pin="$(gpu_env "$gpu")"
  echo "[gpu$gpu] START $s ($pin) → $log"
  # One sign per GPU; jobs_nn MetaDrive workers share that single card.
  env $pin python -m traffic_bench.eval run \
    "policies=$GPU_POLICIES" \
    "ego_variants=$EGO_VARIANTS" \
    "sign=$s" \
    "manifest=$man" \
    jobs=1 \
    "jobs_nn=$JOBS_NN" \
    >"$log" 2>&1
  echo "[gpu$gpu] DONE  $s"
}

# --- CPU pool: fill up to CPU_WORKERS, refill as jobs finish ---
cpu_pids=()
cpu_signs=("${SIGNS[@]}")
cpu_i=0

launch_cpu() {
  while [[ ${#cpu_pids[@]} -lt $CPU_WORKERS && $cpu_i -lt ${#cpu_signs[@]} ]]; do
    local s="${cpu_signs[$cpu_i]}"
    cpu_i=$((cpu_i + 1))
    run_cpu_sign "$s" &
    cpu_pids+=("$!")
  done
}

# --- GPU pool: one sign per GPU ---
gpu_pids=()
gpu_signs=("${SIGNS[@]}")
gpu_i=0
gpu_slot=0  # index into GPU_LIST for next free assignment — actually track per-slot

declare -a gpu_slot_pid
declare -a gpu_slot_busy
for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
  gpu_slot_pid[$i]=""
  gpu_slot_busy[$i]=0
done

launch_gpu() {
  local i g s
  for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
    if [[ ${gpu_slot_busy[$i]} -eq 0 && $gpu_i -lt ${#gpu_signs[@]} ]]; then
      g="${GPU_LIST[$i]}"
      s="${gpu_signs[$gpu_i]}"
      gpu_i=$((gpu_i + 1))
      run_gpu_sign "$s" "$g" &
      gpu_slot_pid[$i]=$!
      gpu_slot_busy[$i]=1
    fi
  done
}

reap_cpu() {
  local new=() p
  for p in "${cpu_pids[@]:-}"; do
    [[ -z "$p" ]] && continue
    if kill -0 "$p" 2>/dev/null; then
      new+=("$p")
    else
      wait "$p" || echo "[cpu] worker $p failed (see logs)"
    fi
  done
  cpu_pids=("${new[@]:-}")
}

reap_gpu() {
  local i p
  for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
    p="${gpu_slot_pid[$i]:-}"
    [[ -z "$p" ]] && continue
    if ! kill -0 "$p" 2>/dev/null; then
      wait "$p" || echo "[gpu] worker $p failed (see logs)"
      gpu_slot_pid[$i]=""
      gpu_slot_busy[$i]=0
    fi
  done
}

launch_cpu
launch_gpu

while true; do
  reap_cpu
  reap_gpu
  launch_cpu
  launch_gpu
  # done when both queues drained and no workers left
  cpu_left=$(( ${#cpu_signs[@]} - cpu_i + ${#cpu_pids[@]} ))
  gpu_busy=0
  for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
    gpu_busy=$((gpu_busy + gpu_slot_busy[$i]))
  done
  gpu_left=$(( ${#gpu_signs[@]} - gpu_i + gpu_busy ))
  if [[ $cpu_left -le 0 && $gpu_left -le 0 ]]; then
    break
  fi
  sleep 5
done

echo "ALL DONE. logs in $LOG_DIR"
