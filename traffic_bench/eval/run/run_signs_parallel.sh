#!/usr/bin/env bash
# Parallel eval across signs (collect.sh-style).
#
# CPU policies: queue of signs, up to CPU_WORKERS concurrent eval runs
#   (each run uses jobs= for scene parallelism inside the sign).
# GPU policies: up to one sign per GPU; each run uses jobs_nn on that single GPU.
#
# Usage (from repo root, conda env active):
#   bash traffic_bench/eval/run/run_signs_parallel.sh
#   SPLIT=test CPU_WORKERS=4 GPUS=1,2,3,4,5,6,7 JOBS=16 JOBS_NN=16 bash ...
#   SIGNS="stop yield" SPLIT=train CPU_WORKERS=3 GPUS=1,2,3 JOBS=16 JOBS_NN=16 bash ...
#
# Signs with a ready real_manifest.jsonl are scheduled first. Signs still
# missing a manifest stay pending and the script polls until they appear
# (WAIT_POLL seconds; WAIT_TIMEOUT=0 means wait forever).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

: "${CPU_WORKERS:=4}"
: "${GPUS:=1,2,3,4,5,6,7}"
: "${JOBS:=16}"
: "${JOBS_NN:=16}"
# Full benchmark = 16 baselines:
#   idm × {default,s1–s4} + idm_rule × {default,s1–s4}  (10, CPU)
#   ppo_lidar + ppo_rule                                  (2, CPU)
#   carl + carl_rule + plant2 + plant2_rule               (4, GPU)
: "${EGO_VARIANTS:=[default,s1,s2,s3,s4]}"
: "${CPU_POLICIES:=[idm,idm_rule,ppo_lidar,ppo_rule]}"
: "${GPU_POLICIES:=[carl,carl_rule,plant2,plant2_rule]}"
# Manifest split under data/runs/<sign>/<SPLIT>/ (train | test).
: "${SPLIT:=train}"
# Optional subset, space- or comma-separated hydra sign ids.
#   SIGNS="stop yield main_road" …
# Empty / unset → full default list below.
: "${SIGNS:=}"
: "${LOG_DIR:=data/eval_parallel_logs/${SPLIT}}"
# How often to re-check pending signs for a newly written manifest.
: "${WAIT_POLL:=30}"
# Max seconds to wait for missing manifests (0 = forever).
: "${WAIT_TIMEOUT:=0}"

case "$SPLIT" in
  train|test) ;;
  *)
    echo "SPLIT must be 'train' or 'test' (got: $SPLIT)" >&2
    exit 1
    ;;
esac

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

DEFAULT_SIGNS=(
  main_road secondary yield stop roundabout blocked_road no_entry
  no_turn/right no_turn/left
  direction/straight direction/right direction/left
  direction/straight_right direction/straight_left direction/left_right
  one_way/right one_way/left
  detour/right detour/left detour/either
  speed_limit min_speed residential_zone zone_speed_limit crosswalk
)

if [[ -n "${SIGNS}" ]]; then
  # Accept "a b c" or "a,b,c"
  SIGNS="${SIGNS//,/ }"
  # shellcheck disable=SC2206
  SIGNS_ARR=($SIGNS)
else
  SIGNS_ARR=("${DEFAULT_SIGNS[@]}")
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
echo "SPLIT=$SPLIT  CPU_WORKERS=$CPU_WORKERS  GPUS=${GPU_LIST[*]}  JOBS=$JOBS  JOBS_NN=$JOBS_NN"
echo "CPU_POLICIES=$CPU_POLICIES  GPU_POLICIES=$GPU_POLICIES  EGO_VARIANTS=$EGO_VARIANTS"
echo "SIGNS (${#SIGNS_ARR[@]}): ${SIGNS_ARR[*]}"
echo "manifests: data/runs/<sign>/$SPLIT/  (ready first; wait for the rest)"
echo "wait: poll=${WAIT_POLL}s timeout=${WAIT_TIMEOUT}s (0=forever)"
echo "logs: $LOG_DIR"
echo "GPU pin: CUDA_VISIBLE_DEVICES=<index> (cluster NVD left intact)"

manifest_path() {
  local d; d="$(sign_dir "$1")"
  echo "data/runs/$d/$SPLIT/real_manifest.jsonl"
}

has_manifest() {
  [[ -f "$(manifest_path "$1")" ]]
}

# Ready signs first, then signs still waiting on a manifest (stable within each group).
order_ready_first() {
  local -n _src=$1
  local -n _dst=$2
  local ready=() waiting=() s
  _dst=()
  for s in "${_src[@]}"; do
    [[ -z "$s" ]] && continue
    if has_manifest "$s"; then
      ready+=("$s")
    else
      waiting+=("$s")
    fi
  done
  # Prefer if/then over ((n)) && … — with set -e an empty waiting[] makes
  # ((0)) the function's last status and aborts the whole script.
  if ((${#ready[@]})); then
    _dst+=("${ready[@]}")
  fi
  if ((${#waiting[@]})); then
    _dst+=("${waiting[@]}")
  fi
}

run_cpu_sign() {
  local s="$1"
  local d; d="$(sign_dir "$s")"
  local man="data/runs/$d/$SPLIT"
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
  local man="data/runs/$d/$SPLIT"
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

# --- CPU pool: launch ready signs first; keep missing-manifest signs pending ---
cpu_pids=()
cpu_pending=()
order_ready_first SIGNS_ARR cpu_pending

launch_cpu() {
  local kept=() s
  for s in "${cpu_pending[@]}"; do
    [[ -z "$s" ]] && continue
    if [[ ${#cpu_pids[@]} -ge $CPU_WORKERS ]]; then
      kept+=("$s")
      continue
    fi
    if has_manifest "$s"; then
      run_cpu_sign "$s" &
      cpu_pids+=("$!")
    else
      kept+=("$s")
    fi
  done
  cpu_pending=("${kept[@]}")
}

# --- GPU pool: one sign per GPU ---
gpu_pending=()
order_ready_first SIGNS_ARR gpu_pending

declare -a gpu_slot_pid
declare -a gpu_slot_busy
for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
  gpu_slot_pid[$i]=""
  gpu_slot_busy[$i]=0
done

launch_gpu() {
  local i g s kept=()
  # Re-order so newly written manifests jump ahead of still-waiting signs.
  local ordered=()
  order_ready_first gpu_pending ordered
  gpu_pending=("${ordered[@]}")

  for s in "${gpu_pending[@]}"; do
    [[ -z "$s" ]] && continue
    if ! has_manifest "$s"; then
      kept+=("$s")
      continue
    fi
    local assigned=0
    for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
      if [[ ${gpu_slot_busy[$i]} -eq 0 ]]; then
        g="${GPU_LIST[$i]}"
        run_gpu_sign "$s" "$g" &
        gpu_slot_pid[$i]=$!
        gpu_slot_busy[$i]=1
        assigned=1
        break
      fi
    done
    if [[ $assigned -eq 0 ]]; then
      kept+=("$s")
    fi
  done
  gpu_pending=("${kept[@]}")
}

reap_cpu() {
  local new=() p
  for p in "${cpu_pids[@]}"; do
    [[ -z "$p" ]] && continue
    if kill -0 "$p" 2>/dev/null; then
      new+=("$p")
    else
      wait "$p" || echo "[cpu] worker $p failed (see logs)"
    fi
  done
  cpu_pids=("${new[@]}")
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

gpu_busy_count() {
  local i n=0
  for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
    n=$((n + gpu_slot_busy[$i]))
  done
  echo "$n"
}

pending_without_manifest() {
  local -n _pend=$1
  local s out=()
  for s in "${_pend[@]}"; do
    [[ -z "$s" ]] && continue
    has_manifest "$s" || out+=("$s")
  done
  echo "${out[*]}"
}

# Initial ready/waiting snapshot
_ready=()
_waiting=()
for s in "${SIGNS_ARR[@]}"; do
  if has_manifest "$s"; then
    _ready+=("$s")
  else
    _waiting+=("$s")
  fi
done
echo "ready now (${#_ready[@]}): ${_ready[*]:-(none)}"
echo "waiting on manifest (${#_waiting[@]}): ${_waiting[*]:-(none)}"

wait_started_at=$SECONDS
last_wait_log=-999999

launch_cpu
launch_gpu

while true; do
  # Re-prioritize CPU pending when new manifests appear while workers are busy.
  if [[ ${#cpu_pending[@]} -gt 0 ]]; then
    _ord=()
    order_ready_first cpu_pending _ord
    cpu_pending=("${_ord[@]}")
  fi

  reap_cpu
  reap_gpu
  launch_cpu
  launch_gpu

  cpu_busy=${#cpu_pids[@]}
  gpu_busy="$(gpu_busy_count)"
  if [[ ${#cpu_pending[@]} -eq 0 && ${#gpu_pending[@]} -eq 0 && $cpu_busy -eq 0 && $gpu_busy -eq 0 ]]; then
    break
  fi

  miss_cpu="$(pending_without_manifest cpu_pending)"
  miss_gpu="$(pending_without_manifest gpu_pending)"
  if [[ -n "$miss_cpu$miss_gpu" && $cpu_busy -eq 0 && $gpu_busy -eq 0 ]]; then
    elapsed=$((SECONDS - wait_started_at))
    if [[ $WAIT_TIMEOUT -gt 0 && $elapsed -ge $WAIT_TIMEOUT ]]; then
      echo "ERROR: timed out after ${elapsed}s waiting for manifests:" >&2
      [[ -n "$miss_cpu" ]] && echo "  cpu: $miss_cpu" >&2
      [[ -n "$miss_gpu" ]] && echo "  gpu: $miss_gpu" >&2
      exit 1
    fi
    if [[ $((SECONDS - last_wait_log)) -ge $WAIT_POLL ]]; then
      echo "[wait] no ready manifests; polling every ${WAIT_POLL}s (elapsed ${elapsed}s)"
      [[ -n "$miss_cpu" ]] && echo "[wait] cpu pending: $miss_cpu"
      [[ -n "$miss_gpu" ]] && echo "[wait] gpu pending: $miss_gpu"
      last_wait_log=$SECONDS
    fi
  fi

  sleep 5
done

echo "ALL DONE. logs in $LOG_DIR"
