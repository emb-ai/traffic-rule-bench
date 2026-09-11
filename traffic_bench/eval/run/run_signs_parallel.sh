#!/usr/bin/env bash
# Parallel eval across signs (collect.sh-style).
#
# CPU policies: queue of signs, up to CPU_WORKERS concurrent eval runs
#   (each run uses jobs= for scene parallelism inside the sign).
# GPU policies: pack free GPUs onto ready signs (1 sign can take several
#   cards via cuda_devices=). JOBS_NN is per GPU.
# After all eval runs finish, metrics combine writes data/runs/_all/<SPLIT>/.
#   (each `eval run` already writes per-sign csv/report).
#
# Usage (from repo root, conda env active):
#   bash traffic_bench/eval/run/run_signs_parallel.sh
#   SPLIT=test CPU_WORKERS=4 GPUS=1,2,3,4,5,6,7 JOBS=16 JOBS_NN=16 bash ...
#   SIGNS="stop yield" SPLIT=train CPU_WORKERS=3 GPUS=1,2,3 JOBS=16 JOBS_NN=16 bash ...
#   # two CaRL baselines, skip CPU, use all 4 GPUs on one sign:
#   SPLIT=test SIGNS="no_turn/left" POLICIES=carl,carl_rule \
#   GPUS=0,1,2,3 JOBS_NN=32 bash traffic_bench/eval/run/run_signs_parallel.sh
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
# Optional: POLICIES=carl,carl_rule  (or [carl,carl_rule]) overrides both lists
# and skips the empty side (CPU and/or GPU).
: "${POLICIES:=}"
# Manifest split under data/runs/<sign>/<SPLIT>/ (train | test).
: "${SPLIT:=train}"
# Optional subset, space- or comma-separated hydra sign ids.
#   SIGNS="stop yield main_road" …
# Empty / unset → full default list below.
: "${SIGNS:=}"
: "${LOG_DIR:=data/eval_parallel_logs/${SPLIT}}"
# How often to re-check pending signs for a newly written manifest.
: "${WAIT_POLL:=30}"
# Max seconds to wait for missing manifests (0 = wait forever).
: "${WAIT_TIMEOUT:=0}"
# After eval, merge per-sign CSVs → data/runs/_all/<SPLIT>/. 0 = skip combine
# (per-sign reports from `eval run` are still written).
: "${RUN_METRICS:=1}"
run_failed=0

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

policy_tokens() {
  local raw="${1:-}"
  raw="${raw//[\[\]]/}"
  raw="${raw//,/ }"
  local t
  for t in $raw; do
    case "$t" in
      "" | none | skip | "-" | null) continue ;;
    esac
    printf '%s\n' "$t"
  done
}

hydra_policy_list() {
  local -a toks=()
  local t
  for t in "$@"; do
    [[ -n "$t" ]] && toks+=("$t")
  done
  if ((${#toks[@]} == 0)); then
    echo ""
    return
  fi
  local IFS=,
  echo "[${toks[*]}]"
}

is_cpu_policy() {
  case "$1" in
    idm | idm_rule | ppo_lidar | ppo_rule) return 0 ;;
    *) return 1 ;;
  esac
}

is_gpu_policy() {
  case "$1" in
    carl | carl_rule | plant2 | plant2_rule) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -n "${POLICIES}" ]]; then
  cpu_toks=()
  gpu_toks=()
  while IFS= read -r t; do
    [[ -z "$t" ]] && continue
    if is_cpu_policy "$t"; then
      cpu_toks+=("$t")
    elif is_gpu_policy "$t"; then
      gpu_toks+=("$t")
    else
      echo "ERROR: unknown policy '$t' in POLICIES (cpu: idm,idm_rule,ppo_lidar,ppo_rule; gpu: carl,carl_rule,plant2,plant2_rule)" >&2
      exit 1
    fi
  done < <(policy_tokens "$POLICIES")
  CPU_POLICIES="$(hydra_policy_list ${cpu_toks[@]+"${cpu_toks[@]}"})"
  GPU_POLICIES="$(hydra_policy_list ${gpu_toks[@]+"${gpu_toks[@]}"})"
fi

CPU_POLICIES="$(hydra_policy_list $(policy_tokens "$CPU_POLICIES"))"
GPU_POLICIES="$(hydra_policy_list $(policy_tokens "$GPU_POLICIES"))"

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
# Drop empty tokens from GPUS="" / trailing commas.
_gpu_clean=()
for g in "${GPU_LIST[@]+"${GPU_LIST[@]}"}"; do
  [[ -n "$g" ]] && _gpu_clean+=("$g")
done
GPU_LIST=("${_gpu_clean[@]+"${_gpu_clean[@]}"}")

cpu_enabled=1
gpu_enabled=1
if [[ -z "$CPU_POLICIES" ]] || [[ "${CPU_WORKERS}" -lt 1 ]]; then
  cpu_enabled=0
fi
if [[ -z "$GPU_POLICIES" ]] || [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  gpu_enabled=0
fi
if [[ $cpu_enabled -eq 0 && $gpu_enabled -eq 0 ]]; then
  echo "ERROR: no policies to run (set POLICIES=… or CPU_POLICIES=/GPU_POLICIES=)" >&2
  exit 1
fi

echo "SPLIT=$SPLIT  CPU_WORKERS=$CPU_WORKERS  GPUS=${GPU_LIST[*]:-(none)}  JOBS=$JOBS  JOBS_NN=$JOBS_NN (per GPU)"
echo "CPU_POLICIES=${CPU_POLICIES:-(skip)}  GPU_POLICIES=${GPU_POLICIES:-(skip)}  EGO_VARIANTS=$EGO_VARIANTS"
echo "SIGNS (${#SIGNS_ARR[@]}): ${SIGNS_ARR[*]}"
echo "manifests: data/runs/<sign>/$SPLIT/  (ready first; wait for the rest)"
echo "wait: poll=${WAIT_POLL}s timeout=${WAIT_TIMEOUT}s (0=forever)"
echo "logs: $LOG_DIR"
echo "metrics combine: $RUN_METRICS  (per-sign reports always run at end of each eval)"
echo "GPU: pack free cards onto ready signs via cuda_devices= (JOBS_NN × n_gpus)"

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
  local gpus="$2"
  local d; d="$(sign_dir "$s")"
  local man="data/runs/$d/$SPLIT"
  local tag="${gpus//,/-}"
  local log="$LOG_DIR/gpu${tag}_${d}.log"
  local -a gs=()
  IFS=',' read -ra gs <<< "$gpus"
  local n=${#gs[@]}
  local jobs_nn=$((JOBS_NN * n))
  echo "[gpu $gpus] START $s  jobs_nn=$jobs_nn (JOBS_NN=$JOBS_NN × $n) → $log"
  # Do not pin CUDA_VISIBLE_DEVICES to a single index: policies.py then
  # refuses to spread jobs_nn across cuda_devices=. Pass the list to Hydra.
  python -m traffic_bench.eval run \
    "policies=$GPU_POLICIES" \
    "ego_variants=$EGO_VARIANTS" \
    "sign=$s" \
    "manifest=$man" \
    jobs=1 \
    "jobs_nn=$jobs_nn" \
    "cuda_devices=[$gpus]" \
    >"$log" 2>&1
  echo "[gpu $gpus] DONE  $s"
}

# --- CPU pool: launch ready signs first; keep missing-manifest signs pending ---
cpu_pids=()
cpu_pending=()
if [[ $cpu_enabled -eq 1 ]]; then
  order_ready_first SIGNS_ARR cpu_pending
fi

launch_cpu() {
  [[ $cpu_enabled -eq 1 ]] || return 0
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

# --- GPU pool: pack free GPUs onto ready signs ---
gpu_pending=()
if [[ $gpu_enabled -eq 1 ]]; then
  order_ready_first SIGNS_ARR gpu_pending
fi

declare -a gpu_slot_pid
declare -a gpu_slot_busy
for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
  gpu_slot_pid[$i]=""
  gpu_slot_busy[$i]=0
done

launch_gpu() {
  [[ $gpu_enabled -eq 1 ]] || return 0
  local i s
  # Re-order so newly written manifests jump ahead of still-waiting signs.
  local ordered=()
  order_ready_first gpu_pending ordered
  gpu_pending=("${ordered[@]}")

  local -a free_idx=()
  for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
    if [[ ${gpu_slot_busy[$i]} -eq 0 ]]; then
      free_idx+=("$i")
    fi
  done
  local n_free=${#free_idx[@]}
  if [[ $n_free -eq 0 ]]; then
    return 0
  fi

  local -a ready_signs=() waiting_signs=()
  for s in "${gpu_pending[@]}"; do
    [[ -z "$s" ]] && continue
    if has_manifest "$s"; then
      ready_signs+=("$s")
    else
      waiting_signs+=("$s")
    fi
  done
  local n_ready=${#ready_signs[@]}
  if [[ $n_ready -eq 0 ]]; then
    gpu_pending=("${waiting_signs[@]+"${waiting_signs[@]}"}")
    return 0
  fi

  local n_launch=$n_ready
  if [[ $n_launch -gt $n_free ]]; then
    n_launch=$n_free
  fi
  local per=$((n_free / n_launch))
  local extra=$((n_free % n_launch))
  [[ $per -lt 1 ]] && per=1

  local fi=0 launched=0 kept=()
  for s in "${ready_signs[@]}"; do
    if [[ $launched -ge $n_launch ]]; then
      kept+=("$s")
      continue
    fi
    local k=$per
    if [[ $extra -gt 0 ]]; then
      k=$((k + 1))
      extra=$((extra - 1))
    fi
    local -a take=()
    local j
    for ((j = 0; j < k; j++)); do
      take+=("${GPU_LIST[${free_idx[$fi]}]}")
      fi=$((fi + 1))
    done
    local gpus
    local IFS=,
    gpus="${take[*]}"
    unset IFS
    run_gpu_sign "$s" "$gpus" &
    local pid=$!
    for ((j = fi - k; j < fi; j++)); do
      gpu_slot_pid[${free_idx[$j]}]=$pid
      gpu_slot_busy[${free_idx[$j]}]=1
    done
    launched=$((launched + 1))
  done
  gpu_pending=("${kept[@]+"${kept[@]}"}" "${waiting_signs[@]+"${waiting_signs[@]}"}")
}

reap_cpu() {
  local new=() p
  for p in "${cpu_pids[@]}"; do
    [[ -z "$p" ]] && continue
    if kill -0 "$p" 2>/dev/null; then
      new+=("$p")
    else
      wait "$p" || { echo "[cpu] worker $p failed (see logs)"; run_failed=1; }
    fi
  done
  cpu_pids=("${new[@]}")
}

reap_gpu() {
  local i j p
  local -a dead=()
  for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do
    p="${gpu_slot_pid[$i]:-}"
    [[ -z "$p" ]] && continue
    if ! kill -0 "$p" 2>/dev/null; then
      local seen=0
      for d in "${dead[@]+"${dead[@]}"}"; do
        [[ "$d" == "$p" ]] && seen=1
      done
      if [[ $seen -eq 0 ]]; then
        dead+=("$p")
        wait "$p" || { echo "[gpu] worker $p failed (see logs)"; run_failed=1; }
      fi
    fi
  done
  for p in "${dead[@]+"${dead[@]}"}"; do
    for ((j = 0; j < ${#GPU_LIST[@]}; j++)); do
      if [[ "${gpu_slot_pid[$j]:-}" == "$p" ]]; then
        gpu_slot_pid[$j]=""
        gpu_slot_busy[$j]=0
      fi
    done
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

echo "ALL RUNS DONE. logs in $LOG_DIR"

if [[ "$run_failed" -ne 0 ]]; then
  echo "ERROR: one or more eval workers failed; skip metrics combine. See $LOG_DIR" >&2
  exit 1
fi

if [[ "$RUN_METRICS" != "0" ]]; then
  if [[ -n "${SIGNS}" ]]; then
    _combine_sign="$(IFS=,; echo "${SIGNS_ARR[*]}")"
  else
    _combine_sign="all"
  fi
  echo "[metrics] combine sign=${_combine_sign} paths.split=$SPLIT"
  python -m traffic_bench.eval metrics combine \
    "sign=${_combine_sign}" \
    "paths.split=$SPLIT" \
    || echo "[metrics] combine failed (per-sign reports still under data/runs/<sign>/$SPLIT/eval_out)"
fi

echo "ALL DONE. logs in $LOG_DIR"
