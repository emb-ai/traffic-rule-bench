#!/usr/bin/env bash
# Spatial FT eval: wait for all 7 trainings → hypothesis checks → 7-GPU waves.
#
# Per wave (best → ep029 → ep024 → ep019 → ep014 → ep009 → ep004):
#   GPU0=lr1e6, GPU1=lr5e6, … GPU6=lr1e4 — each its own ckpt + metrics folder.
#   Steps per ckpt: rule-signs → summarize_reports → fv_fast → fv_fast_detour
#
# Usage:
#   nohup bash collected_trajectories/launch_spatial_ft_eval_7gpu.sh \
#     >> collected_trajectories/logs_pipeline_spatial_signs/nohup_spatial_eval.out 2>&1 &
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

CT="$PIPELINE_DIR"
REPO="$TRB_ROOT"
CKPT_ROOT="$REPO/plant2/PlanT/checkpoints_ft"
# Override for re-runs (e.g. after eval input fix):
#   METRICS_ROOT=.../spatial_signs_eval_boxesfix TAG_SUFFIX=_boxesfix SLOTS=best \
#   SKIP_WAIT_FT=1 SKIP_HYPOTHESIS=1 bash launch_spatial_ft_eval_7gpu.sh
METRICS_ROOT="${METRICS_ROOT:-$SHEPELEV/plant2_ft_metrics/spatial_signs_eval}"
SIGNS_DIR="$REPO/pdd-bench/scripts/per_sign_bench/plant2_rule_test"
SIGNS_OUT="$SIGNS_DIR/output"
PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
FAST="$CT/run_eval_fast_plant2ft.sh"
HYP="$CT/run_eval_hypothesis_checks.sh"
LOGDIR="$SHEPELEV/collected_trajectories/logs_pipeline_spatial_signs"
TAG_SUFFIX="${TAG_SUFFIX:-}"
SKIP_WAIT_FT="${SKIP_WAIT_FT:-0}"
SKIP_HYPOTHESIS="${SKIP_HYPOTHESIS:-0}"
mkdir -p "$METRICS_ROOT" "$LOGDIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SDL_AUDIODRIVER=dummy PER_SIGN_COMPLIANT_NPC=1

# Optimized after hypothesis checks (override via env)
SIGNS_JOBS=${SIGNS_JOBS:-20}
SCENES_PER_JOB=${SCENES_PER_JOB:-32}
FV_NSHARDS=${FV_NSHARDS:-28}
FV_CONCURRENCY=${FV_CONCURRENCY:-28}
FV_NSHARDS_PER_GPU=${FV_NSHARDS_PER_GPU:-8}
FV_CONCURRENCY_PER_GPU=${FV_CONCURRENCY_PER_GPU:-8}
MAX_EVAL_RETRIES=${MAX_EVAL_RETRIES:-3}
MAX_HYP_RETRIES=${MAX_HYP_RETRIES:-2}

NFS2=/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova
MANIFEST_V61=${MANIFEST_V61:-$NFS2/traffic-rule-bench/pdd-bench/benchmark_output_speed/balanced/run_v61_a6/catalog_fv_test20.jsonl}
SCENES_V61=${SCENES_V61:-$NFS2/traffic-rule-bench/pdd-bench/scenes_balanced}
MANIFEST_DETOUR=${MANIFEST_DETOUR:-$NFS2/traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl}
SCENES_DETOUR=${SCENES_DETOUR:-$NFS2/sdc/pdd-bench/scenes}

# GPU i → learning rate addon suffix
LRS=(1e6 5e6 1e5 3e5 5e5 7e5 1e4)
# Eval wave order (override: SLOTS="best" or SLOTS="best ep029")
if [[ -n "${SLOTS:-}" ]]; then
  # shellcheck disable=SC2206
  SLOTS=($SLOTS)
else
  SLOTS=(best ep029 ep024 ep019 ep014 ep009 ep004)
fi

log() { echo "[$(date -Is)] $*"; }

log_snippet() {
  local logf="$1" n="${2:-40}"
  [[ -f "$logf" ]] || return 0
  log "--- error tail ($logf) ---"
  tail -n "$n" "$logf" 2>/dev/null | sed 's/^/  /'
}

ensure_thread_env() {
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export TORCH_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
}

# Read eval log tail; apply known mitigations. Returns 0 if a fix was applied.
apply_eval_fix() {
  local step="$1" logf="$2" tag="${3:-}"
  local snip fixed=0
  snip="$(tail -n 60 "$logf" 2>/dev/null || true)"
  [[ -n "$snip" ]] || return 1

  if echo "$snip" | rg -qi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED'; then
    if [[ "$step" == signs ]]; then
      SIGNS_JOBS=$((SIGNS_JOBS / 2)); (( SIGNS_JOBS < 1 )) && SIGNS_JOBS=1
      SCENES_PER_JOB=$((SCENES_PER_JOB / 2)); (( SCENES_PER_JOB < 1 )) && SCENES_PER_JOB=1
      log "fix[$tag] signs OOM → jobs=$SIGNS_JOBS spj=$SCENES_PER_JOB"
    else
      FV_NSHARDS_PER_GPU=$((FV_NSHARDS_PER_GPU / 2)); (( FV_NSHARDS_PER_GPU < 1 )) && FV_NSHARDS_PER_GPU=1
      FV_CONCURRENCY_PER_GPU=$FV_NSHARDS_PER_GPU
      FV_NSHARDS=$((FV_NSHARDS / 2)); (( FV_NSHARDS < 1 )) && FV_NSHARDS=1
      FV_CONCURRENCY=$FV_NSHARDS
      log "fix[$tag] fv OOM → per_gpu shards=$FV_NSHARDS_PER_GPU global=$FV_NSHARDS"
    fi
    fixed=1
  fi

  if echo "$snip" | rg -qi 'torch\.get_num_threads\(\)=|MKL_NUM_THREADS|thread oversubscription'; then
    ensure_thread_env
    log "fix[$tag] thread oversubscription → OMP/MKL/TORCH=1"
    fixed=1
  fi

  if echo "$snip" | rg -qi 'No such file|FileNotFoundError|missing MANIFEST|missing SCENES|missing CKPT'; then
    log "fix[$tag] path/NFS glitch — wait 30s and retry"
    sleep 30
    fixed=1
  fi

  if echo "$snip" | rg -qi 'Address already in use|EADDRINUSE|Resource temporarily unavailable'; then
    log "fix[$tag] port/socket busy — wait 20s"
    sleep 20
    fixed=1
  fi

  if echo "$snip" | rg -qi 'Segmentation fault|Killed|SIGKILL|worker.*died|BrokenProcessPool|EOFError'; then
    if [[ "$step" == signs ]]; then
      SIGNS_JOBS=$((SIGNS_JOBS * 2 / 3)); (( SIGNS_JOBS < 1 )) && SIGNS_JOBS=1
      log "fix[$tag] worker crash → signs jobs=$SIGNS_JOBS"
    else
      FV_NSHARDS_PER_GPU=$((FV_NSHARDS_PER_GPU * 2 / 3)); (( FV_NSHARDS_PER_GPU < 1 )) && FV_NSHARDS_PER_GPU=1
      FV_CONCURRENCY_PER_GPU=$FV_NSHARDS_PER_GPU
      log "fix[$tag] worker crash → fv per_gpu shards=$FV_NSHARDS_PER_GPU"
    fi
    fixed=1
  fi

  if [[ "$step" == fv_* ]] && echo "$snip" | rg -qi 'shard|multiprocessing|pickle|RemoteTraceback'; then
    if [[ -n "$tag" ]]; then
      local sub="${step#fv_}"
      rm -rf "$METRICS_ROOT/$tag/$sub/shards" "$METRICS_ROOT/$tag/$sub/parts" 2>/dev/null || true
      log "fix[$tag] cleared partial fv shards ($sub)"
      fixed=1
    fi
  fi

  if echo "$snip" | rg -qi 'catalog\.jsonl|MANIFEST has .* rows \(>5000\)'; then
    log "fix[$tag] wrong manifest — forcing catalog_fv_test20 paths"
    fixed=1
  fi

  if (( fixed == 0 )); then
    ensure_thread_env
    log "fix[$tag] no pattern match — ensure thread env and retry"
  fi
  return 0
}

run_hypothesis_checks() {
  local attempt=0 rc=0
  while (( attempt < MAX_HYP_RETRIES )); do
    attempt=$((attempt + 1))
    log "hypothesis checks attempt=$attempt"
    if bash "$HYP"; then
      log "hypothesis checks OK"
      return 0
    fi
    rc=$?
    log "hypothesis checks failed rc=$rc"
    log_snippet /tmp/eval_hypothesis_results.log 30
    apply_eval_fix hypothesis /tmp/eval_hypothesis_results.log hyp || true
    sleep 15
  done
  log "hypothesis checks gave up after $MAX_HYP_RETRIES attempts (continuing eval)"
  return 0
}

wait_all_ft() {
  log "waiting for all 7 spatial FT jobs (epoch=029 + no tmux) …"
  while true; do
    local missing=0 alive=0
    for lr in "${LRS[@]}"; do
      local d="$CKPT_ROOT/fvexp30_spatial_lr${lr}"
      if [[ ! -f "$d/epoch=029_fvexp30_spatial_lr${lr}_1.ckpt" ]]; then
        missing=$((missing + 1))
      fi
    done
    alive=$(tmux ls 2>/dev/null | rg -c 'arbelyaev-ft-spatial-lr' || true)
    log "ft: missing_ep029=$missing tmux_sessions=$alive"
    if (( missing == 0 && alive == 0 )); then
      log "all FT done"
      return 0
    fi
    sleep 120
  done
}

resolve_ckpt() {
  local lr="$1" slot="$2"
  local d="$CKPT_ROOT/fvexp30_spatial_lr${lr}"
  case "$slot" in
    best)
      ls "$d"/best_*_fvexp30_spatial_lr${lr}_1.ckpt 2>/dev/null | head -1
      ;;
    ep*)
      local ep="${slot#ep}"
      ls "$d"/epoch="${ep}"_fvexp30_spatial_lr${lr}_1.ckpt 2>/dev/null | head -1
      ;;
    *)
      return 1
      ;;
  esac
}

make_tag() {
  local lr="$1" slot="$2" ckpt="$3"
  local bn ep base
  bn=$(basename "$ckpt")
  if [[ "$slot" == best && "$bn" =~ best_([0-9]+)_ ]]; then
    base=$(printf 'fvexp30_spatial_lr%s_best%s' "$lr" "${BASH_REMATCH[1]}")
  elif [[ "$bn" =~ epoch=([0-9]+)_ ]]; then
    base=$(printf 'fvexp30_spatial_lr%s_ep%s' "$lr" "${BASH_REMATCH[1]}")
  else
    base=$(printf 'fvexp30_spatial_lr%s_%s' "$lr" "$slot")
  fi
  printf '%s%s' "$base" "$TAG_SUFFIX"
}

setup_tag_dir() {
  local tag="$1" ckpt="$2"
  mkdir -p "$METRICS_ROOT/$tag"/{logs,fv_fast,fv_fast_detour}
  printf '%s\n' "$ckpt" > "$METRICS_ROOT/$tag/ckpt.txt"
  mkdir -p "$SIGNS_OUT/$tag"
  ln -sfn "$SIGNS_OUT/$tag" "$METRICS_ROOT/$tag/signs"
}

signs_done() { [[ -f "$SIGNS_OUT/$1/_summary/summary.md" ]]; }
fv_done() { [[ -f "$METRICS_ROOT/$1/$2/reports/report_cumulative.md" ]]; }

run_signs() {
  local tag="$1" ckpt="$2" gpu="$3"
  local logf="$METRICS_ROOT/$tag/logs/eval_checkpoint.log"
  local attempt=0 rc=0

  if signs_done "$tag"; then
    log "SIGNS SKIP $tag"
    return 0
  fi

  while (( attempt < MAX_EVAL_RETRIES )); do
    attempt=$((attempt + 1))
    if signs_done "$tag"; then
      log "SIGNS OK $tag (summary exists)"
      return 0
    fi
    log "SIGNS START tag=$tag gpu=$gpu jobs=$SIGNS_JOBS spj=$SCENES_PER_JOB attempt=$attempt"
    ensure_thread_env
    (
      cd "$SIGNS_DIR"
      unset PYTHONPATH
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PY" -u eval_checkpoint_on_test.py \
        --policies plant2 \
        --model-paths "plant2:$ckpt" \
        --jobs "$SIGNS_JOBS" \
        --scenes-per-job "$SCENES_PER_JOB" \
        --keep-going \
        --run-name "$tag"
    ) >>"$logf" 2>&1
    rc=$?
    "$PY" "$SIGNS_DIR/summarize_reports.py" \
      --run-name "$tag" \
      --baseline plant2_default \
      --out-dir "$SIGNS_OUT/$tag/_summary" >>"$logf" 2>&1 || true

    if signs_done "$tag"; then
      log "SIGNS DONE tag=$tag rc=$rc"
      return 0
    fi
    log "SIGNS FAIL tag=$tag rc=$rc attempt=$attempt/$MAX_EVAL_RETRIES"
    log_snippet "$logf"
    apply_eval_fix signs "$logf" "$tag" || true
    sleep 10
  done
  log "SIGNS GAVE UP tag=$tag"
  return 1
}

run_fv() {
  local tag="$1" ckpt="$2" sub="$3" manifest="$4" scenes="$5" gpus="$6"
  local logf="$METRICS_ROOT/$tag/logs/run_eval_fast_${sub}.log"
  local nshards=$FV_NSHARDS concurrency=$FV_CONCURRENCY
  local attempt=0 rc=0 step="fv_${sub}"

  if [[ "$gpus" != *" "* ]]; then
    nshards=${FV_NSHARDS_PER_GPU:-8}
    concurrency=${FV_CONCURRENCY_PER_GPU:-8}
  fi
  if fv_done "$tag" "$sub"; then
    log "FV SKIP tag=$tag sub=$sub"
    return 0
  fi

  while (( attempt < MAX_EVAL_RETRIES )); do
    attempt=$((attempt + 1))
    if fv_done "$tag" "$sub"; then
      log "FV OK tag=$tag sub=$sub (report exists)"
      return 0
    fi
    nshards=$FV_NSHARDS
    concurrency=$FV_CONCURRENCY
    if [[ "$gpus" != *" "* ]]; then
      nshards=${FV_NSHARDS_PER_GPU:-8}
      concurrency=${FV_CONCURRENCY_PER_GPU:-8}
    fi
    log "FV START tag=$tag sub=$sub gpus=$gpus shards=$nshards conc=$concurrency attempt=$attempt"
    ensure_thread_env
    CKPT="$ckpt" OUT="$METRICS_ROOT/$tag/$sub" TAG="$tag" \
      NN_POLICIES=plant2 IDM_FAMILY_POLICIES= CPU_SINGLE_POLICIES= \
      MANIFEST="$manifest" SCENES="$scenes" \
      GPUS="$gpus" NSHARDS="$nshards" CONCURRENCY="$concurrency" \
      bash "$FAST" >>"$logf" 2>&1
    rc=$?

    if fv_done "$tag" "$sub"; then
      log "FV DONE tag=$tag sub=$sub rc=$rc"
      return 0
    fi
    log "FV FAIL tag=$tag sub=$sub rc=$rc attempt=$attempt/$MAX_EVAL_RETRIES"
    log_snippet "$logf"
    apply_eval_fix "$step" "$logf" "$tag" || true
    sleep 10
  done
  log "FV GAVE UP tag=$tag sub=$sub"
  return 1
}

run_one_ckpt() {
  local gpu="$1" lr="$2" slot="$3"
  local ckpt tag rc=0
  ckpt=$(resolve_ckpt "$lr" "$slot") || true
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "SKIP gpu=$gpu lr=$lr slot=$slot (ckpt missing)"
    return 0
  fi
  tag=$(make_tag "$lr" "$slot" "$ckpt")
  setup_tag_dir "$tag" "$ckpt"

  run_signs "$tag" "$ckpt" "$gpu" || rc=1
  if (( rc == 0 )); then
    run_fv "$tag" "$ckpt" fv_fast "$MANIFEST_V61" "$SCENES_V61" "$gpu" || rc=1
  fi
  if (( rc == 0 )); then
    run_fv "$tag" "$ckpt" fv_fast_detour "$MANIFEST_DETOUR" "$SCENES_DETOUR" "$gpu" || rc=1
  fi

  if (( rc == 0 )); then
    log "COMPLETE tag=$tag"
    return 0
  fi
  log "INCOMPLETE tag=$tag rc=$rc (will retry in wave recovery if needed)"
  return 1
}

run_wave() {
  local slot="$1"
  log "======== WAVE slot=$slot (7 LRs on GPUs 0-6) ========"
  local pids=() gpulrs=() i gpu lr fail=0

  for i in "${!LRS[@]}"; do
    gpu=$i
    lr=${LRS[$i]}
    run_one_ckpt "$gpu" "$lr" "$slot" &
    pids+=($!)
    gpulrs+=("${gpu}|${lr}")
  done

  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || fail=$((fail + 1))
  done
  log "WAVE slot=$slot first pass fail=$fail"

  # Retry failed LRs sequentially (avoid 7× retry storm on shared NFS/GPU).
  local retry_round=0
  while (( fail > 0 && retry_round < MAX_EVAL_RETRIES )); do
    retry_round=$((retry_round + 1))
    log "WAVE slot=$slot recovery round=$retry_round fail=$fail"
    local still_fail=0 entry g l
    for entry in "${gpulrs[@]}"; do
      IFS='|' read -r g l <<<"$entry"
      local ckpt tag
      ckpt=$(resolve_ckpt "$l" "$slot") || true
      [[ -n "$ckpt" && -f "$ckpt" ]] || continue
      tag=$(make_tag "$l" "$slot" "$ckpt")
      if signs_done "$tag" && fv_done "$tag" fv_fast && fv_done "$tag" fv_fast_detour; then
        continue
      fi
      run_one_ckpt "$g" "$l" "$slot" || still_fail=$((still_fail + 1))
    done
    fail=$still_fail
  done

  log "WAVE slot=$slot done fail=$fail"
  return $fail
}

# --- main ---
log "launch_spatial_ft_eval_7gpu start"
log "METRICS_ROOT=$METRICS_ROOT TAG_SUFFIX='$TAG_SUFFIX' SLOTS=${SLOTS[*]}"
log "SKIP_WAIT_FT=$SKIP_WAIT_FT SKIP_HYPOTHESIS=$SKIP_HYPOTHESIS"

if [[ "$SKIP_WAIT_FT" == "1" ]]; then
  log "SKIP wait_all_ft (requested)"
else
  wait_all_ft
fi

if [[ "$SKIP_HYPOTHESIS" == "1" ]]; then
  log "SKIP hypothesis checks (requested)"
elif [[ ! -f /tmp/eval_hypothesis_results.log ]] || ! rg -q 'done' /tmp/eval_hypothesis_results.log 2>/dev/null; then
  run_hypothesis_checks
fi

TOTAL_FAIL=0
for slot in "${SLOTS[@]}"; do
  if run_wave "$slot"; then
    log "wave slot=$slot OK"
  else
    log "wave slot=$slot FAILED after retries"
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi
done

log "ALL WAVES DONE total_wave_fail=$TOTAL_FAIL"
STATE_FILE="${STATE_FILE:-/tmp/pipeline_spatial_signs.state}"
printf '%s\n' "stage=eval_done" "updated=$(date -Is)" "wave_fail=$TOTAL_FAIL" \
  "METRICS_ROOT=$METRICS_ROOT" "TAG_SUFFIX=$TAG_SUFFIX" "SLOTS=${SLOTS[*]}" \
  > "$STATE_FILE"
