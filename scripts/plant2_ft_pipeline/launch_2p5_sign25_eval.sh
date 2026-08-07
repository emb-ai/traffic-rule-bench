#!/usr/bin/env bash
# Eval 2.5-only FT ckpts (best + ep029) on plant2_rule scenes for sign 2.5 only.
#
# Note: catalog_fv_test20 / detour_v1 contain no sign_code=2.5 rows, so fv_fast
# and fv_fast_detour are skipped. The relevant pipeline step is plant2_rule_test
# with --only 2.5 (153 scenes).
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

CT="$PIPELINE_DIR"
REPO="$TRB_ROOT"
CKPT_ROOT="$REPO/plant2/PlanT/checkpoints_ft"
METRICS_ROOT="${METRICS_ROOT:-$SHEPELEV/plant2_ft_metrics/spatial_2p5_eval_sign25}"
SIGNS_DIR="$REPO/pdd-bench/scripts/per_sign_bench/plant2_rule_test"
SIGNS_OUT="$SIGNS_DIR/output"
PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
LOGDIR="$CT/logs_pipeline_spatial_2p5_sign25"
mkdir -p "$METRICS_ROOT" "$LOGDIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SDL_AUDIODRIVER=dummy
export PER_SIGN_COMPLIANT_NPC=1

SIGNS_JOBS=${SIGNS_JOBS:-8}
SCENES_PER_JOB=${SCENES_PER_JOB:-20}
MAX_EVAL_RETRIES=${MAX_EVAL_RETRIES:-3}
ONLY_SIGNS="${ONLY_SIGNS:-2.5}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/orchestrator.log"; }

# gpu|lr|slot|ckpt_basename_glob_hint
# Explicit paths for clarity.
declare -a JOBS=(
  "0|1e4|best|$CKPT_ROOT/fvexp30_spatial_2p5_lr1e4/best_023_fvexp30_spatial_2p5_lr1e4_1.ckpt"
  "1|1e4|ep029|$CKPT_ROOT/fvexp30_spatial_2p5_lr1e4/epoch=029_fvexp30_spatial_2p5_lr1e4_1.ckpt"
  "2|1e5|best|$CKPT_ROOT/fvexp30_spatial_2p5_lr1e5/best_028_fvexp30_spatial_2p5_lr1e5_1.ckpt"
  "3|1e5|ep029|$CKPT_ROOT/fvexp30_spatial_2p5_lr1e5/epoch=029_fvexp30_spatial_2p5_lr1e5_1.ckpt"
)

make_tag() {
  local lr="$1" slot="$2" ckpt="$3"
  local bn
  bn=$(basename "$ckpt")
  if [[ "$slot" == best && "$bn" =~ best_([0-9]+)_ ]]; then
    printf 'fvexp30_spatial_2p5_lr%s_best%s_sign25' "$lr" "${BASH_REMATCH[1]}"
  elif [[ "$bn" =~ epoch=([0-9]+)_ ]]; then
    printf 'fvexp30_spatial_2p5_lr%s_ep%s_sign25' "$lr" "${BASH_REMATCH[1]}"
  else
    printf 'fvexp30_spatial_2p5_lr%s_%s_sign25' "$lr" "$slot"
  fi
}

signs_done() { [[ -f "$SIGNS_OUT/$1/_summary/summary.md" ]]; }

run_signs_one() {
  local tag="$1" ckpt="$2" gpu="$3"
  local logf="$METRICS_ROOT/$tag/logs/eval_checkpoint.log"
  local attempt=0 rc=0

  mkdir -p "$METRICS_ROOT/$tag/logs" "$SIGNS_OUT/$tag"
  ln -sfn "$SIGNS_OUT/$tag" "$METRICS_ROOT/$tag/signs"
  printf '%s\n' "$ckpt" > "$METRICS_ROOT/$tag/ckpt.txt"
  echo "ONLY_SIGNS=$ONLY_SIGNS" > "$METRICS_ROOT/$tag/eval_filter.txt"

  if signs_done "$tag"; then
    log "SIGNS SKIP $tag"
    return 0
  fi

  while (( attempt < MAX_EVAL_RETRIES )); do
    attempt=$((attempt + 1))
    if signs_done "$tag"; then
      log "SIGNS OK $tag"
      return 0
    fi
    log "SIGNS START tag=$tag gpu=$gpu only=$ONLY_SIGNS attempt=$attempt"
    (
      cd "$SIGNS_DIR"
      unset PYTHONPATH
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PY" -u eval_checkpoint_on_test.py \
        --policies plant2 \
        --model-paths "plant2:$ckpt" \
        --jobs "$SIGNS_JOBS" \
        --scenes-per-job "$SCENES_PER_JOB" \
        --only "$ONLY_SIGNS" \
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
      # Mark fv steps N/A for this 2.5-only run
      mkdir -p "$METRICS_ROOT/$tag/logs"
      cat >"$METRICS_ROOT/$tag/logs/fv_skipped.txt" <<EOF
fv_fast / fv_fast_detour skipped: catalog_fv_test20 and detour catalogs have no sign_code=2.5.
EOF
      return 0
    fi
    log "SIGNS FAIL tag=$tag rc=$rc attempt=$attempt/$MAX_EVAL_RETRIES"
    tail -n 40 "$logf" | sed 's/^/  /' || true
    sleep 10
  done
  log "SIGNS GAVE UP tag=$tag"
  return 1
}

run_job() {
  local gpu="$1" lr="$2" slot="$3" ckpt="$4"
  local tag
  if [[ ! -f "$ckpt" ]]; then
    log "MISSING ckpt=$ckpt"
    return 1
  fi
  tag=$(make_tag "$lr" "$slot" "$ckpt")
  log "JOB gpu=$gpu lr=$lr slot=$slot tag=$tag"
  run_signs_one "$tag" "$ckpt" "$gpu"
}

log "launch_2p5_sign25_eval start"
log "METRICS_ROOT=$METRICS_ROOT ONLY_SIGNS=$ONLY_SIGNS"
log "SKIP fv_fast/detour (no 2.5 rows in those catalogs)"

pids=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r gpu lr slot ckpt <<<"$spec"
  run_job "$gpu" "$lr" "$slot" "$ckpt" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done

log "ALL DONE fail=$fail"
echo "---- summaries ----"
for spec in "${JOBS[@]}"; do
  IFS='|' read -r gpu lr slot ckpt <<<"$spec"
  tag=$(make_tag "$lr" "$slot" "$ckpt")
  if signs_done "$tag"; then
    echo "OK $tag"
    cat "$SIGNS_OUT/$tag/_summary/summary.md"
    echo
  else
    echo "FAIL $tag"
  fi
done
exit "$fail"
