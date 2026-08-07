#!/usr/bin/env bash
# Wait for FT best/last ckpts for tsfix addons, then eval sign 2.5.
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

CT="$PIPELINE_DIR"
PLAN_T="$TRB_ROOT/plant2/PlanT"
CKPT_ROOT="$PLAN_T/checkpoints_ft"
METRICS_ROOT="${METRICS_ROOT:-$SHEPELEV/plant2_ft_metrics/spatial_2p5_tsfix_eval_sign25}"
SIGNS_DIR="$SHEPELEV/traffic-rule-bench/pdd-bench/scripts/per_sign_bench/plant2_rule_test"
SIGNS_OUT="$SIGNS_DIR/output"
PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
LOGDIR="$CT/logs_pipeline_2p5_tsfix"
mkdir -p "$METRICS_ROOT" "$LOGDIR"

POLL_SEC="${POLL_SEC:-120}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-72000}"  # 20h
ONLY_SIGNS="${ONLY_SIGNS:-2.5}"
SIGNS_JOBS=${SIGNS_JOBS:-8}
SCENES_PER_JOB=${SCENES_PER_JOB:-20}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SDL_AUDIODRIVER=dummy
export PER_SIGN_COMPLIANT_NPC=1

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/watch_eval.log"; }

ADDONS=(fvexp30_spatial_2p5_tsfix_lr1e4 fvexp30_spatial_2p5_tsfix_lr1e5)

pick_best() { ls -1t "$1"/best_*.ckpt 2>/dev/null | head -1 || true; }
pick_last() { ls -1t "$1"/epoch=*.ckpt 2>/dev/null | head -1 || true; }

make_tag() {
  local lr_tag="$1" slot="$2" ckpt="$3"
  local bn; bn=$(basename "$ckpt")
  if [[ "$slot" == best && "$bn" =~ best_([0-9]+)_ ]]; then
    printf 'fvexp30_spatial_2p5_tsfix_lr%s_best%s_sign25' "$lr_tag" "${BASH_REMATCH[1]}"
  elif [[ "$bn" =~ epoch=([0-9]+)_ ]]; then
    printf 'fvexp30_spatial_2p5_tsfix_lr%s_ep%s_sign25' "$lr_tag" "${BASH_REMATCH[1]}"
  else
    printf 'fvexp30_spatial_2p5_tsfix_lr%s_%s_sign25' "$lr_tag" "$slot"
  fi
}

signs_done() { [[ -f "$SIGNS_OUT/$1/_summary/summary.md" ]]; }

ft_finished() {
  # true if no run_lit_finetune for our addons, and epoch=029 exists for both
  local a
  for a in "${ADDONS[@]}"; do
    [[ -f "$(pick_last "$CKPT_ROOT/$a")" ]] || return 1
  done
  if pgrep -af 'run_lit_finetune' | rg -q 'fvexp30_spatial_2p5_tsfix'; then
    return 1
  fi
  return 0
}

run_signs_one() {
  local tag="$1" ckpt="$2" gpu="$3"
  local logf="$METRICS_ROOT/$tag/logs/eval_checkpoint.log"
  mkdir -p "$METRICS_ROOT/$tag/logs" "$SIGNS_OUT/$tag"
  ln -sfn "$SIGNS_OUT/$tag" "$METRICS_ROOT/$tag/signs"
  printf '%s\n' "$ckpt" > "$METRICS_ROOT/$tag/ckpt.txt"
  echo "ONLY_SIGNS=$ONLY_SIGNS" > "$METRICS_ROOT/$tag/eval_filter.txt"
  if signs_done "$tag"; then
    log "SIGNS SKIP $tag"
    return 0
  fi
  log "SIGNS START tag=$tag gpu=$gpu ckpt=$ckpt"
  (
    cd "$SIGNS_DIR"
    unset PYTHONPATH
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" -u eval_checkpoint_on_test.py \
      --policies plant2 --model-paths "plant2:$ckpt" \
      --jobs "$SIGNS_JOBS" --scenes-per-job "$SCENES_PER_JOB" \
      --only "$ONLY_SIGNS" --keep-going --run-name "$tag"
    "$PY" summarize_reports.py --run-name "$tag" --baseline plant2_default \
      --out-dir "output/$tag/_summary"
  ) >>"$logf" 2>&1
  if signs_done "$tag"; then
    log "SIGNS DONE $tag"
    return 0
  fi
  log "SIGNS FAIL $tag"; tail -40 "$logf" | sed 's/^/  /' || true
  return 1
}

log "watch_eval start METRICS_ROOT=$METRICS_ROOT"
elapsed=0
while (( elapsed < MAX_WAIT_SEC )); do
  if ft_finished; then
    log "FT finished — launching eval"
    break
  fi
  # Also start eval early once both best + last exist (optional: wait for final)
  ready=1
  for a in "${ADDONS[@]}"; do
    [[ -n "$(pick_last "$CKPT_ROOT/$a")" ]] || ready=0
  done
  if [[ "$ready" == 1 ]] && ! pgrep -af 'run_lit_finetune' | rg -q 'fvexp30_spatial_2p5_tsfix'; then
    log "ckpts present and FT processes gone"
    break
  fi
  log "waiting FT… elapsed=${elapsed}s"
  sleep "$POLL_SEC"
  elapsed=$((elapsed + POLL_SEC))
done

declare -a JOBS=()
gpu=0
for addon in "${ADDONS[@]}"; do
  lr_tag=${addon##*_lr}
  for slot in best last; do
    if [[ "$slot" == best ]]; then ckpt=$(pick_best "$CKPT_ROOT/$addon"); else ckpt=$(pick_last "$CKPT_ROOT/$addon"); fi
    if [[ -z "$ckpt" ]]; then
      log "WARN missing $addon $slot"
      continue
    fi
    tag=$(make_tag "$lr_tag" "$slot" "$ckpt")
    JOBS+=("$gpu|$tag|$ckpt")
    gpu=$(( (gpu + 1) % 7 ))
  done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
  log "ERROR no ckpts to eval"
  exit 1
fi

pids=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r g tag ckpt <<<"$spec"
  run_signs_one "$tag" "$ckpt" "$g" &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
log "ALL DONE fail=$fail metrics=$METRICS_ROOT"
exit "$fail"
