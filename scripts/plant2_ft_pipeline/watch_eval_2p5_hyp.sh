#!/usr/bin/env bash
# Wait for hyp FT ckpts, then eval sign 2.5 (best + ep029/last) on free GPUs.
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

CT="$PIPELINE_DIR"
PLAN_T="$TRB_ROOT/plant2/PlanT"
CKPT_ROOT="$PLAN_T/checkpoints_ft"
METRICS_ROOT="${METRICS_ROOT:-$SHEPELEV/plant2_ft_metrics/spatial_2p5_hyp_eval_sign25}"
SIGNS_DIR="$SHEPELEV/traffic-rule-bench/pdd-bench/scripts/per_sign_bench/plant2_rule_test"
SIGNS_OUT="$SIGNS_DIR/output"
PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
LOGDIR="$CT/logs_pipeline_2p5_hyp"
mkdir -p "$METRICS_ROOT" "$LOGDIR"

POLL_SEC="${POLL_SEC:-180}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-86400}"
ONLY_SIGNS="${ONLY_SIGNS:-2.5}"
SIGNS_JOBS=${SIGNS_JOBS:-8}
SCENES_PER_JOB=${SCENES_PER_JOB:-20}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SDL_AUDIODRIVER=dummy
export PER_SIGN_COMPLIANT_NPC=1

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/watch_eval.log"; }

ADDONS=(
  fvexp30_2p5_h1_path0_sw5_lr1e4
  fvexp30_2p5_h1_path0_sw5_lr1e5
  fvexp30_2p5_h2_cw15_lr1e4
  fvexp30_2p5_h2_cw15_lr1e5
  fvexp30_2p5_h1h2_path0_sw5_cw15_lr1e4
  fvexp30_2p5_h1h2_path0_sw5_cw15_lr1e5
  fvexp30_2p5_h5_noaug_lr1e5
)

pick_best() { ls -1t "$1"/best_*.ckpt 2>/dev/null | head -1 || true; }
pick_last() { ls -1t "$1"/epoch=*.ckpt 2>/dev/null | head -1 || true; }

make_tag() {
  local addon="$1" slot="$2" ckpt="$3"
  local bn; bn=$(basename "$ckpt")
  local base="${addon}"
  if [[ "$slot" == best && "$bn" =~ best_([0-9]+)_ ]]; then
    printf '%s_best%s_sign25' "$base" "${BASH_REMATCH[1]}"
  elif [[ "$bn" =~ epoch=([0-9]+)_ ]]; then
    printf '%s_ep%s_sign25' "$base" "${BASH_REMATCH[1]}"
  else
    printf '%s_%s_sign25' "$base" "$slot"
  fi
}

signs_done() { [[ -f "$SIGNS_OUT/$1/_summary/summary.md" ]]; }

ft_finished() {
  local a
  for a in "${ADDONS[@]}"; do
    [[ -f "$(pick_last "$CKPT_ROOT/$a")" ]] || return 1
  done
  if pgrep -af 'run_lit_finetune' | rg -q 'fvexp30_2p5_h(1|2|5|1h2)'; then
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

log "watch_eval_hyp start METRICS_ROOT=$METRICS_ROOT"
elapsed=0
while (( elapsed < MAX_WAIT_SEC )); do
  if ft_finished; then
    log "FT finished — launching eval"
    break
  fi
  ready=1
  for a in "${ADDONS[@]}"; do
    [[ -n "$(pick_last "$CKPT_ROOT/$a")" ]] || ready=0
  done
  if [[ "$ready" == 1 ]] && ! pgrep -af 'run_lit_finetune' | rg -q 'fvexp30_2p5_h(1|2|5|1h2)'; then
    log "ckpts present and FT processes gone"
    break
  fi
  n_done=0
  for a in "${ADDONS[@]}"; do
    [[ -n "$(pick_last "$CKPT_ROOT/$a")" ]] && n_done=$((n_done + 1))
  done
  log "waiting FT… done_addons=${n_done}/${#ADDONS[@]} elapsed=${elapsed}s"
  sleep "$POLL_SEC"
  elapsed=$((elapsed + POLL_SEC))
done

declare -a JOBS=()
gpu=0
for addon in "${ADDONS[@]}"; do
  for slot in best last; do
    if [[ "$slot" == best ]]; then ckpt=$(pick_best "$CKPT_ROOT/$addon"); else ckpt=$(pick_last "$CKPT_ROOT/$addon"); fi
    if [[ -z "$ckpt" ]]; then
      log "WARN missing $addon $slot"
      continue
    fi
    tag=$(make_tag "$addon" "$slot" "$ckpt")
    JOBS+=("$gpu|$tag|$ckpt")
    gpu=$(( (gpu + 1) % 7 ))
  done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
  log "ERROR no ckpts to eval"
  exit 1
fi

fail=0
wave=()
flush_wave() {
  local pids=() spec g tag ckpt pid
  for spec in "${wave[@]}"; do
    IFS='|' read -r g tag ckpt <<<"$spec"
    run_signs_one "$tag" "$ckpt" "$g" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=$((fail + 1))
  done
  wave=()
}

for spec in "${JOBS[@]}"; do
  wave+=("$spec")
  if (( ${#wave[@]} >= 7 )); then
    flush_wave
  fi
done
if (( ${#wave[@]} > 0 )); then
  flush_wave
fi

TABLE="$METRICS_ROOT/SIGN_SR_TABLE.md"
{
  echo "# Sign 2.5 SR — hyp FT (H1/H2/H1+H2/H5)"
  echo
  echo "Baseline tsfix: Sign SR ≈ 0.078 (lr1e5) / 0 (lr1e4)."
  echo
  echo "| tag | Sign SR | Dest rate | notes |"
  echo "|---|---|---|---|"
  declare -A seen=()
  for d in "$METRICS_ROOT"/fvexp30_*/signs/_summary/summary.md "$SIGNS_OUT"/fvexp30_2p5_h*/_summary/summary.md; do
    [[ -f "$d" ]] || continue
    if [[ "$d" == */signs/_summary/summary.md ]]; then
      tag=$(basename "$(dirname "$(dirname "$(dirname "$d")")")")
    else
      tag=$(basename "$(dirname "$(dirname "$d")")")
    fi
    [[ -n "${seen[$tag]:-}" ]] && continue
    seen[$tag]=1
    row=$(rg -N '^\| 2\.5 \|' "$d" 2>/dev/null | head -1 || true)
    sr=""; dest=""
    if [[ -n "$row" ]]; then
      sr=$(awk -F'|' '{gsub(/^ +| +$/,"",$6); print $6}' <<<"$row")
      dest=$(awk -F'|' '{gsub(/^ +| +$/,"",$5); print $5}' <<<"$row")
    fi
    note=""
    [[ "$tag" == *h1_path0* && "$tag" != *h1h2* ]] && note="H1"
    [[ "$tag" == *h2_cw15* ]] && note="H2"
    [[ "$tag" == *h1h2* ]] && note="H1+H2"
    [[ "$tag" == *h5_noaug* ]] && note="H5"
    [[ "$tag" == *lr1e5* ]] && note="${note} lr1e5"
    [[ "$tag" == *lr1e4* ]] && note="${note} lr1e4"
    echo "| $tag | ${sr:-?} | ${dest:-?} | $note |"
  done | sort
} > "$TABLE" 2>/dev/null || true

log "ALL DONE fail=$fail metrics=$METRICS_ROOT table=$TABLE"
exit "$fail"
