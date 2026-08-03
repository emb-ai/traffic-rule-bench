#!/usr/bin/env bash
# Queue plant2-ft evals: for each fvexp30 checkpoint (lr × epoch)
#   1) eval_checkpoint_on_test.py  -> plant2_rule_test/output/<tag>/  (policy=plant2)
#   2) run_eval_fast_plant2ft.sh   -> metrics/<tag>/fv_fast/   (policy=plant2, FT ckpt)
#
# Skips steps that already finished. Waits while other arbelyaev-eval-* are busy
# before starting a new signs eval (so we don't pile onto the current 4).
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

REPO=$SHEPELEV/traffic-rule-bench
CKPT_ROOT=$REPO/plant2/PlanT/checkpoints_ft
METRICS_ROOT=$SHEPELEV/plant2_ft_metrics
SIGNS_OUT=$REPO/pdd-bench/scripts/per_sign_bench/plant2_rule_test/output
SIGNS_DIR=$REPO/pdd-bench/scripts/per_sign_bench/plant2_rule_test
PY=$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python
FAST=$SHEPELEV/collected_trajectories/run_eval_fast_plant2ft.sh
LOG_DIR=$METRICS_ROOT/_queue_logs
mkdir -p "$METRICS_ROOT" "$LOG_DIR"

# Map filename -> tag
# best_000_fvexp30_lr1e5_1.ckpt -> fvexp30_lr1e5_best000
# epoch=004_fvexp30_lr1e5_1.ckpt -> fvexp30_lr1e5_ep004
tag_from_ckpt () {
  local f bn
  f="$1"; bn=$(basename "$f")
  # Keep zero-padded digits as strings (printf %d treats 009 as invalid octal).
  if [[ "$bn" =~ best_([0-9]+)_fvexp30_(lr[0-9e]+)_ ]]; then
    printf 'fvexp30_%s_best%s' "${BASH_REMATCH[2]}" "${BASH_REMATCH[1]}"
  elif [[ "$bn" =~ epoch=([0-9]+)_fvexp30_(lr[0-9e]+)_ ]]; then
    printf 'fvexp30_%s_ep%s' "${BASH_REMATCH[2]}" "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

# Legacy run-name used by already-running last/ep029 evals: fvexp30_lr1e5 / fvexp30_lr1e5_plant2
legacy_last_run () {
  local tag="$1"
  if [[ "$tag" =~ ^(fvexp30_lr[0-9e]+)_ep029$ ]]; then
    echo "${BASH_REMATCH[1]}_plant2"
    echo "${BASH_REMATCH[1]}"
  fi
}

signs_done () {
  local tag="$1" legacy
  if [[ -f "$SIGNS_OUT/$tag/_summary/summary.md" ]]; then return 0; fi
  while read -r legacy; do
    [[ -z "$legacy" ]] && continue
    if [[ -f "$SIGNS_OUT/$legacy/_summary/summary.md" ]]; then return 0; fi
  done < <(legacy_last_run "$tag")
  return 1
}

signs_running () {
  local tag="$1" legacy
  pgrep -af "eval_checkpoint_on_test.py" | grep -F "--run-name $tag" >/dev/null 2>&1 && return 0
  while read -r legacy; do
    [[ -z "$legacy" ]] && continue
    pgrep -af "eval_checkpoint_on_test.py" | grep -F "--run-name $legacy" >/dev/null 2>&1 && return 0
  done < <(legacy_last_run "$tag")
  return 1
}

fv_done () {
  local tag="$1"
  [[ -f "$METRICS_ROOT/$tag/fv_fast/reports/report_cumulative.md" ]]
}

any_signs_eval_busy () {
  pgrep -f "eval_checkpoint_on_test.py" >/dev/null 2>&1
}

setup_tag_dir () {
  local tag="$1" ckpt="$2" d legacy
  d="$METRICS_ROOT/$tag"
  mkdir -p "$d/fv_fast" "$d/logs"
  printf '%s\n' "$ckpt" > "$d/ckpt.txt"
  # link signs output (prefer exact tag; else plant2 legacy last name)
  if [[ -d "$SIGNS_OUT/$tag" ]]; then
    ln -sfn "$SIGNS_OUT/$tag" "$d/signs"
  else
    local linked=0
    while read -r legacy; do
      [[ -z "$legacy" ]] && continue
      if [[ -d "$SIGNS_OUT/$legacy" ]]; then
        ln -sfn "$SIGNS_OUT/$legacy" "$d/signs"
        linked=1
        break
      fi
    done < <(legacy_last_run "$tag")
    if [[ "$linked" -eq 0 ]]; then
      mkdir -p "$SIGNS_OUT/$tag"
      ln -sfn "$SIGNS_OUT/$tag" "$d/signs"
    fi
  fi
}

run_signs () {
  local tag="$1" ckpt="$2" run_name="$3" log
  log="$METRICS_ROOT/$tag/logs/eval_checkpoint.log"
  echo "[$(date -Is)] SIGNS START tag=$tag run_name=$run_name"
  (
    cd "$SIGNS_DIR"
    "$PY" -u eval_checkpoint_on_test.py \
      --policies plant2 \
      --model-paths "plant2:$ckpt" \
      --jobs 1 \
      --keep-going \
      --run-name "$run_name"
  ) >"$log" 2>&1
  local rc=$?
  echo "[$(date -Is)] SIGNS DONE tag=$tag rc=$rc"
  # refresh symlink
  ln -sfn "$SIGNS_OUT/$run_name" "$METRICS_ROOT/$tag/signs"
  return $rc
}

run_fv () {
  local tag="$1" ckpt="$2" log
  log="$METRICS_ROOT/$tag/logs/run_eval_fast.log"
  echo "[$(date -Is)] FV_FAST START tag=$tag"
  CKPT="$ckpt" OUT="$METRICS_ROOT/$tag/fv_fast" TAG="$tag" \
    NN_POLICIES=plant2 \
    GPUS="${GPUS:-0 1 2 3}" NSHARDS="${NSHARDS:-12}" CONCURRENCY="${CONCURRENCY:-8}" \
    bash "$FAST" >"$log" 2>&1
  local rc=$?
  echo "[$(date -Is)] FV_FAST DONE tag=$tag rc=$rc"
  return $rc
}

# Build ordered list: lr then epoch
mapfile -t CKPTS < <(
  find "$CKPT_ROOT"/fvexp30_lr* \( -name 'best_*.ckpt' -o -name 'epoch=*.ckpt' \) | sort
)

echo "=== plant2-ft eval queue $(date -Is) ==="
echo "checkpoints: ${#CKPTS[@]}"
echo "metrics root: $METRICS_ROOT"

# Create all folders first
INDEX="$METRICS_ROOT/INDEX.md"
{
  echo "# plant2-ft metrics"
  echo
  echo "| tag | ckpt | signs | fv_fast |"
  echo "|---|---|---|---|"
} > "$INDEX"

for ckpt in "${CKPTS[@]}"; do
  tag=$(tag_from_ckpt "$ckpt") || continue
  setup_tag_dir "$tag" "$ckpt"
  s_status=pending; f_status=pending
  signs_done "$tag" && s_status=done
  signs_running "$tag" && s_status=running
  fv_done "$tag" && f_status=done
  echo "| \`$tag\` | \`$(basename "$ckpt")\` | $s_status | $f_status |" >> "$INDEX"
done
echo "created/updated ${#CKPTS[@]} metric folders under $METRICS_ROOT"

# Process queue sequentially for resource safety
for ckpt in "${CKPTS[@]}"; do
  tag=$(tag_from_ckpt "$ckpt") || continue
  setup_tag_dir "$tag" "$ckpt"
  echo
  echo "======== $tag ========"
  echo "ckpt=$ckpt"

  # --- 1) signs eval ---
  if signs_done "$tag"; then
    echo "[skip] signs already done"
  elif signs_running "$tag"; then
    echo "[wait] signs already running; waiting for finish..."
    while signs_running "$tag"; do sleep 60; done
    echo "[wait] signs process ended"
  else
    # Wait so we don't stack on top of the current 4 parallel evals
    while any_signs_eval_busy; do
      echo "[wait] other eval_checkpoint busy; sleep 120s"
      sleep 120
    done
    # For ep029 prefer parallel *_plant2 run-name
    run_name="$tag"
    while read -r legacy; do
      [[ -z "$legacy" ]] && continue
      if [[ "$legacy" == *_plant2 ]]; then
        run_name="$legacy"
        break
      fi
    done < <(legacy_last_run "$tag")
    run_signs "$tag" "$ckpt" "$run_name" || echo "[warn] signs rc!=0 for $tag"
  fi

  # --- 2) fv fast (plant2 base + FT ckpt) ---
  if fv_done "$tag"; then
    echo "[skip] fv_fast already done"
  else
    # Prefer not overlapping heavy FV with signs evals
    while any_signs_eval_busy; do
      echo "[wait] signs evals still busy before fv_fast; sleep 120s"
      sleep 120
    done
    run_fv "$tag" "$ckpt" || echo "[warn] fv_fast rc!=0 for $tag"
  fi
done

echo "=== QUEUE COMPLETE $(date -Is) ==="
