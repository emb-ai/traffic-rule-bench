#!/usr/bin/env bash
# Parallel plant2-ft eval queue (sign ckpts first, then baseline subset).
# Per checkpoint:
#   1) eval_checkpoint_on_test.py  -> plant2_rule_test/output/<tag>/  (policy=plant2)
#   2) run_eval_fast_plant2ft.sh   -> plant2_ft_metrics/<tag>/fv_fast/       (v61 speed catalog)
#   3) run_eval_fast_plant2ft.sh   -> plant2_ft_metrics/<tag>/fv_fast_detour/ (detour 4.2.x)
#
# Resume-safe: skips steps that already have summary/report markers.
# GPU safety: round-robin CUDA_VISIBLE_DEVICES; hard cap PROCS_PER_GPU (default 50).
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

# --- parallelism (GPU memory: ~0.9 GiB/process → 50/card keeps ~45 GiB of 80) ---
GPUS=(${GPUS:-0 1 2 3 4 5 6})
PROCS_PER_GPU=${PROCS_PER_GPU:-50}
SIGNS_PARALLEL=${SIGNS_PARALLEL:-8}          # concurrent checkpoints for signs eval
SIGNS_JOBS=${SIGNS_JOBS:-20}                 # workers inside one checkpoint
SCENES_PER_JOB=${SCENES_PER_JOB:-32}         # scenes per run_benchmark process
FV_PARALLEL=${FV_PARALLEL:-4}               # concurrent FV checkpoints
FV_NSHARDS=${FV_NSHARDS:-28}
FV_CONCURRENCY=${FV_CONCURRENCY:-28}

# catalogs / scenes for run_eval_fast
NFS2=/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova
MANIFEST_V61=${MANIFEST_V61:-$NFS2/traffic-rule-bench/pdd-bench/benchmark_output_speed/balanced/run_v61_a6/catalog_fv_test20.jsonl}
SCENES_V61=${SCENES_V61:-$NFS2/traffic-rule-bench/pdd-bench/scenes_balanced}
MANIFEST_DETOUR=${MANIFEST_DETOUR:-$NFS2/traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl}
SCENES_DETOUR=${SCENES_DETOUR:-$NFS2/sdc/pdd-bench/scenes}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 TORCH_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export SDL_AUDIODRIVER=dummy
export PER_SIGN_COMPLIANT_NPC=1

# --- tag mapping ---
# best_002_fvexp30_sign_lr1e5_1.ckpt -> fvexp30_sign_lr1e5_best002
# last_ft_fvexp30_sign_lr1e5_1.ckpt  -> fvexp30_sign_lr1e5_lastft
# best_000_fvexp30_lr1e5_1.ckpt      -> fvexp30_lr1e5_best000
# epoch=029_fvexp30_lr1e5_1.ckpt     -> fvexp30_lr1e5_ep029
tag_from_ckpt () {
  local bn; bn=$(basename "$1")
  if [[ "$bn" =~ best_([0-9]+)_fvexp30_sign_(lr[0-9e]+)_ ]]; then
    printf 'fvexp30_sign_%s_best%s' "${BASH_REMATCH[2]}" "${BASH_REMATCH[1]}"
  elif [[ "$bn" =~ last_ft_fvexp30_sign_(lr[0-9e]+)_ ]]; then
    printf 'fvexp30_sign_%s_lastft' "${BASH_REMATCH[1]}"
  elif [[ "$bn" =~ best_([0-9]+)_fvexp30_(lr[0-9e]+)_ ]]; then
    printf 'fvexp30_%s_best%s' "${BASH_REMATCH[2]}" "${BASH_REMATCH[1]}"
  elif [[ "$bn" =~ epoch=([0-9]+)_fvexp30_(lr[0-9e]+)_ ]]; then
    printf 'fvexp30_%s_ep%s' "${BASH_REMATCH[2]}" "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

# Wave 1: sign (8). Wave 2: baseline best+ep029 (8).
build_ckpt_list () {
  local lr addon
  for lr in 1e4 1e5 3e5 7e5; do
    addon=fvexp30_sign_lr${lr}
    for f in best_002_${addon}_1.ckpt last_ft_${addon}_1.ckpt; do
      [[ -f "$CKPT_ROOT/$addon/$f" ]] && echo "$CKPT_ROOT/$addon/$f"
    done
  done
  for lr in 1e4 1e5 3e5 7e5; do
    addon=fvexp30_lr${lr}
    for f in best_000_${addon}_1.ckpt epoch=029_${addon}_1.ckpt; do
      [[ -f "$CKPT_ROOT/$addon/$f" ]] && echo "$CKPT_ROOT/$addon/$f"
    done
  done
}

signs_done () { [[ -f "$SIGNS_OUT/$1/_summary/summary.md" ]]; }
fv_done () { [[ -f "$METRICS_ROOT/$1/$2/reports/report_cumulative.md" ]]; }

setup_tag_dir () {
  local tag="$1" ckpt="$2"
  mkdir -p "$METRICS_ROOT/$tag/fv_fast" "$METRICS_ROOT/$tag/fv_fast_detour" "$METRICS_ROOT/$tag/logs"
  printf '%s\n' "$ckpt" > "$METRICS_ROOT/$tag/ckpt.txt"
  mkdir -p "$SIGNS_OUT/$tag"
  ln -sfn "$SIGNS_OUT/$tag" "$METRICS_ROOT/$tag/signs"
}

# Soft GPU budget check: refuse to launch if SIGNS_JOBS > PROCS_PER_GPU
if (( SIGNS_JOBS > PROCS_PER_GPU )); then
  echo "ERROR: SIGNS_JOBS=$SIGNS_JOBS exceeds PROCS_PER_GPU=$PROCS_PER_GPU" >&2
  exit 1
fi

run_signs_one () {
  local tag="$1" ckpt="$2" gpu="$3" log rc
  log="$METRICS_ROOT/$tag/logs/eval_checkpoint.log"
  if signs_done "$tag"; then
    echo "[$(date -Is)] SIGNS SKIP done tag=$tag"
    return 0
  fi
  echo "[$(date -Is)] SIGNS START tag=$tag gpu=$gpu jobs=$SIGNS_JOBS scenes_per_job=$SCENES_PER_JOB"
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
  ) >"$log" 2>&1
  rc=$?
  ln -sfn "$SIGNS_OUT/$tag" "$METRICS_ROOT/$tag/signs"
  echo "[$(date -Is)] SIGNS DONE tag=$tag rc=$rc"
  # Prove sign_emb path from adapter log
  rg -n "sign_emb=" "$log" | head -3 || true
  return $rc
}

run_fv_one () {
  local tag="$1" ckpt="$2" out_subdir="$3" manifest="$4" scenes="$5" log rc
  log="$METRICS_ROOT/$tag/logs/run_eval_fast_${out_subdir}.log"
  if fv_done "$tag" "$out_subdir"; then
    echo "[$(date -Is)] FV SKIP done tag=$tag out=$out_subdir"
    return 0
  fi
  if [[ ! -f "$manifest" ]]; then
    echo "[$(date -Is)] FV FAIL missing MANIFEST=$manifest" | tee -a "$log"
    return 1
  fi
  if [[ ! -d "$scenes" ]]; then
    echo "[$(date -Is)] FV FAIL missing SCENES=$scenes" | tee -a "$log"
    return 1
  fi
  echo "[$(date -Is)] FV START tag=$tag out=$out_subdir"
  CKPT="$ckpt" OUT="$METRICS_ROOT/$tag/$out_subdir" TAG="$tag" \
    NN_POLICIES=plant2 \
    MANIFEST="$manifest" SCENES="$scenes" \
    GPUS="${GPUS[*]}" NSHARDS="$FV_NSHARDS" CONCURRENCY="$FV_CONCURRENCY" \
    bash "$FAST" >"$log" 2>&1
  rc=$?
  echo "[$(date -Is)] FV DONE tag=$tag out=$out_subdir rc=$rc"
  return $rc
}

# --- build list ---
mapfile -t CKPTS < <(build_ckpt_list)
echo "=== plant2-ft PARALLEL eval queue $(date -Is) ==="
echo "checkpoints: ${#CKPTS[@]}"
echo "metrics root: $METRICS_ROOT"
echo "GPUS=${GPUS[*]} SIGNS_PARALLEL=$SIGNS_PARALLEL SIGNS_JOBS=$SIGNS_JOBS SCENES_PER_JOB=$SCENES_PER_JOB"
echo "FV_PARALLEL=$FV_PARALLEL FV_NSHARDS=$FV_NSHARDS FV_CONCURRENCY=$FV_CONCURRENCY"
echo "MANIFEST_V61=$MANIFEST_V61"
echo "SCENES_V61=$SCENES_V61"
echo "MANIFEST_DETOUR=$MANIFEST_DETOUR"
echo "SCENES_DETOUR=$SCENES_DETOUR"

INDEX="$METRICS_ROOT/INDEX.md"
{
  echo "# plant2-ft metrics (parallel queue)"
  echo
  echo "Started: $(date -Is)"
  echo
  echo "| tag | ckpt | signs | fv_fast | fv_fast_detour |"
  echo "|---|---|---|---|---|"
} > "$INDEX"

declare -a TAGS=()
for ckpt in "${CKPTS[@]}"; do
  tag=$(tag_from_ckpt "$ckpt") || continue
  TAGS+=("$tag")
  setup_tag_dir "$tag" "$ckpt"
  s=pending; f=pending; d=pending
  signs_done "$tag" && s=done
  fv_done "$tag" fv_fast && f=done
  fv_done "$tag" fv_fast_detour && d=done
  echo "| \`$tag\` | \`$(basename "$ckpt")\` | $s | $f | $d |" >> "$INDEX"
done
echo "indexed ${#CKPTS[@]} checkpoints"

# Split waves: first 8 = sign, rest = baseline
WAVE1=("${CKPTS[@]:0:8}")
WAVE2=("${CKPTS[@]:8}")

run_signs_wave () {
  local -n wave=$1
  local i=0 ckpt tag gpu pids=()
  echo
  echo "======== SIGNS WAVE (${#wave[@]} ckpts, parallel=$SIGNS_PARALLEL) ========"
  for ckpt in "${wave[@]}"; do
    tag=$(tag_from_ckpt "$ckpt") || continue
    setup_tag_dir "$tag" "$ckpt"
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    # throttle to SIGNS_PARALLEL
    while (( $(jobs -rp | wc -l) >= SIGNS_PARALLEL )); do sleep 15; done
    run_signs_one "$tag" "$ckpt" "$gpu" &
    pids+=($!)
    i=$((i + 1))
  done
  wait
  echo "======== SIGNS WAVE DONE ========"
}

run_fv_wave () {
  local -n wave=$1
  local ckpt tag
  echo
  echo "======== FV WAVE (${#wave[@]} ckpts) ========"
  for ckpt in "${wave[@]}"; do
    tag=$(tag_from_ckpt "$ckpt") || continue
    setup_tag_dir "$tag" "$ckpt"
    while (( $(jobs -rp | wc -l) >= FV_PARALLEL )); do sleep 20; done
    (
      run_fv_one "$tag" "$ckpt" fv_fast "$MANIFEST_V61" "$SCENES_V61"
      run_fv_one "$tag" "$ckpt" fv_fast_detour "$MANIFEST_DETOUR" "$SCENES_DETOUR"
    ) &
  done
  wait
  echo "======== FV WAVE DONE ========"
}

# Refresh INDEX statuses
refresh_index () {
  {
    echo "# plant2-ft metrics (parallel queue)"
    echo
    echo "Updated: $(date -Is)"
    echo
    echo "| tag | ckpt | signs | fv_fast | fv_fast_detour |"
    echo "|---|---|---|---|---|"
    for ckpt in "${CKPTS[@]}"; do
      tag=$(tag_from_ckpt "$ckpt") || continue
      s=pending; f=pending; d=pending
      signs_done "$tag" && s=done
      fv_done "$tag" fv_fast && f=done
      fv_done "$tag" fv_fast_detour && d=done
      echo "| \`$tag\` | \`$(basename "$ckpt")\` | $s | $f | $d |"
    done
  } > "$INDEX"
}

run_signs_wave WAVE1
run_fv_wave WAVE1
run_signs_wave WAVE2
run_fv_wave WAVE2
refresh_index

echo "=== QUEUE COMPLETE $(date -Is) ==="
echo "INDEX: $INDEX"
