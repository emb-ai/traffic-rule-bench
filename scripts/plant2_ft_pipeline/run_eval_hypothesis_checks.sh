#!/usr/bin/env bash
# Quick hypothesis checks for spatial FT eval (~5 min each, smoke where GPU needed).
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
SIGNS="$SHEPELEV/traffic-rule-bench/pdd-bench/scripts/per_sign_bench/plant2_rule_test"
CKPT_ROOT="$SHEPELEV/traffic-rule-bench/plant2/PlanT/checkpoints_ft/fvexp30_spatial_lr1e4"
CKPT=$(ls "$CKPT_ROOT"/best_*_1.ckpt 2>/dev/null | head -1)
LOG=/tmp/eval_hypothesis_results.log
OUT=/tmp/eval_hypothesis_smoke

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
export PYTHONNOUSERSITE=1 SDL_AUDIODRIVER=dummy PER_SIGN_COMPLIANT_NPC=1

exec > >(tee -a "$LOG") 2>&1
echo "======== eval hypothesis checks $(date -Is) ========"
echo "CKPT=$CKPT"
echo "LOG=$LOG"

# --- H4: manifest guard ---
echo; echo "=== H4 catalog_fv_test20 guard ==="
MANIFEST=$("$PY" - <<'PY'
import re, sys
p = open("$PIPELINE_DIR/run_eval_fast_plant2ft.sh").read()
m = re.search(r'MANIFEST=\$\{MANIFEST:-([^}]+)\}', p)
print(m.group(1) if m else "MISSING")
PY
)
echo "default MANIFEST=$MANIFEST"
if [[ "$MANIFEST" != *catalog_fv_test20.jsonl* ]]; then
  echo "H4 FAIL: default manifest is not catalog_fv_test20"
else
  echo "H4 PASS: default is catalog_fv_test20"
fi

# --- H8 ---
echo; echo "=== H8 plant2-only ==="
rg -n 'NN_POLICIES=\$\{NN_POLICIES:-plant2\}' "$SHEPELEV/collected_trajectories/run_eval_fast_plant2ft.sh" \
  && echo "H8 PASS" || echo "H8 FAIL"

# --- H5: thread counts ---
echo; echo "=== H5 thread oversubscription ==="
(
  unset OMP_NUM_THREADS MKL_NUM_THREADS TORCH_NUM_THREADS
  "$PY" -c "import torch; print('default torch threads:', torch.get_num_threads())"
)
OMP_NUM_THREADS=1 "$PY" -c "import torch; print('with OMP=1:', torch.get_num_threads())"

if [[ -f "$CKPT" ]]; then
  echo "H5 smoke: eval 2 scenes with OMP=1 vs inherited (if GPU free)"
  if nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F', ' '$2+0<10{found=1} END{exit !found}'; then
    GPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F', ' '$2+0<10{print $1; exit}')
    echo "using idle GPU $GPU"
    t0=$(date +%s)
    (cd "$SIGNS" && CUDA_VISIBLE_DEVICES=$GPU "$PY" -u eval_checkpoint_on_test.py \
      --policies plant2 --model-paths "plant2:$CKPT" --only 2.1 --n-scenes 2 \
      --jobs 4 --scenes-per-job 1 --run-name h5_omp1 --keep-going) >/dev/null
    t1=$(date +%s)
    echo "H5 smoke wall_s=$((t1-t0)) with OMP=1"
  else
    echo "H5 SKIP GPU smoke (all GPUs busy with FT); static check only"
    echo "H5 VERDICT: enforce OMP=1 in all eval launchers (default torch=112 without it)"
  fi
else
  echo "H5 SKIP: no probe ckpt"
fi

# --- H7: NFS cold vs warm ---
echo; echo "=== H7 NFS read latency ==="
SCENE_ROOT="$SHEPELEV/traffic-rule-bench/pdd-bench/scripts/per_sign_bench/main_sign/scenes/2_1"
if [[ -d "$SCENE_ROOT" ]]; then
  ROUTE=$(find "$SCENE_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
  if [[ -n "$ROUTE" && -d "$ROUTE/boxes" ]]; then
    LIST=$(find "$ROUTE/boxes" -name '*.json.gz' 2>/dev/null | head -40)
  if [[ -n "$LIST" ]]; then
    t0=$(date +%s.%N)
    echo "$LIST" | xargs cat >/dev/null 2>&1
    t1=$(date +%s.%N)
    cold=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    t0=$(date +%s.%N)
    echo "$LIST" | xargs cat >/dev/null 2>&1
    t1=$(date +%s.%N)
    warm=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    echo "H7 cold_s=$cold warm_s=$warm ratio=$(awk -v c="$cold" -v w="$warm" 'BEGIN{if(w>0)printf "%.1f", c/w; else print "n/a"}')"
  else
    echo "H7 SKIP: no json.gz under $SCENE_ROOT"
  fi
else
  echo "H7 SKIP: scene root missing"
fi

# --- H6: need GPU — sample during FV smoke or defer ---
echo; echo "=== H6 CPU-bound / GPU idle ==="
if nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F', ' '$2+0<10{found=1} END{exit !found}'; then
  GPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F', ' '$2+0<10{print $1; exit}')
  FAST_OUT="$OUT/h6_fv"
  rm -rf "$FAST_OUT"
  mkdir -p "$FAST_OUT"
  NFS2=/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/traffic-rule-bench/pdd-bench
  MANIFEST="$NFS2/benchmark_output_speed/balanced/run_v61_a6/catalog_fv_test20.jsonl"
  SCENES="$NFS2/scenes_balanced"
  if [[ -f "$MANIFEST" && -f "$CKPT" ]]; then
    echo "H6: mini FV 2 shards CONCURRENCY=2 on GPU $GPU (background nvidia-smi)"
    nvidia-smi dmon -s u -d 2 -c 15 > "$OUT/h6_gpu_util.log" 2>&1 &
    DMON=$!
    CKPT="$CKPT" OUT="$FAST_OUT" NN_POLICIES=plant2 GPUS="$GPU" NSHARDS=2 CONCURRENCY=2 \
      MANIFEST="$MANIFEST" SCENES="$SCENES" MAX_STEPS=200 \
      bash "$SHEPELEV/collected_trajectories/run_eval_fast_plant2ft.sh" || true
    wait $DMON 2>/dev/null || true
    echo "H6 gpu util sample:"
    awk '{if(NR>2){gsub(/[^0-9]/,"",$2); if($2!=""){s+=$2; n++}}} END{if(n)printf "avg_gpu_util=%.0f%% samples=%d\n", s/n, n}' "$OUT/h6_gpu_util.log"
    echo "H6 cpu during run: see vmstat in log"
  fi
else
  echo "H6 SKIP FV smoke (GPUs busy). Training snapshot:"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  echo "H6 NOTE: FT uses ~85-95% GPU; eval sim expected lower util → scale jobs/CONCURRENCY not GPU count"
fi

# --- H1: rule-signs jobs/spj (if GPU free) ---
echo; echo "=== H1 jobs x scenes-per-job ==="
if [[ -f "$CKPT" ]] && nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F', ' '$2+0<10{found=1} END{exit !found}'; then
  GPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F', ' '$2+0<10{print $1; exit}')
  for cfg in "8 1" "20 32"; do
    set -- $cfg; J=$1; SPJ=$2; TAG="h1_j${J}_spj${SPJ}"
    t0=$(date +%s)
    (cd "$SIGNS" && CUDA_VISIBLE_DEVICES=$GPU "$PY" -u eval_checkpoint_on_test.py \
      --policies plant2 --model-paths "plant2:$CKPT" --only 2.1,2.4 --n-scenes 4 \
      --jobs "$J" --scenes-per-job "$SPJ" --run-name "$TAG" --keep-going) >/dev/null
    echo "H1 $TAG wall_s=$(( $(date +%s) - t0 ))"
  done
else
  echo "H1 SKIP (GPUs busy); use jobs=20 spj=32 from queue defaults"
fi

echo; echo "======== done $(date -Is) → $LOG ========"
