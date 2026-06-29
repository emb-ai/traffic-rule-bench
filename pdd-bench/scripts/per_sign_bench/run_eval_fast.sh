#!/usr/bin/env bash
# run_eval_fast.sh — faster eval: per-BASELINE parallelism + run_benchmark.py
# DIRECTLY (no per-job metrics) + a SINGLE aggregate at the end.
#
# vs run_eval_8gpu.sh (which runs eval_pipeline per shard×stream, with each job
# running its stream's baselines SEQUENTIALLY and rebuilding metrics ~96x):
#   - every (policy, ego-variant) is its own job → all baselines run in parallel
#   - calls run_benchmark.py directly → sim only, no per-job build/aggregate/report
#   - GPU baselines pinned one per GPU (round-robin); CPU baselines in a CPU pool
#     running CONCURRENTLY with the GPU pool
#   - metrics built ONCE at the end from episodes_*.jsonl
#
#   nohup bash scripts/per_sign_bench/run_eval_fast.sh > /tmp/eval_fast.log 2>&1 & disown
#
# ---- config (EDIT for the server) -------------------------------------------
REPO=/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench
MANIFEST=$REPO/benchmark_output_speed/balanced/check_100x5/mani/sumo_manifest.jsonl
SCENES=$REPO/scenes_balanced
OUT=$REPO/benchmark_output_speed/balanced/check_100x5/eval_fast

NGPU=8
GPUS="0 1 2 3 4 5 6 7"
NSHARDS=8                   # scene shards per baseline (more = finer balance, more ckpt loads)
CONCURRENCY=16             # GPU-baseline workers (NN: carl/plant2 families); ~2 per GPU
CONCURRENCY_CPU=24         # CPU-baseline workers (idm-family / rule_compliant / ppo_lidar)
MAX_STEPS=1500

# Skip end-of-zone signs (harmless if the manifest has none).
EXCLUDE_CODES="3.25 5.22 5.32"

PY=/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/plant2/bin/python
PLANT2_ACTION_MODE=pid

# ego-variants for IDM-family (idm, comprehensive_rule_expert). NN + rule_compliant
# + ppo_lidar always run once (default). Use "default" for a leaner/faster run.
EGO_VARIANTS="default,s1,s2,s3,s4"

# Baselines. Empty string disables that group.
NN_POLICIES="carl carl_rule plant2 plant2_rule"          # GPU, need a checkpoint
IDM_FAMILY_POLICIES="idm comprehensive_rule_expert"      # CPU, run per ego-variant
CPU_SINGLE_POLICIES="rule_compliant ppo_lidar"           # CPU, run once (default)

CARL_CKPT=/home/jovyan/shares/SR006.nfs2/smirnova/sdc/CaRL/nuPlan/checkpoints/nuplan_51479_1B/model_best.pth
PLANT_CKPT=/home/jovyan/shares/SR006.nfs2/smirnova/sdc/pdd-bench/checkpoints/epoch%3D029_final_3.ckpt
# -----------------------------------------------------------------------------

export PER_SIGN_COMPLIANT_NPC=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 TORCH_NUM_THREADS=1
export SDL_AUDIODRIVER=dummy
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null; chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

set -uo pipefail
cd "$REPO"
SHARD_DIR="$OUT/shards"
mkdir -p "$SHARD_DIR" "$OUT/logs" "$OUT/parts"
GPU_ARR=($GPUS)
BENCH=scripts/per_sign_bench

# ---- 0. drop end-of-zone codes (exact sign_code match) ----------------------
SRC_MANIFEST="$MANIFEST"
if [ -n "${EXCLUDE_CODES// /}" ]; then
  EXCL_RE=$(printf '%s' "$EXCLUDE_CODES" | sed -e 's/\./\\./g' -e 's/  */|/g')
  FMANIFEST="$OUT/manifest_filtered.jsonl"
  grep -Ev "\"sign_code\":[[:space:]]*\"(${EXCL_RE})\"" "$MANIFEST" > "$FMANIFEST" || true
  echo "manifest: $(wc -l < "$MANIFEST") -> $(wc -l < "$FMANIFEST") rows (dropped: $EXCLUDE_CODES)"
  SRC_MANIFEST="$FMANIFEST"
fi

# ---- 1. round-robin shards --------------------------------------------------
rm -f "$SHARD_DIR"/shard_*.jsonl
awk -v n="$NSHARDS" -v dir="$SHARD_DIR" 'NF{ f=sprintf("%s/shard_%02d.jsonl",dir,(NR-1)%n); print >> f }' "$SRC_MANIFEST"
echo "shards: $(ls "$SHARD_DIR"/shard_*.jsonl | wc -l) x ~$(( $(wc -l < "$SRC_MANIFEST") / NSHARDS )) rows"

# ---- 2. job lists: one per (policy, ego-variant, shard) ---------------------
# spec = "policy|variant|model|shardfile|tag|runname"
gpu_jobs=(); cpu_jobs=()
add_shards () {  # where(gpu|cpu) policy variant model
  local where="$1" policy="$2" variant="$3" model="$4" s sf tag rn
  rn="${policy}_${variant}"
  for s in $(seq 0 $((NSHARDS-1))); do
    sf=$(printf "%s/shard_%02d.jsonl" "$SHARD_DIR" "$s")
    tag=$(printf "%s_s%02d" "$rn" "$s")
    if [ "$where" = cpu ]; then cpu_jobs+=("$policy|$variant|$model|$sf|$tag|$rn")
    else                        gpu_jobs+=("$policy|$variant|$model|$sf|$tag|$rn"); fi
  done
}
for p in $NN_POLICIES; do
  case "$p" in carl|carl_rule) ck="$CARL_CKPT";; plant2|plant2_rule) ck="$PLANT_CKPT";; *) ck="";; esac
  add_shards gpu "$p" default "$ck"
done
for p in $IDM_FAMILY_POLICIES; do
  for v in ${EGO_VARIANTS//,/ }; do add_shards cpu "$p" "$v" ""; done
done
for p in $CPU_SINGLE_POLICIES; do add_shards cpu "$p" default ""; done
echo "gpu jobs: ${#gpu_jobs[@]} (NN x $NSHARDS) on $NGPU GPUs; cpu jobs: ${#cpu_jobs[@]}"

# ---- 3. run one baseline-shard (run_benchmark.py directly, no metrics) -------
run_job () {  # gpu spec   (gpu="cpu" -> hide CUDA)
  local gpu="$1" spec="$2" policy variant model sf tag rn cvd rc
  IFS='|' read -r policy variant model sf tag rn <<< "$spec"
  cvd="$gpu"; [ "$gpu" = cpu ] && cvd=""
  local args=( --policy "$policy" --run-name "$rn" --ego-variant "$variant"
    --manifest "$sf" --scenes-root "$SCENES" --backends sumo
    --max-steps "$MAX_STEPS" --benchmark-output "$OUT/parts/$tag/benchmark" )
  [ -n "$model" ] && args+=( --model-path "$model" )
  case "$policy" in plant2|plant2_rule) args+=( --plant2-action-mode "$PLANT2_ACTION_MODE" );; esac
  echo "[$(date +%H:%M:%S)] gpu=$gpu START $tag ($(wc -l < "$sf") scenes)"
  CUDA_VISIBLE_DEVICES="$cvd" "$PY" "$BENCH/run_benchmark.py" "${args[@]}" \
    >> "$OUT/logs/$tag.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] gpu=$gpu DONE  $tag rc=$rc"
}

# ---- 4. two pools running CONCURRENTLY: GPU (NN) + CPU (rule-based) ----------
gpu_lane () { local li="$1" gpu="${GPU_ARR[$(( li % NGPU ))]}" i
  for i in "${!gpu_jobs[@]}"; do
    [ $(( i % CONCURRENCY )) -eq "$li" ] && run_job "$gpu" "${gpu_jobs[$i]}"
  done; }
cpu_lane () { local li="$1" i
  for i in "${!cpu_jobs[@]}"; do
    [ $(( i % CONCURRENCY_CPU )) -eq "$li" ] && run_job cpu "${cpu_jobs[$i]}"
  done; }
echo "=== launching $CONCURRENCY GPU workers + $CONCURRENCY_CPU CPU workers ==="
for li in $(seq 0 $((CONCURRENCY-1)));      do gpu_lane "$li" & done
for li in $(seq 0 $((CONCURRENCY_CPU-1)));  do cpu_lane "$li" & done
wait
echo "=== all baselines done ==="

# ---- 5. metrics ONCE: per-part CSV from episodes -> merge -> aggregate -> MD -
EMPTY="$OUT/_no_manifests"; mkdir -p "$EMPTY"
COMB="$OUT/metrics_per_episode.csv"; : > "$COMB"
while read -r pe; do
  [ -n "$pe" ] || continue
  [ -n "$(find "$pe" -name 'episodes_*.jsonl' -print -quit 2>/dev/null)" ] || continue
  tag=$(basename "$(dirname "$(dirname "$pe")")")
  csv="$OUT/parts/_csv_$tag.csv"
  "$PY" "$BENCH/build_episode_metrics_csv.py" --episodes-root "$pe" \
      --out "$csv" --manifests-root "$EMPTY" >/dev/null 2>&1 || continue
  if [ -s "$COMB" ]; then tail -n +2 "$csv" >> "$COMB"; else cat "$csv" > "$COMB"; fi
done < <(find "$OUT/parts" -type d -name policy_eval | sort)
echo "combined CSV: $COMB ($(($(wc -l < "$COMB")-1)) episodes)"

"$PY" "$BENCH/aggregate_episode_metrics.py" --csv "$COMB" --out-dir "$OUT"
"$PY" "$BENCH/generate_cumulative_markdown_report.py" --run-root "$OUT"
echo "REPORT: $OUT/reports/report_cumulative.md"
