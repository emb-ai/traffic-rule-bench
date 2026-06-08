#!/usr/bin/env bash
# Evaluate policies over the manifest, PARALLEL across N GPUs by SCENE-SHARDING.
#
# Why sharding: one run_benchmark process uses exactly ONE GPU and steps scenes
# sequentially. Listing many GPUs in CUDA_VISIBLE_DEVICES does NOT parallelize it.
# We split the 634-row manifest into NSHARDS shards and run CONCURRENCY worker
# processes in parallel. The real bottleneck is CPU (SUMO + MetaDrive sim step),
# not the GPU, so CONCURRENCY can exceed the GPU count — each worker is pinned to
# GPU (worker_index % NGPU), several workers may share a card. Each (stream x
# shard) is a job run by one worker. At the end all per-shard
# metrics_per_episode.csv are merged and one cumulative report is rebuilt.
#
# Streams (set a stream's *_POLICIES empty to disable it):
#   carl   : carl,carl_rule        (CaRL env,   needs CKPTS_CARL)
#   plant2 : plant2,plant2_rule    (PlanT2 env, needs CKPTS_PLANT)
#   rest   : idm,comprehensive_rule_expert,rule_compliant,ppo_lidar  (no ckpt)
#
# Usage (detached):
#   nohup bash scripts/per_sign_bench/run_eval_8gpu.sh > /tmp/eval_8gpu.log 2>&1 &
#   disown
#   tail -f $OUT/logs/*.log
set -uo pipefail

# ---- config (EDIT for the server) -------------------------------------------
REPO=/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench
MANIFEST=$REPO/benchmark_output/new/sumo_manifest.jsonl
SCENES=/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/scenes_eval
OUT=$REPO/benchmark_output/new/eval8

NGPU=8                       # number of physical GPUs
GPUS="0 1 2 3 4 5 6 7"       # GPU ids (space-separated)
# Parallel worker processes. The bottleneck is CPU (SUMO + MetaDrive sim step),
# NOT the GPU — nvidia-smi looks near-idle even for NN policies. So run MORE
# workers than GPUs: set CONCURRENCY ~= nproc (minus a few), watch RAM. Workers
# are spread over the GPUs round-robin, so several workers may share one card.
CONCURRENCY=16               # parallel worker processes (>= NGPU; CPU-bound)
NSHARDS=$CONCURRENCY         # one scene shard per worker

# Per-stream interpreter (eval_pipeline spawns run_benchmark with THIS python).
PY_CARL=/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/carl/bin/python
PY_PLANT=/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/plant2/bin/python
PY_REST=/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/plant2/bin/python

# Policies per stream. EMPTY string disables that stream.
CARL_POLICIES="carl,carl_rule"
PLANT_POLICIES="plant2,plant2_rule"
REST_POLICIES="idm,comprehensive_rule_expert,rule_compliant,ppo_lidar"

# Checkpoints (policy:path,policy:path) for the NN streams.
CKPTS_CARL="carl:/home/jovyan/shares/SR006.nfs2/smirnova/sdc/CaRL/nuPlan/checkpoints/nuplan_51479_1B/model_best.pth,carl_rule:/home/jovyan/shares/SR006.nfs2/smirnova/sdc/CaRL/nuPlan/checkpoints/nuplan_51479_1B/model_best.pth"
CKPTS_PLANT="plant2:/home/jovyan/shares/SR006.nfs2/smirnova/sdc/pdd-bench/checkpoints/epoch%3D029_final_3.ckpt,plant2_rule:/home/jovyan/shares/SR006.nfs2/smirnova/sdc/pdd-bench/checkpoints/epoch%3D029_final_3.ckpt"

EGO_VARIANTS="default"       # idm-family ego variants; "default,s1,s2,s3,s4" for all 5
# -----------------------------------------------------------------------------

export PER_SIGN_COMPLIANT_NPC=1
# Pin BLAS/OpenMP to 1 thread per process: with many workers, numpy/torch would
# otherwise each spawn ~nproc threads -> massive oversubscription & CPU thrash.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export SDL_AUDIODRIVER=dummy                          # silence ALSA/audio spam (headless)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null; chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

cd "$REPO"
SHARD_DIR="$OUT/shards"
mkdir -p "$SHARD_DIR" "$OUT/logs" "$OUT/parts"

# ---- 1. split manifest round-robin into NSHARDS balanced shards -------------
rm -f "$SHARD_DIR"/shard_*.jsonl
awk -v n="$NSHARDS" -v dir="$SHARD_DIR" 'NF{ f=sprintf("%s/shard_%02d.jsonl",dir,(NR-1)%n); print >> f }' "$MANIFEST"
echo "shards:"; for s in "$SHARD_DIR"/shard_*.jsonl; do echo "  $s: $(wc -l < "$s") rows"; done

# ---- 2. build job list: "py|policies|ckpts|shardfile|tag" -------------------
GPU_ARR=($GPUS)
jobs=()
add_jobs () {  # name py policies ckpts
  local name="$1" py="$2" pol="$3" ck="$4" s sf tag
  [ -z "$pol" ] && { echo "stream $name: disabled (no policies)"; return; }
  for s in $(seq 0 $((NSHARDS-1))); do
    sf=$(printf "%s/shard_%02d.jsonl" "$SHARD_DIR" "$s")
    tag=$(printf "%s_s%02d" "$name" "$s")
    jobs+=("$py|$pol|$ck|$sf|$tag")
  done
}
add_jobs carl   "$PY_CARL"  "$CARL_POLICIES"  "$CKPTS_CARL"
add_jobs plant2 "$PY_PLANT" "$PLANT_POLICIES" "$CKPTS_PLANT"
add_jobs rest   "$PY_REST"  "$REST_POLICIES"  ""
echo "total jobs: ${#jobs[@]}  across $NGPU GPUs"

# ---- 3. run one job (one shard for one stream) ------------------------------
run_job () {  # gpu spec
  local gpu="$1" spec="$2" py pol ck sf tag ckarg odir rc
  IFS='|' read -r py pol ck sf tag <<< "$spec"
  ckarg=(); [ -n "$ck" ] && ckarg=(--model-paths "$ck")
  odir="$OUT/parts/eval_$tag"
  echo "[$(date +%H:%M:%S)] gpu=$gpu START $tag ($pol, $(wc -l < "$sf") scenes)"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$py" scripts/per_sign_bench/eval_pipeline.py \
      --policies "$pol" "${ckarg[@]}" \
      --ego-variants "$EGO_VARIANTS" \
      --manifest "$sf" \
      --scenes-root "$SCENES" \
      --backends sumo \
      --out-dir "$odir" >> "$OUT/logs/$tag.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] gpu=$gpu DONE  $tag rc=$rc"
}

# ---- 4. CONCURRENCY workers, each pinned to a GPU (round-robin), jobs serial --
# Workers > GPUs is fine: load is CPU-bound, cards have spare capacity, so worker
# `li` runs on GPU `li % NGPU` (several workers can share one card).
lane () {  # worker_index
  local li="$1" gpu="${GPU_ARR[$(( li % NGPU ))]}" i
  for i in "${!jobs[@]}"; do
    [ $(( i % CONCURRENCY )) -eq "$li" ] && run_job "$gpu" "${jobs[$i]}"
  done
}
echo "=== launching $CONCURRENCY workers on $NGPU GPUs ==="
for li in $(seq 0 $((CONCURRENCY-1))); do lane "$li" & done
wait
echo "=== all workers done ==="

# ---- 5. merge per-shard episode CSVs into one combined CSV ------------------
COMB="$OUT/metrics_per_episode.csv"; first=1
for csv in "$OUT"/parts/eval_*/metrics_per_episode.csv; do
  [ -f "$csv" ] || continue
  if [ $first -eq 1 ]; then cp "$csv" "$COMB"; first=0; else tail -n +2 "$csv" >> "$COMB"; fi
done
[ -f "$COMB" ] && echo "combined CSV: $COMB ($(($(wc -l < "$COMB")-1)) episodes)"

# ---- 6. rebuild aggregations + cumulative markdown report on the combined ---
if [ -f "$COMB" ]; then
  "$PY_REST" scripts/per_sign_bench/aggregate_episode_metrics.py --csv "$COMB" --out-dir "$OUT"
  "$PY_REST" scripts/per_sign_bench/generate_cumulative_markdown_report.py \
      --run-root "$OUT" --cumulative "$OUT/reports/cumulative.json"
  echo "REPORT: $OUT/reports/report_cumulative.md"
fi
