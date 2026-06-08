#!/usr/bin/env bash
# Evaluate ALL policies on the materialized NEW manifest, split across 2 GPUs.
#   GPU 0 : carl,carl_rule        (CaRL env)            + "rest" no-ckpt stream
#   GPU 1 : plant2,plant2_rule    (PlanT2 env)
# "rest" = idm, comprehensive_rule_expert, rule_compliant, ppo_lidar (no checkpoint;
# idm-family is CPU-bound, shares GPU 0 with carl). Each stream -> its own out-dir,
# its own log, builds metrics_per_episode.csv + reports/ automatically (eval_pipeline).
#
# Usage on the server (detached):
#   nohup bash scripts/per_sign_bench/run_eval_2gpu.sh > /tmp/eval_2gpu.log 2>&1 &
#   tail -f $OUT/eval_*.log
# Edit the config block for your server first.
set -uo pipefail

# ---- config (EDIT for the server) -------------------------------------------
REPO=/home/jovyan/.../sdc/pdd-bench                       # repo root (has scripts/per_sign_bench)
MANIFEST=$REPO/benchmark_output/new/sumo_manifest.jsonl   # materialized NEW manifest
SCENES=/home/jovyan/.../sdc/pdd-bench/scenes_new          # net.xml root (relative to which net_path resolves!)
OUT=$REPO/benchmark_output/new/eval                       # results root

CARL_GPU=0
PLANT_GPU=1
REST_GPU=0                                                # no-ckpt policies; share GPU 0

# Per-stream python — eval_pipeline spawns run_benchmark with THIS interpreter, so
# each stream uses its env. carl/plant2 usually need DIFFERENT envs.
PY_CARL=/home/jovyan/.mlspace/envs/carl/bin/python3
PY_PLANT=/home/jovyan/.mlspace/envs/plant2/bin/python3
PY_REST=/home/jovyan/.mlspace/envs/plant2/bin/python3     # any env with metadrive+sumo

# Checkpoints (policy:path,policy:path). Leave a stream's ckpts empty to skip it.
CKPTS_CARL="carl:/home/jovyan/.../CaRL/.../model_best.pth,carl_rule:/home/jovyan/.../CaRL/.../model_best.pth"
CKPTS_PLANT="plant2:/home/jovyan/.../checkpoints/epoch%3D029_final_3.ckpt,plant2_rule:/home/jovyan/.../checkpoints/epoch%3D029_final_3.ckpt"

REST_POLICIES="idm,comprehensive_rule_expert,rule_compliant,ppo_lidar"
EGO_VARIANTS="default"     # idm-family ego variants; "default,s1,s2,s3,s4" for all 5
# -----------------------------------------------------------------------------

cd "$REPO/.." 2>/dev/null || cd "$REPO"     # run from repo (scripts/per_sign_bench resolvable)
mkdir -p "$OUT"

# Run one GPU stream: a single eval_pipeline over the manifest for `policies`.
run_stream () {
  local gpu="$1" py="$2" policies="$3" ckpts="$4" tag="$5"
  local ckpt_arg=()
  [ -n "$ckpts" ] && ckpt_arg=(--model-paths "$ckpts")
  echo "[$(date +%H:%M:%S)] [$tag gpu=$gpu] START  ($policies)"
  CUDA_VISIBLE_DEVICES="$gpu" PER_SIGN_COMPLIANT_NPC=1 \
    "$py" scripts/per_sign_bench/eval_pipeline.py \
      --policies "$policies" \
      "${ckpt_arg[@]}" \
      --ego-variants "$EGO_VARIANTS" \
      --manifest "$MANIFEST" \
      --scenes-root "$SCENES" \
      --backends sumo \
      --out-dir "$OUT/eval_$tag"
  echo "[$(date +%H:%M:%S)] [$tag gpu=$gpu] DONE rc=$?"
}

run_stream "$CARL_GPU"  "$PY_CARL"  "carl,carl_rule"     "$CKPTS_CARL"  carl   > "$OUT/eval_carl.log"   2>&1 &
PID_CARL=$!
run_stream "$PLANT_GPU" "$PY_PLANT" "plant2,plant2_rule" "$CKPTS_PLANT" plant2 > "$OUT/eval_plant2.log" 2>&1 &
PID_PLANT=$!
run_stream "$REST_GPU"  "$PY_REST"  "$REST_POLICIES"     ""             rest   > "$OUT/eval_rest.log"   2>&1 &
PID_REST=$!

echo "carl   PID $PID_CARL  (GPU $CARL_GPU)  -> $OUT/eval_carl.log"
echo "plant2 PID $PID_PLANT (GPU $PLANT_GPU) -> $OUT/eval_plant2.log"
echo "rest   PID $PID_REST  (GPU $REST_GPU)  -> $OUT/eval_rest.log"

wait "$PID_CARL";  rc_carl=$?
wait "$PID_PLANT"; rc_plant=$?
wait "$PID_REST";  rc_rest=$?
echo "ALL DONE  carl=$rc_carl plant2=$rc_plant rest=$rc_rest"

# ---- optional: merge per-stream episode CSVs into one combined report --------
COMB="$OUT/metrics_per_episode.csv"
first=1
for tag in carl plant2 rest; do
  csv="$OUT/eval_$tag/metrics_per_episode.csv"
  [ -f "$csv" ] || continue
  if [ $first -eq 1 ]; then cp "$csv" "$COMB"; first=0
  else tail -n +2 "$csv" >> "$COMB"; fi
done
[ -f "$COMB" ] && echo "combined metrics: $COMB  ($(($(wc -l < "$COMB")-1)) episodes)"
