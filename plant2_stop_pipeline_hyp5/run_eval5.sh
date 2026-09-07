#!/usr/bin/env bash
# Eval all 5 hypothesis checkpoints (last_ft, epoch 9) with the exact harness
# used for the 0.738/0.738 reference numbers.
#
# Usage: bash run_eval5.sh
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
PY="/home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python"
PRIORITY="$TRB_ROOT/pdd-bench/scripts/per_sign_bench/priority_bench"
CKPT_ROOT="$TRB_ROOT/plant2/PlanT/checkpoints_ft"
STOP_DATA="$TRB_ROOT/stop_data"
TEST_MANIFEST="$STOP_DATA/output/ts_test/real_manifest.jsonl"
SCENES="$STOP_DATA/scenes"
WORK="$TRB_ROOT/plant2_stop_pipeline_hyp5"

names=(h1_persist h2_reweight h3_memory h4_auxloss h5_pool)
gpus=(3 4 5 6 7)

for i in "${!names[@]}"; do
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  addon="stop_hyp_${name}"
  ckpt_dir="$CKPT_ROOT/$addon"
  ckpt=$(ls -1t "$ckpt_dir"/epoch=004*.ckpt 2>/dev/null | head -1)
  if [[ -z "$ckpt" ]]; then
    echo "[skip] $name: no epoch=004 ckpt in $ckpt_dir yet"
    continue
  fi
  out="$WORK/eval/${name}_ep4"
  mkdir -p "$out"
  echo "[eval] $name ckpt=$ckpt gpu=$gpu -> $out"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$PRIORITY/eval_pipeline.py" \
      --policies plant2 \
      --model-paths "plant2:$ckpt" \
      --manifest "$TEST_MANIFEST" \
      --scenes-root "$SCENES" \
      --out-dir "$out" \
      --plant2-action-mode pid \
      --jobs 8 \
      --backends sumo \
      > "$WORK/logs/eval_${name}.log" 2>&1
  ) &
done
wait
echo "ALL_EVALS_DONE"
for name in "${names[@]}"; do
  report="$WORK/eval/${name}_ep4/reports/report_cumulative.md"
  echo "=== $name ==="
  [[ -f "$report" ]] && grep -A2 -m1 "Sign compliance SR" "$report" || echo "  (no report at $report)"
done
