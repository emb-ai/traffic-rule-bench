#!/usr/bin/env bash
# Eval the corrected 30-epoch hypothesis checkpoints (best-by-val-loss,
# matching the exact methodology that produced the 0.738 reference).
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
PY="/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python"
PRIORITY="$TRB_ROOT/pdd-bench/scripts/per_sign_bench/priority_bench"
CKPT_ROOT="$TRB_ROOT/plant2/PlanT/checkpoints_ft"
STOP_DATA="$TRB_ROOT/stop_data"
TEST_MANIFEST="$STOP_DATA/output/ts_test/real_manifest.jsonl"
SCENES="$STOP_DATA/scenes"
WORK="$TRB_ROOT/plant2_stop_pipeline_hyp5"

names=(h1_persist h2_reweight h3_memory h4_auxloss h5_pool)
gpus=(0 1 3 4 5)
kinds=(best)

for i in "${!names[@]}"; do
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  addon="stop_hyp_${name}_ep30"
  ckpt_dir="$CKPT_ROOT/$addon"
  for kind in "${kinds[@]}"; do
    if [[ "$kind" == "best" ]]; then
      ckpt=$(ls -1t "$ckpt_dir"/best_*.ckpt 2>/dev/null | head -1)
    else
      ckpt=$(ls -1t "$ckpt_dir"/last_ft_*.ckpt 2>/dev/null | head -1)
    fi
    if [[ -z "$ckpt" ]]; then
      echo "[skip] $name $kind: no ckpt in $ckpt_dir yet"
      continue
    fi
    out="$WORK/eval/${name}_ep30_${kind}"
    mkdir -p "$out"
    echo "[eval] $name $kind ckpt=$ckpt gpu=$gpu -> $out"
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
        > "$WORK/logs/eval_${name}_ep30_${kind}.log" 2>&1
    ) &
  done
done
wait
echo "ALL_EVALS_DONE"
for name in "${names[@]}"; do
  for kind in "${kinds[@]}"; do
    report="$WORK/eval/${name}_ep30_${kind}/reports/report_cumulative.md"
    echo "=== $name $kind ==="
    [[ -f "$report" ]] && grep -A2 -m1 "Sign compliance SR" "$report" || echo "  (no report at $report)"
  done
done
