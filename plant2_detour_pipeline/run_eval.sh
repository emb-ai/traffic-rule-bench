#!/usr/bin/env bash
# Detour (4.2.1/4.2.2/4.2.3) eval for one checkpoint:
#   1. Full FV-fast metrics on the 122-scene detour test catalog (no gifs).
#   2. 5 sample GIFs (a small illustrative subset, not the full 122) into
#      this run's own gifs/ dir, per sign so all three are represented.
#
# Usage: bash run_eval.sh <tag> <ckpt_path> [gpu]
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
SM="/home/jovyan/shares/SR006.nfs2/smirnova"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
BENCH="$TRB_ROOT/pdd-bench/scripts/per_sign_bench"
PY="/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python"
WORK="$TRB_ROOT/plant2_detour_pipeline"

# collect_boxes (bench/plant2_frames.py) defaults its sign allowlist to "2.5"
# only, for backward compat with existing stop-sign checkpoints (flipping the
# global default to "all" was tried and confirmed to break them). Detour eval
# needs its own three PDD codes visible in x_objs instead.
export PLANT2_DUMP_SIGN_CLASSES="4.2.1,4.2.2,4.2.3"

TAG="${1:?usage: run_eval.sh <tag> <ckpt_path> [gpu]}"
CKPT="${2:?usage: run_eval.sh <tag> <ckpt_path> [gpu]}"
GPU="${3:-7}"

MANIFEST="$SM/traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl"
SCENES="$SM/sdc/pdd-bench/scenes"
OUT="$WORK/eval/$TAG"
GIF_MANIFEST="$OUT/gif_sample_manifest.jsonl"

[[ -f "$CKPT" ]] || { echo "ERROR: missing ckpt: $CKPT"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "ERROR: missing manifest: $MANIFEST"; exit 1; }
[[ -d "$SCENES" ]] || { echo "ERROR: missing scenes: $SCENES"; exit 1; }

mkdir -p "$OUT"
echo "=== [1/2] full FV-fast metrics: $TAG (122 scenes, no gifs) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$PIPELINE/eval/eval_full.py" fv \
  --ckpt "$CKPT" \
  --out "$OUT/fv_fast" \
  --manifest "$MANIFEST" \
  --scenes "$SCENES" \
  --gpus "$GPU" \
  --nshards 4 \
  --concurrency 4 \
  --exclude-codes "" \
  2>&1 | tee "$OUT/eval_metrics.log"

echo "=== [2/2] 5 sample GIFs (one 4.2.1, one 4.2.2, one 4.2.3 w/ cones + one 4.2.1, one 4.2.3 w/o cones — false-positive check) ==="
"$PY" - "$MANIFEST" "$GIF_MANIFEST" <<'PYEOF'
import json, sys
manifest_path, out_path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(manifest_path) if l.strip()]
by_key = {}
for r in rows:
    by_key.setdefault((r["sign_code"], r.get("detour_cones")), r)
picks = []
for code in ("4.2.1", "4.2.2", "4.2.3"):
    if (code, True) in by_key:
        picks.append(by_key[(code, True)])
# fill remaining slots (up to 5) with detour_cones=False rows for a false-positive spot check
for code in ("4.2.1", "4.2.3"):
    if len(picks) >= 5:
        break
    if (code, False) in by_key:
        picks.append(by_key[(code, False)])
picks = picks[:5]
with open(out_path, "w") as f:
    for r in picks:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(picks)} rows -> {out_path}")
for r in picks:
    print(" ", r["sign_code"], r["scene_id"], "cones=", r.get("detour_cones"))
PYEOF

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$BENCH/run_benchmark.py" \
  --policy plant2 \
  --run-name "${TAG}_gifs" \
  --ego-variant default \
  --manifest "$GIF_MANIFEST" \
  --scenes-root "$SCENES" \
  --backends sumo \
  --max-steps 1500 \
  --benchmark-output "$OUT/gif_bench" \
  --model-path "$CKPT" \
  --plant2-action-mode pid \
  --save-gifs \
  --gif-dir "$OUT/gifs" \
  2>&1 | tee "$OUT/gif_run.log"

echo "DONE tag=$TAG out=$OUT"
echo "  metrics: $OUT/fv_fast/reports/report_cumulative.md"
echo "  gifs:    $OUT/gifs/"
