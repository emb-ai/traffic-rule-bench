#!/usr/bin/env bash
# Repack mini benchmark expert pkls -> Plant2 .pt, fine-tune PlanT, then run_benchmark_v2.
#
# Uses conda env "plant2" (see PLANT2_PY below). Run inside tmux, e.g.:
#   tmux new -s plant2_ft
#   bash /path/to/run_plant2_mini_repack_train_eval.sh
#
# Optional: limit repack for a smoke test
#   export REPACK_LIMIT=5
# Parallel repack (spawn; use 2–4 on machines with enough RAM)
#   export REPACK_NUM_WORKERS=4
#
set -euo pipefail

PLANT2_PY="${PLANT2_PY:-python}"
if ! command -v "$PLANT2_PY" >/dev/null 2>&1 && [[ ! -x "$PLANT2_PY" ]]; then
  echo "Set PLANT2_PY to the plant2 conda env python (default: python on PATH)" >&2
  exit 1
fi

# Default benchmark root: symlink under this repo (pdd-bench/data/benchmark_mini).
BENCH_MINI="${BENCH_MINI:-pdd-bench/data/benchmark_mini}"
# Maps old absolute paths in sidecars to this checkout's pdd-bench tree
REMAP_OLD="${REMAP_OLD:-/old/sdc}"

OUT_PT="${OUT_PT:-pdd-bench/outputs/plant2_repack_mini}"
INIT_CKPT="${INIT_CKPT:-plant2/models/epoch%3D029_final_3.ckpt}"
TRAIN_OUT="${TRAIN_OUT:-pdd-bench/outputs/plant2_supervised_from_mini}"
SCENES_DIR="${SCENES_DIR:-pdd-bench/scenes}"
BENCH_V2_OUT="${BENCH_V2_OUT:-pdd-bench/outputs/benchmark_v2_after_mini_ft}"

REPACK_SCRIPT="pdd-bench/scripts/agents/train/repack_benchmark_expert_pkl_to_plant2_pt.py"
TRAIN_SCRIPT="pdd-bench/scripts/agents/train/train_plant2_from_carl_trajectories.py"
EVAL_SCRIPT="pdd-bench/scripts/run_benchmark_v2.py"

LOG="${LOG:-pdd-bench/outputs/plant2_mini_pipeline.log}"
mkdir -p "$(dirname "$LOG")" "$OUT_PT" "$TRAIN_OUT" "$BENCH_V2_OUT"
exec > >(tee -a "$LOG") 2>&1

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

echo "=== $(date -Is) plant2 mini pipeline ==="
echo "PLANT2_PY=$PLANT2_PY"
echo "OUT_PT=$OUT_PT  TRAIN_OUT=$TRAIN_OUT"

REPACK_EXTRA=()
if [[ -n "${REPACK_LIMIT:-}" ]]; then
  REPACK_EXTRA+=(--limit "$REPACK_LIMIT")
  echo "REPACK_LIMIT=$REPACK_LIMIT (subset)"
fi
REPACK_NUM_WORKERS="${REPACK_NUM_WORKERS:-1}"
echo "REPACK_NUM_WORKERS=$REPACK_NUM_WORKERS"

echo "--- repack ---"
"$PLANT2_PY" "$REPACK_SCRIPT" \
  --benchmark-root "$BENCH_MINI" \
  --output-dir "$OUT_PT" \
  --sdc-root "." \
  --remap-net-path "$REMAP_OLD:." \
  --num-workers "$REPACK_NUM_WORKERS" \
  "${REPACK_EXTRA[@]}"

N_PT=$(find "$OUT_PT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)
if [[ "$N_PT" -lt 1 ]]; then
  echo "No .pt files in $OUT_PT — abort." >&2
  exit 1
fi

echo "--- train (fine-tune from $INIT_CKPT) ---"
# Avoid ${VAR:-1e-5}: some bash versions misparse the default and throw "unexpected token '('".
TRAIN_LR="${LR:-0.00001}"
"$PLANT2_PY" "$TRAIN_SCRIPT" \
  --data-dir "$OUT_PT" \
  --checkpoint_file "$INIT_CKPT" \
  --output_dir "$TRAIN_OUT" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --lr "$TRAIN_LR"

FINAL_PT="$TRAIN_OUT/plant2_supervised_2nd_final.pt"
if [[ ! -f "$FINAL_PT" ]]; then
  echo "Missing $FINAL_PT" >&2
  exit 1
fi

echo "--- run_benchmark_v2 ($FINAL_PT) ---"
"$PLANT2_PY" "$EVAL_SCRIPT" \
  --checkpoint "$FINAL_PT" \
  --scenes-dir "$SCENES_DIR" \
  --output-dir "$BENCH_V2_OUT" \
  --run-name "${RUN_NAME:-plant2_after_mini_ft}" \
  --max-scenes "${MAX_SCENES:-10}" \
  --max-steps "${MAX_STEPS:-800}" \
  --no-gifs

echo "=== $(date -Is) done ==="
echo "Log: $LOG"
echo "Fine-tuned weights: $FINAL_PT"
echo "Benchmark results: $BENCH_V2_OUT"
