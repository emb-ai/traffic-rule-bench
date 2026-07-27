#!/usr/bin/env bash
# Build oracle_metrics_summary_top2.md for a one_way (5.7.1+5.7.2) collection OUT_BASE.
#
# Usage:
#   ./make_oracle_table.sh output/trajectories_<ts>
#   ./make_oracle_table.sh output/trajectories_<ts> output/trajectories_<ts>/experts
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PER_SIGN="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_BASE="${1:?usage: $0 <OUT_BASE> [experts_dir]}"
# Resolve before cd — relative OUT_BASE is relative to the caller's cwd.
if [[ "$OUT_BASE" != /* ]]; then
    OUT_BASE="$(pwd)/$OUT_BASE"
fi
OUT_BASE="$(cd -- "$OUT_BASE" && pwd)"

EXPERTS_DIR="${2:-$OUT_BASE/experts}"
if [[ "$EXPERTS_DIR" != /* ]]; then
    EXPERTS_DIR="$(pwd)/$EXPERTS_DIR"
fi
HORIZON="${HORIZON:-1500}"

ALL_RUNS="$OUT_BASE/_merged/all_runs.jsonl"
if [ ! -s "$ALL_RUNS" ]; then
    ALL_RUNS="$EXPERTS_DIR/all_runs_dedup.jsonl"
fi
PICKS="$EXPERTS_DIR/experts_scene_uid_top2.jsonl"

if [ ! -s "$ALL_RUNS" ]; then
    echo "ERROR: missing all_runs at $ALL_RUNS"
    exit 1
fi
if [ ! -s "$PICKS" ]; then
    echo "ERROR: missing picks $PICKS — run select_experts_coverage.py first"
    exit 1
fi

TABLE_OUT="$OUT_BASE/oracle_metrics"
mkdir -p "$TABLE_OUT"

cd "$PER_SIGN"
"$PYTHON_BIN" make_oracle_metrics_table.py \
    --jsonl "$ALL_RUNS" \
    --picks "$PICKS" \
    --signs 5.7.1 5.7.2 \
    --horizon "$HORIZON" \
    --beta 0.25 \
    --output-dir "$TABLE_OUT"

cp -f "$TABLE_OUT/oracle_metrics_summary.md" \
      "$TABLE_OUT/oracle_metrics_summary_top2.md"
echo "Wrote $TABLE_OUT/oracle_metrics_summary_top2.md"
