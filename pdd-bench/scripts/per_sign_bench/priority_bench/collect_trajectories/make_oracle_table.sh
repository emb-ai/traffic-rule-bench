#!/usr/bin/env bash
# Build oracle_metrics_summary_top2.md for a priority collection OUT_BASE.
#
# Usage:
#   SIGN=yield ./make_oracle_table.sh output/trajectories_yield_<ts>
#   SIGN=main ./make_oracle_table.sh output/trajectories_main_<ts>
#   ./make_oracle_table.sh output/trajectories_<ts> output/trajectories_<ts>/experts
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORACLE_DIR="$(cd -- "$SCRIPT_DIR/../oracle" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_BASE="${1:?usage: $0 <OUT_BASE> [experts_dir]}"
# Resolve relative OUT_BASE against the caller's cwd *before* we cd elsewhere.
OUT_BASE="$(cd -- "$OUT_BASE" && pwd)"
EXPERTS_DIR="${2:-$OUT_BASE/experts}"
if [ -d "$EXPERTS_DIR" ]; then
    EXPERTS_DIR="$(cd -- "$EXPERTS_DIR" && pwd)"
fi
HORIZON="${HORIZON:-1500}"

# Infer PDD code from layout if SIGN unset.
: "${SIGN:=}"
: "${PDD_CODE:=}"
if [ -z "$PDD_CODE" ]; then
    case "${SIGN}" in
        yield|2.4|2_4) PDD_CODE=2.4 ;;
        main|main_road|2.1|2_1) PDD_CODE=2.1 ;;
        stop|stop_sign|2.5|2_5) PDD_CODE=2.5 ;;
        secondary|secondary_road|2.3|2_3|2.3.1|2.3.2|2.3.3) PDD_CODE=2.3 ;;
        *)
            if [ -d "$OUT_BASE/_manifests/2_1" ] || compgen -G "$OUT_BASE"'/*/2_1/all_runs.jsonl' > /dev/null; then
                PDD_CODE=2.1
            elif [ -d "$OUT_BASE/_manifests/2_5" ] || compgen -G "$OUT_BASE"'/*/2_5/all_runs.jsonl' > /dev/null; then
                PDD_CODE=2.5
            elif [ -d "$OUT_BASE/_manifests/2_3" ] || compgen -G "$OUT_BASE"'/*/2_3/all_runs.jsonl' > /dev/null; then
                PDD_CODE=2.3
            else
                PDD_CODE=2.4
            fi
            ;;
    esac
fi

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

"$PYTHON_BIN" "$ORACLE_DIR/make_oracle_metrics_table.py" \
    --jsonl "$ALL_RUNS" \
    --picks "$PICKS" \
    --signs "$PDD_CODE" \
    --horizon "$HORIZON" \
    --beta 0.25 \
    --output-dir "$TABLE_OUT"

cp -f "$TABLE_OUT/oracle_metrics_summary.md" \
      "$TABLE_OUT/oracle_metrics_summary_top2.md"
echo "Wrote $TABLE_OUT/oracle_metrics_summary_top2.md (signs=$PDD_CODE)"
