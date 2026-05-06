#!/usr/bin/env bash
# Compute metrics for a single run produced by run_benchmark_mini.py.
#
# Usage:
#   bash run_metrics_single_run.sh \
#       --run-dir eval_out/runs/var_0/idm_default \
#       --out-dir eval_out/metrics_idm_default \
#       --policy  idm_default
#
# Inputs:
#   --run-dir DIR    Run output dir (must contain replays/ from --emit-replay-sidecar).
#   --out-dir DIR    Metrics output (default: <run-dir>/metrics).
#   --policy NAME    Policy/baseline name. Default: auto-detected from episodes_*.jsonl.
#   --preset NAME    Category preset (default: task_oriented).
#
# Outputs:
#   <out>/metrics_per_episode.csv
#   <out>/aggregations/*.csv
#   <out>/reports/report_cumulative.md
#   <out>/reports/report_cumulative_categories.md

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$SCRIPT_DIR"

RUN_DIR=""
OUT_DIR=""
POLICY=""
PRESET="task_oriented"

usage() {
    sed -n '2,21p' "$0" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --run-dir)  RUN_DIR="$2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --policy)   POLICY="$2"; shift 2 ;;
        --preset)   PRESET="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

[ -n "$RUN_DIR" ] || { echo "ERROR: --run-dir is required" >&2; usage; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)" || { echo "ERROR: bad --run-dir" >&2; exit 2; }

REPLAYS_SRC="$RUN_DIR/replays"
if [ ! -d "$REPLAYS_SRC" ]; then
    echo "ERROR: $REPLAYS_SRC does not exist." >&2
    echo "       Re-run the benchmark with --emit-replay-sidecar." >&2
    exit 2
fi

if [ -z "$POLICY" ]; then
    EP=$(ls "$RUN_DIR"/episodes_*.jsonl 2>/dev/null | head -1 || true)
    if [ -n "$EP" ]; then
        POLICY=$(basename "$EP" .jsonl)
        POLICY="${POLICY#episodes_}"
    fi
fi
[ -n "$POLICY" ] || { echo "ERROR: cannot infer --policy (no episodes_*.jsonl in $RUN_DIR)" >&2; exit 2; }

: "${OUT_DIR:=$RUN_DIR/metrics}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

VAR_DIR="$OUT_DIR/runs/var_0/$POLICY"
mkdir -p "$VAR_DIR"
ln -sfn "$REPLAYS_SRC" "$VAR_DIR/replays"

NO_MANIFESTS="$OUT_DIR/_no_manifests"

LOG_DIR="$OUT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "run_metrics_single_run.sh"
echo "  RUN_DIR  = $RUN_DIR"
echo "  OUT_DIR  = $OUT_DIR"
echo "  POLICY   = $POLICY"
echo "  PRESET   = $PRESET"
echo "  LOG_DIR  = $LOG_DIR"
echo "================================================================"

run_step() {
    local label="$1"; shift
    local log="$LOG_DIR/$label.log"
    echo "  [run] $label"
    if ! "$@" > "$log" 2>&1; then
        echo "  [FAIL] $label — see $log" >&2
        tail -20 "$log" >&2
        return 1
    fi
}

run_step "build_csv" \
    python3 "$SCRIPTS/build_episode_metrics_csv.py" \
        --runs-root "$OUT_DIR/runs" \
        --out "$OUT_DIR/metrics_per_episode.csv" \
        --vars 0 \
        --manifests-root "$NO_MANIFESTS" \
    || exit 1

run_step "aggregate" \
    python3 "$SCRIPTS/aggregate_episode_metrics.py" \
        --csv "$OUT_DIR/metrics_per_episode.csv" \
        --out-dir "$OUT_DIR" \
    || exit 1

run_step "md_cumulative" \
    python3 "$SCRIPTS/generate_cumulative_markdown_report.py" \
        --run-root "$OUT_DIR" \
        --cumulative "$OUT_DIR/reports/cumulative.json" \
    || true

run_step "md_categories" \
    python3 "$SCRIPTS/generate_category_aggregation_report.py" \
        --run-root "$OUT_DIR" \
        --cumulative "$OUT_DIR/reports/cumulative.json" \
        --preset "$PRESET" \
    || true

echo
echo "================================================================"
echo "DONE."
echo "  CSV:       $OUT_DIR/metrics_per_episode.csv"
echo "  Aggreg.:   $OUT_DIR/aggregations/"
echo "  Reports:   $OUT_DIR/reports/report_cumulative.md"
echo "             $OUT_DIR/reports/report_cumulative_categories.md"
echo "  Logs:      $LOG_DIR"
echo "================================================================"
