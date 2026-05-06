#!/usr/bin/env bash
# Full metrics pipeline: consolidate replays + per-var reports + combined report.
#
# Pipeline:
#   1. consolidate_replays.py            (one consolidated <baseline>_replays.jsonl per var)
#   2. per-var: build_episode_metrics_csv → aggregate → MD reports
#   3. combined: same chain but over ALL vars
#
# Output layout:
#   $ROOT/metrics_per_episode.csv               ← combined
#   $ROOT/aggregations/                         ← combined
#   $ROOT/reports/                              ← combined
#   $ROOT/per_var/var_<i>/metrics_per_episode.csv
#   $ROOT/per_var/var_<i>/aggregations/
#   $ROOT/per_var/var_<i>/reports/
#
# Env knobs:
#   ROOT=...                  benchmark dir (required)
#   VARS="0 1 2 3 4 5 6 7 8 9"   space-separated var indices
#   SKIP_CONSOLIDATE=1        skip step 1 (use existing <baseline>_replays.jsonl)
#   SKIP_PER_VAR=1            skip step 2 (only combined report)
#   SKIP_COMBINED=1           skip step 3 (only per-var reports)
#   DEDUP=1                   pass --dedup to consolidate_replays.py

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${SCRIPTS:-$SCRIPT_DIR}"

: "${ROOT:?ROOT must be set (benchmark output dir)}"
: "${VARS:=0 1 2 3 4 5 6 7 8 9}"
: "${SKIP_CONSOLIDATE:=0}"
: "${SKIP_PER_VAR:=0}"
: "${SKIP_COMBINED:=0}"
: "${CONSOLIDATE_PARALLEL:=1}"
: "${DEDUP:=0}"

LOG_DIR="$ROOT/logs/metrics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "run_full_metrics_pipeline.sh"
echo "  ROOT       = $ROOT"
echo "  SCRIPTS    = $SCRIPTS"
echo "  VARS       = $VARS"
echo "  LOG_DIR    = $LOG_DIR"
echo "================================================================"

if [ ! -d "$ROOT/runs" ]; then
    echo "ERROR: $ROOT/runs does not exist" >&2
    exit 2
fi

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

# === Step 1: consolidate per var ============================================
if [ "$SKIP_CONSOLIDATE" != "1" ]; then
    echo
    echo "=== Step 1: consolidate replays for vars: $VARS ==="
    DEDUP_FLAG=()
    [ "$DEDUP" = "1" ] && DEDUP_FLAG=(--dedup)

    if [ "$CONSOLIDATE_PARALLEL" = "1" ]; then
        for i in $VARS; do
            (
                run_step "consolidate_var_$i" \
                    python3 "$SCRIPTS/consolidate_replays.py" \
                        --runs-var "$ROOT/runs/var_$i" "${DEDUP_FLAG[@]}"
            ) &
        done
        wait
    else
        for i in $VARS; do
            run_step "consolidate_var_$i" \
                python3 "$SCRIPTS/consolidate_replays.py" \
                    --runs-var "$ROOT/runs/var_$i" "${DEDUP_FLAG[@]}" \
                || echo "  [continue] var_$i failed but moving on"
        done
    fi
else
    echo
    echo "=== Step 1: SKIPPED (SKIP_CONSOLIDATE=1) ==="
fi

# === Step 2: per-var pipeline ===============================================
per_var_pipeline() {
    local i="$1"
    local var_dir="$ROOT/per_var/var_$i"
    mkdir -p "$var_dir"

    run_step "var_${i}_build_csv" \
        python3 "$SCRIPTS/build_episode_metrics_csv.py" \
            --runs-root "$ROOT/runs" \
            --out "$var_dir/metrics_per_episode.csv" \
            --vars "$i" \
            --manifests-root "$ROOT/chunks" || return 1

    run_step "var_${i}_aggregate" \
        python3 "$SCRIPTS/aggregate_episode_metrics.py" \
            --csv "$var_dir/metrics_per_episode.csv" \
            --out-dir "$var_dir" || return 1

    run_step "var_${i}_md_cumulative" \
        python3 "$SCRIPTS/generate_cumulative_markdown_report.py" \
            --run-root "$var_dir" \
            --cumulative "$var_dir/reports/cumulative.json" || true

    run_step "var_${i}_md_categories" \
        python3 "$SCRIPTS/generate_category_aggregation_report.py" \
            --run-root "$var_dir" \
            --cumulative "$var_dir/reports/cumulative.json" \
            --preset task_oriented || true

    run_step "var_${i}_md_2node" \
        python3 "$SCRIPTS/merge_and_report_2node.py" \
            --node1 "$var_dir/reports/cumulative_2node.json" \
            --node2 "$var_dir/reports/cumulative_2node.json" \
            --out-dir "$var_dir/reports" || true
}

if [ "$SKIP_PER_VAR" != "1" ]; then
    echo
    echo "=== Step 2: per-var reports ==="
    for i in $VARS; do
        echo "--- var_$i ---"
        per_var_pipeline "$i" || echo "  [continue] var_$i partially failed"
    done
else
    echo
    echo "=== Step 2: SKIPPED (SKIP_PER_VAR=1) ==="
fi

# === Step 3: combined report =================================================
if [ "$SKIP_COMBINED" != "1" ]; then
    echo
    echo "=== Step 3: combined report (all vars) ==="

    run_step "all_build_csv" \
        python3 "$SCRIPTS/build_episode_metrics_csv.py" \
            --runs-root "$ROOT/runs" \
            --out "$ROOT/metrics_per_episode.csv" \
            --vars "$(echo $VARS | tr ' ' ',')" \
            --manifests-root "$ROOT/chunks"

    run_step "all_aggregate" \
        python3 "$SCRIPTS/aggregate_episode_metrics.py" \
            --csv "$ROOT/metrics_per_episode.csv" \
            --out-dir "$ROOT"

    run_step "all_md_cumulative" \
        python3 "$SCRIPTS/generate_cumulative_markdown_report.py" \
            --run-root "$ROOT" \
            --cumulative "$ROOT/reports/cumulative.json" || true

    run_step "all_md_categories" \
        python3 "$SCRIPTS/generate_category_aggregation_report.py" \
            --run-root "$ROOT" \
            --cumulative "$ROOT/reports/cumulative.json" \
            --preset task_oriented || true

    run_step "all_md_2node" \
        python3 "$SCRIPTS/merge_and_report_2node.py" \
            --node1 "$ROOT/reports/cumulative_2node.json" \
            --node2 "$ROOT/reports/cumulative_2node.json" \
            --out-dir "$ROOT/reports" || true
else
    echo
    echo "=== Step 3: SKIPPED (SKIP_COMBINED=1) ==="
fi

echo
echo "================================================================"
echo "DONE.  Logs in: $LOG_DIR"
echo
echo "Combined report:"
echo "  $ROOT/metrics_per_episode.csv"
echo "  $ROOT/aggregations/"
echo "  $ROOT/reports/report_cumulative.md"
echo "  $ROOT/reports/report_cumulative_categories.md"
echo "================================================================"
