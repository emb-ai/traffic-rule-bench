#!/usr/bin/env bash
# Build oracle_rule baseline + aggregations + markdown report for ready-sign
# test CSVs under per_sign_bench/<family>/benchmark_output/...
#
# Colleague layout (flat):
#   $SM/traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/eval_test20
#
# Ours (per-family):
#   per_sign_bench/<bench>/benchmark_output/.../metrics_per_episode.csv
#   → writes metrics_per_episode_oracle.csv + with_oracle/{aggregations,reports}
#
# Usage:
#   bash run_oracle_metrics.sh                  # all ready jobs that exist
#   bash run_oracle_metrics.sh --list
#   bash run_oracle_metrics.sh --only 4.3,5.19
#   bash run_oracle_metrics.sh --out-dir PATH   # single custom eval dir
#   DRY_RUN=1 bash run_oracle_metrics.sh
#
# Env:
#   RULE_EXPERTS=...   override comma-separated rule-expert baselines
#   PYTHON_BIN=python3

set -euo pipefail

PSB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

RULE_EXPERTS="${RULE_EXPERTS:-comprehensive_rule_expert_default,comprehensive_rule_expert_s1,comprehensive_rule_expert_s2,comprehensive_rule_expert_s3,comprehensive_rule_expert_s4,carl_rule_default,plant2_rule_default,rule_compliant_default}"

ORACLE_PY="$PSB/build_oracle_baseline.py"
AGG_PY="$PSB/aggregate_episode_metrics.py"
MD_PY="$PSB/generate_cumulative_markdown_report.py"

# label|relative_dir_from_PSB|csv_basename
# Paths match summarize_ready_sign_test_metrics.READY_JOBS.
DEFAULT_JOBS=(
  "2.1|main_sign/benchmark_output/test_metrics/test20_batch/2_1/eval_out|metrics_per_episode.csv"
  "2.3.1-2.3.3|secondary_sign/benchmark_output/test_metrics/test20_batch/2_3_1_2_3_3/eval_out|metrics_per_episode.csv"
  "2.4|yield_sign/benchmark_output/test_metrics/test20_batch/2_4/eval_out|metrics_per_episode.csv"
  "2.5|stop_sign/benchmark_output/test_metrics/test20_batch/2_5/eval_out|metrics_per_episode.csv"
  "3.1-3.2|no_entry_signs/benchmark_output/combined/eval_out_test|metrics_per_episode.csv"
  "3.1|no_entry_signs/benchmark_output/3_1/final_metrics_v1/eval_out_test|metrics_per_episode.csv"
  "3.2|no_entry_signs/benchmark_output/3_2/final_metrics_v1/eval_out_test|metrics_per_episode.csv"
  "3.24-5.31|speed_signs/benchmark_output/run_v61_a6/eval_fast|metrics_per_episode_test20.csv"
  "4.2.1-4.2.3|detour_sign/benchmark_output/4_2/eval_test20|metrics_per_episode.csv"
  "4.3|roundabout_sign/benchmark_output/test_metrics/test20_batch/4_3/eval_out|metrics_per_episode.csv"
  "5.7.1-5.7.2|one_way_signs/benchmark_output/combined/eval_out_test|metrics_per_episode.csv"
  "5.7.1|one_way_signs/benchmark_output/5_7_1/final_metrics_v1/eval_out_test|metrics_per_episode.csv"
  "5.7.2|one_way_signs/benchmark_output/5_7_2/final_metrics_v1/eval_out_test|metrics_per_episode.csv"
  "5.15.1-5.15.2|lane_direction_signs/benchmark_output/test_metrics/test20_batch/5_15_1_5_15_2/eval_out|metrics_per_episode.csv"
  "5.19|crosswalk_sign/benchmark_output/test_metrics/test20_batch/5_19/eval_out|metrics_per_episode.csv"
)

ONLY=""
CUSTOM_OUT=""
LIST_ONLY=0

usage() {
  sed -n '2,22p' "$0" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --out-dir) CUSTOM_OUT="$2"; shift 2 ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

should_run_label() {
  local label="$1"
  if [ -z "$ONLY" ]; then
    return 0
  fi
  local IFS=','
  local want
  for want in $ONLY; do
    want="$(echo "$want" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ "$want" = "$label" ] && return 0
  done
  return 1
}

run_one() {
  local label="$1"
  local out_dir="$2"
  local csv_name="$3"

  local csv="$out_dir/$csv_name"
  if [ ! -f "$csv" ]; then
    echo "[skip] $label — missing $csv"
    return 0
  fi

  # Oracle CSV next to source; for metrics_per_episode.csv → *_oracle.csv,
  # for metrics_per_episode_test20.csv → metrics_per_episode_test20_oracle.csv
  local stem="${csv_name%.csv}"
  local oracle_csv="$out_dir/${stem}_oracle.csv"
  local with_oracle="$out_dir/with_oracle"

  echo
  echo "================================================================"
  echo "[oracle] $label"
  echo "  csv     = $csv"
  echo "  out     = $oracle_csv"
  echo "  report  = $with_oracle"
  echo "================================================================"

  if [ "$DRY_RUN" = "1" ]; then
    echo "  [dry-run] skip"
    return 0
  fi

  "$PYTHON_BIN" "$ORACLE_PY" \
    --csv "$csv" \
    --out "$oracle_csv" \
    --rule-experts "$RULE_EXPERTS"

  "$PYTHON_BIN" "$AGG_PY" \
    --csv "$oracle_csv" \
    --out-dir "$with_oracle"

  "$PYTHON_BIN" "$MD_PY" \
    --run-root "$with_oracle" \
    --cumulative "$with_oracle/reports/cumulative.json"

  echo "[ok] $label → $with_oracle/reports/"
}

if [ -n "$CUSTOM_OUT" ]; then
  CUSTOM_OUT="$(cd "$CUSTOM_OUT" && pwd)"
  csv_name="metrics_per_episode.csv"
  if [ ! -f "$CUSTOM_OUT/$csv_name" ] && [ -f "$CUSTOM_OUT/metrics_per_episode_test20.csv" ]; then
    csv_name="metrics_per_episode_test20.csv"
  fi
  run_one "custom" "$CUSTOM_OUT" "$csv_name"
  exit 0
fi

if [ "$LIST_ONLY" = "1" ]; then
  printf '%-16s  %s\n' "LABEL" "CSV"
  for job in "${DEFAULT_JOBS[@]}"; do
    IFS='|' read -r label rel csv_name <<<"$job"
    csv="$PSB/$rel/$csv_name"
    status="missing"
    [ -f "$csv" ] && status="ok"
    printf '%-16s  [%s] %s\n' "$label" "$status" "$rel/$csv_name"
  done
  exit 0
fi

echo "PSB          = $PSB"
echo "RULE_EXPERTS = $RULE_EXPERTS"
echo "ONLY         = ${ONLY:-<all>}"
echo "DRY_RUN      = $DRY_RUN"

n_ok=0
n_skip=0
for job in "${DEFAULT_JOBS[@]}"; do
  IFS='|' read -r label rel csv_name <<<"$job"
  if ! should_run_label "$label"; then
    continue
  fi
  out_dir="$PSB/$rel"
  csv="$out_dir/$csv_name"
  if [ ! -f "$csv" ]; then
    echo "[skip] $label — missing $csv"
    n_skip=$((n_skip + 1))
    continue
  fi
  run_one "$label" "$out_dir" "$csv_name"
  n_ok=$((n_ok + 1))
done

echo
echo "Done: $n_ok ran, $n_skip skipped (missing csv)."
