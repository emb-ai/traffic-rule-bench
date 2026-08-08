#!/usr/bin/env bash
# Rebuild per-checkpoint reports from whatever is already on disk (no re-simulation).
#
#   bash refinalize.sh                      # every run under output/
#   bash refinalize.sh 'fvexp30_lr1e4_*'    # only matching runs
#   POLICY=plant2_rule bash refinalize.sh   # baseline key of another policy
#
# Wrapper labels (per-sign eval_pipeline) keep their metrics in replay sidecars
# under <label>/eval_out/runs, direct tasks (speed + detour) in episodes_*.jsonl
# under direct/*/parts/*/policy_eval — hence the two different source flags.
# Labels that already have reports/cumulative.json are left untouched.
set -uo pipefail
cd "$(dirname "$0")"

POLICY=${POLICY:-plant2}
GLOB=${1:-*}

for run_dir in output/$GLOB; do
    [ -d "$run_dir" ] || continue
    rn=$(basename "$run_dir")
    echo "########## $rn"

    for d in "$run_dir"/*/eval_out; do
        [ -d "$d" ] || continue
        [ -f "$d/reports/cumulative.json" ] && continue
        label=$(basename "$(dirname "$d")")
        if [ ! -d "$d/runs" ]; then
            echo "  -- $label: no replay sidecars (runs/), needs a resumed eval run"
            continue
        fi
        mkdir -p "$d/_no_manifests"
        {
            python3 ../build_episode_metrics_csv.py --runs-root "$d/runs" --vars 0 \
                --out "$d/metrics_per_episode.csv" --manifests-root "$d/_no_manifests" &&
            python3 ../aggregate_episode_metrics.py --csv "$d/metrics_per_episode.csv" \
                --out-dir "$d" &&
            python3 ../generate_cumulative_markdown_report.py --run-root "$d" \
                --cumulative "$d/reports/cumulative.json"
        } > "$d/_refinalize.log" 2>&1 &&
            echo "  ok $label" ||
            echo "  !! $label failed — see $d/_refinalize.log"
    done

    python3 summarize_reports.py --run-name "$rn" --baseline "${POLICY}_default" || true

    DIR="$run_dir/direct"
    [ -d "$DIR" ] || continue
    COMB="$DIR/metrics_per_episode.csv"
    : > "$COMB"
    while read -r pe; do
        tag=$(echo "$pe" | md5sum | cut -c1-8)
        python3 ../build_episode_metrics_csv.py --episodes-root "$pe" \
            --out "$DIR/_csv_$tag.csv" >> "$DIR/_refinalize.log" 2>&1 || continue
        if [ ! -s "$COMB" ]; then cat "$DIR/_csv_$tag.csv" > "$COMB"
        else tail -n +2 "$DIR/_csv_$tag.csv" >> "$COMB"; fi
    done < <(find "$DIR" -type d -name policy_eval | sort)
    if [ -s "$COMB" ]; then
        python3 ../aggregate_episode_metrics.py --csv "$COMB" --out-dir "$DIR" \
            >> "$DIR/_refinalize.log" 2>&1 || true
        python3 ../generate_cumulative_markdown_report.py --run-root "$DIR" \
            --cumulative "$DIR/reports/cumulative.json" >> "$DIR/_refinalize.log" 2>&1 || true
        echo "  direct -> $DIR/reports/report_cumulative.md"
    else
        echo "  direct: no episodes yet"
    fi
done

echo
echo "summaries:  output/<run>/_summary/summary.md"
echo "direct:     output/<run>/direct/reports/report_cumulative.md"
