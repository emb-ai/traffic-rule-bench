#!/usr/bin/env bash
# preview_metrics.sh — build in_zone/compliance metrics from whatever episodes
# are ALREADY on disk in an eval OUT dir, WITHOUT waiting for the full run or a
# clean per-job exit.
#
# Reads parts/*/benchmark/policy_eval/<run>/episodes_*.jsonl (written
# incrementally per episode), so it captures finished, in-progress AND rc!=0
# jobs (e.g. a stream that crashed on its last policy still has the episodes of
# the earlier policies). Writes to <OUT>/preview/ — never clobbers the live
# run's own metrics_per_episode.csv / reports.
#
#   bash preview_metrics.sh /path/to/.../eval
set -uo pipefail
OUT="${1:?usage: preview_metrics.sh <eval_out_dir>}"
PY="${PY:-/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/plant2/bin/python}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)     # scripts/per_sign_bench
REPO=$(cd "$HERE/../.." && pwd)                        # pdd-bench
PREV="$OUT/preview"; EMPTY="$PREV/_no_manifests"
mkdir -p "$PREV" "$EMPTY"
cd "$REPO"

echo "[preview] per-part CSVs from episodes ..."
n=0
while read -r pe; do
  [ -n "$pe" ] || continue
  [ -n "$(find "$pe" -name 'episodes_*.jsonl' -print -quit 2>/dev/null)" ] || continue
  tag=$(basename "$(dirname "$(dirname "$pe")")")      # eval_<tag>
  if "$PY" "$HERE/build_episode_metrics_csv.py" --episodes-root "$pe" \
        --out "$PREV/csv_$tag.csv" --manifests-root "$EMPTY" >/dev/null 2>&1; then
    echo "  $tag: $(($(wc -l < "$PREV/csv_$tag.csv")-1)) episodes"; n=$((n+1))
  else
    echo "  $tag: build failed (skipped)" >&2
  fi
done < <(find "$OUT/parts" -type d -name policy_eval | sort)
[ "$n" -gt 0 ] || { echo "no episodes found under $OUT/parts" >&2; exit 1; }

echo "[preview] merging $n part CSVs ..."
COMB="$PREV/metrics_per_episode.csv"; : > "$COMB"
while read -r c; do
  [ -s "$c" ] || continue
  if [ -s "$COMB" ]; then tail -n +2 "$c" >> "$COMB"; else cat "$c" > "$COMB"; fi
done < <(find "$PREV" -maxdepth 1 -name 'csv_*.csv' | sort)
echo "[preview] merged $(($(wc -l < "$COMB")-1)) episodes"

echo "[preview] aggregate + report ..."
"$PY" "$HERE/aggregate_episode_metrics.py" --csv "$COMB" --out-dir "$PREV"
"$PY" "$HERE/generate_cumulative_markdown_report.py" --run-root "$PREV"
echo "REPORT: $PREV/reports/report_cumulative.md"
