# metrics/

Eval report core used by `priority_bench/eval_pipeline.py`:

1. `build_episode_metrics_csv.py` — episodes / replays → `metrics_per_episode.csv`
2. `aggregate_episode_metrics.py` — CSV → aggregations + `reports/cumulative.json`
3. `generate_cumulative_markdown_report.py` — cumulative JSON → markdown table

Still imports `select_experts` from `priority_bench/oracle/`.
