# metrics/ — episode JSONL → CSV / markdown

Used by `python -m traffic_bench.eval pipeline` and:

```bash
python -m traffic_bench.eval metrics csv --episodes-root <eval_out>/benchmark --out <eval_out>/metrics_per_episode.csv
python -m traffic_bench.eval metrics aggregate --csv <eval_out>/metrics_per_episode.csv --out-dir <eval_out>
python -m traffic_bench.eval metrics report --run-root <eval_out>
```

1. `build_episode_metrics_csv.py` — episodes / replays → `metrics_per_episode.csv`
2. `aggregate_episode_metrics.py` — CSV → aggregations + `reports/cumulative.json`
3. `generate_cumulative_markdown_report.py` — cumulative JSON → markdown table
