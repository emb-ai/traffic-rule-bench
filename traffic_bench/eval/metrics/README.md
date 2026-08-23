# metrics/ — episode JSONL → CSV / markdown

Used after `python -m traffic_bench.eval run policies=all sign=yield`, or on its own:

```bash
python -m traffic_bench.eval metrics csv --episodes-root <eval_out>/benchmark --out <eval_out>/metrics_per_episode.csv
python -m traffic_bench.eval metrics aggregate --csv <eval_out>/metrics_per_episode.csv --out-dir <eval_out>
python -m traffic_bench.eval metrics report --run-root <eval_out>
```

1. `csv.py` — episodes / replays → `metrics_per_episode.csv`
2. `aggregate.py` — CSV → aggregations + `reports/cumulative.json`
3. `report.py` — cumulative JSON → markdown table
