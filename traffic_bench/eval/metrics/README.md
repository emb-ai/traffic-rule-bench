# metrics/

Used after `python -m traffic_bench.eval run policies=all sign=yield`, or on its own:

```bash
python -m traffic_bench.eval metrics csv --episodes-root <eval_out>/benchmark/full/policy_eval --out <eval_out>/metrics_per_episode.csv
python -m traffic_bench.eval metrics aggregate --csv <eval_out>/metrics_per_episode.csv --out-dir <eval_out>
python -m traffic_bench.eval metrics report --run-root <eval_out>
python -m traffic_bench.eval metrics combine sign=all
```

1. `csv.py` — episodes / replays → `metrics_per_episode.csv`
2. `aggregate.py` — CSV → aggregations + `reports/cumulative.json`
3. `report.py` — cumulative JSON → markdown table
4. `combine.py` — per-sign CSVs → `data/runs/_all/…/reports/report_cumulative.md`

## Two aggregations, always both

Every slice is aggregated two ways and both are written:

| Kind | Files / JSON blocks | Meaning |
|---|---|---|
| per-episode | `aggregations/agg_per_*.csv`, `per_baseline` / `per_sign` | every episode weighs the same (original) |
| per-map | `aggregations/agg_per_*_map.csv`, `per_baseline_map` / `per_sign_map` | each map's episodes are collapsed first, then the mean is taken over maps (`n_maps`) |

`report.py` prints both in every cell as `episode / map`.

## SR&Dest

`sr_and_dest` (per-episode column in `metrics_per_episode.csv`, rate in every
aggregation, `SR&Dest` column in the report) = target sign obeyed
(`target_compliant_event`) **and** destination reached (`arrived_dest`), over
every scored episode. A run that crashes before the sign has no violation but
no arrival either, so it scores 0 instead of passing as compliant.

