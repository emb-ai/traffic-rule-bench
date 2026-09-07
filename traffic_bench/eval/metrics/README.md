# metrics/

Used after `python -m traffic_bench.eval run policies=all sign=yield`, or on its own:

```bash
python -m traffic_bench.eval metrics csv --episodes-root <eval_out>/benchmark/full/policy_eval --out <eval_out>/metrics_per_episode.csv
python -m traffic_bench.eval metrics aggregate --csv <eval_out>/metrics_per_episode.csv --out-dir <eval_out>
python -m traffic_bench.eval metrics report --run-root <eval_out>
python -m traffic_bench.eval metrics combine sign=all
python -m traffic_bench.eval metrics plot
```

1. `csv.py` — episodes / replays → `metrics_per_episode.csv`
2. `aggregate.py` — CSV → aggregations + `reports/cumulative.json`
3. `report.py` — cumulative JSON → markdown table
4. `combine.py` — per-sign CSVs → `data/runs/_all/…/reports/report_cumulative.md`
5. `plot_benchmark.py` — scan ready reports → comparison PNGs + `analysis.md`

### Baseline comparison plots

After per-sign `report` steps (or while train eval is still running on finished signs):

```bash
python -m traffic_bench.eval metrics plot
python -m traffic_bench.eval metrics plot --agg map --out-dir docs/static/images/benchmark
```

Reads `data/runs/*/train/eval_out/reports/cumulative.json`, merges `per_sign`
slices, and writes to `data/runs/_all/train/plots/benchmark/` by default:

- `by_sign/<code>.png` — one figure per sign; three metrics; base vs rule expert side by side (IDM / PPO / CaRL)
- `by_family/<group>.png` — four semantic groups (priority / speed / lane obstacle / re-routing), sign compliance, six baselines per sign
- `all_signs_sign_compliance.png` — all signs on one chart, dashed lines separate groups
- `overall_summary.png` — macro average
- `analysis.md` — short auto-generated commentary

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

