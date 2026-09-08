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
| per-map | `aggregations/agg_per_*_map.csv`, `per_baseline_map` / `per_sign_map` | each map's episodes are collapsed first, then the mean is taken over maps (`n_maps`, `episodes_per_map_min` / `_max`) |
| per-map dispersion | `aggregations/agg_per_*_map_ci.csv`, `per_baseline_map_ci` / `per_sign_map_ci` | per metric: the per-map mean, the std over maps and a bootstrap CI of the mean (parameters in the `ci` block) |

`report.py` prints `episode / map` in every cell and follows every table with a
dispersion table (`mean ± std [lo, hi]`).

### What a map is

`map_id` (column in `metrics_per_episode.csv`) names the physical net: the
directory of the manifest's `net_path` (`seg_1067603714/map.net.xml` →
`seg_1067603714`). Every augmented variant of that net — manifest variant
`_v<k>`, spawn lane `_l<n>`, world cell `_rl90_td50_sv1_v2` — shares the id,
so a map with five variants contributes one number, not five. `metrics csv`
takes the manifest from `chunks/var_*/var_*.jsonl`, else from the
`real_manifest.jsonl` next to `eval_out/` (or `--manifests-root`); without one
the suffix is stripped from `scene_id` (`map_id.py`), and CSVs written before
the column existed get the same fallback on load. Within a per-baseline slice
the key is `<pdd_code>|<map_id>`: the same net under another sign is another
scenario.

### mean, std, CI

For every averaged metric of a slice (rates and `avg_*`):

1. each map → the mean over its augmented variants (`aggregate()` on the map's
   episodes);
2. `mean` = the mean of the per-map values — the `map` number of
   `episode / map` (with a balanced design it equals the per-episode mean);
3. `std` = the sample standard deviation (ddof=1) of the per-map values;
4. `[ci_lo, ci_hi]` = percentile bootstrap CI of the mean: the per-map values
   are resampled with replacement `--n-boot` times (default 10000), the mean of
   each resample is taken, and the interval is cut at the `(1 - level) / 2`
   tails (`--ci-level`, default 0.95). The generator is seeded per slice
   (`--ci-seed`, default 0), so an interval does not depend on which other
   slices were computed, and all metrics of a slice share the same resample
   draws. `--n-boot 0` skips the CI (std only).

Counts (`n`, `n_in_zone`, violation totals) are summed over maps, not
averaged, and get no interval.

## SR&Dest

`sr_and_dest` (per-episode column in `metrics_per_episode.csv`, rate in every
aggregation, `SR&Dest` column in the report) = target sign obeyed
(`target_compliant_event`) **and** destination reached (`arrived_dest`), over
every scored episode. A run that crashes before the sign has no violation but
no arrival either, so it scores 0 instead of passing as compliant.

