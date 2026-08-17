# Dataset statistics (reviewer overview)

Scripts that compute a **high-level statistical overview** of the PDD-Bench
scene/scenario set for the sign subset:

`2.1`, `2.3.1–2.3.3`, `2.4`, `2.5`, `3.1–3.2`, `3.24`, `4.2.1–4.2.3`, `4.3`, `4.6`,
`5.7.1–5.7.2`, `5.15.1`, `5.19`, `5.21`, `5.31`.

Speed / zone signs load from map-trimmed
`catalog_balanced_1k2.jsonl`. Detour signs load from
`benchmark_output/detour_v1/catalog.jsonl`.

## Quick start

```bash
cd pdd-bench/scripts/dataset_stats
python run_all.py
# or step-by-step:
python collect_stats.py
python plot_and_report.py
```

## Outputs

| Path | Contents |
|---|---|
| `output/DATASET_OVERVIEW.md` | Reviewer-ready markdown with inline tables + figure links |
| `output/tables/` | `sign_distribution`, `agents_and_duration`, `duration_by_planner`, `map_complexity`, `category_distribution` (`.md` + `.csv`) |
| `output/figures/` | PNG plots (distribution, agents, duration, map complexity, geo footprint) |
| `output/raw/` | Machine-readable JSON/JSONL (`overview.json`, `sign_summary.json`, inventories) |

## Definitions

| Unit | Meaning |
|---|---|
| **Catalog scene** | OSM crop under `pdd-bench/scenes/<sign>/` |
| **Package scene** | Filtered crop under `per_sign_bench/<pkg>/scenes/` |
| **Scenario** | Row in `final_metrics_v1/real_manifest.jsonl` (spawn / density / pedestrian augmentation) |
| **Agents** | Nominal: ego + aux convoy×lanes **or** ego + nuPlan vehicles/frame (+ pedestrians for 5.19) |
| **Horizon** | `horizon_steps × 0.1 s` (default 600 → 60 s) |
| **Realized duration** | Weighted `avg_steps × 0.1 s` from eval aggregations |

## Notes

- `5.15.1` lives under `lane_direction_signs` (no top-level `scenes/5.15.1` mirror).
- `2.3.1/2.3.2/2.3.3` share the `secondary_sign/scenes/2_3` package; manifests are split by `pdd_code`.
- Speed signs use map-trimmed `catalog_balanced_1k2.jsonl` (~1200 scenarios / sign).
  Rebuild with: `python trim_speed_catalog.py --target-scenarios 1200`
