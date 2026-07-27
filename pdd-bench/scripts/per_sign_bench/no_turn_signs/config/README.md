Hydra config for the **3.18.1 / 3.18.2** no-turn family.

```bash
# Default: 3.18.1
python generate_manifest.py

python generate_manifest.py sign.pdd_code=3.18.2
```

| Key | Default | Notes |
|-----|---------|-------|
| `pdd_code` | `3.18.1` | Active member (`lib/no_turn_sign_spec.py`). |
| `scenes_dir` | `null` → `scenes/<slug>` | Cropped scenes for the active `sign.pdd_code` |
| `output_base` | `benchmark_output/<slug>` | Via `${pdd_slug:...}` resolver |
| `scenario.max_scenarios` | `null` | Cap dual-path geometry picks (`null` = all); density tiers multiply rows on top. |
| `scenario.max_scenarios_per_scene` | `5` | Cap dual-path variants per crop (`null` = all). |
| `simulation.traffic_density_augment` | `true` | If true: multiply rows by nuPlan density tiers. If false: one row per scenario with fixed `traffic_density` (use `0.0` for no NPCs). |

Auxiliary traffic is **off**: the task is route compliance (forbidden vs allowed first exit).

```bash
# No density-tier augmentation (single row per dual-path scenario, no NPCs):
python generate_manifest.py simulation.traffic_density_augment=false simulation.traffic_density=0.0

# Cap total dual-path scenarios (density tiers still multiply rows if augment=true):
python generate_manifest.py scenario.max_scenarios=10
```
