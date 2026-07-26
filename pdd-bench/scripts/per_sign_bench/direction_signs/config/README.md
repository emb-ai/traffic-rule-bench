# Configuration

Hydra config for the **4.1.1–4.1.6** direction-sign family.

## Usage

```bash
# Default: 4.1.1
python generate_manifest.py

# Another family member
python generate_manifest.py sign.pdd_code=4.1.3 paths.output_base=benchmark_output/4_1_3

# Overrides
python generate_manifest.py gif.enabled=true
```

## Groups

### `sign`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `pdd_code` | `4.1.1` | Active member of 4.1.1–4.1.6 (`lib/direction_sign_spec.py`). Dual-path crop/manifest supported for all six codes. |

### `paths`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scenes_base` | `scenes` | Parent of per-sign folders (`scenes/4_1_1`, …) |
| `scenes_dir` | `null` → `scenes/<slug>` | Cropped scenes for the active `sign.pdd_code` |
| `output_base` | `benchmark_output/4_1_1` | Base output directory |
| `experiment_name` | `<timestamp>` | Experiment folder name |

### `scenario` / `simulation` / `gif`

Same shapes as other junction benches (`main_sign`, `stop_sign`). This family
has **no auxiliary agents**: the 4.1.x task is route compliance, so scenes
contain only the ego vehicle plus background traffic.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scenario.max_scenarios` | `null` | Cap dual-path geometry picks (`null` = all); density tiers multiply rows on top. |
| `scenario.max_scenarios_per_scene` | `5` | Cap dual-path variants per crop (`null` = all). |
| `simulation.traffic_density_augment` | `true` | If true: multiply rows by nuPlan density tiers. If false: one row per scenario with fixed `traffic_density` (use `0.0` for no NPCs). |

`simulation.traffic_density_augment: true` (default) expands each dual-path
scenario into 3 rows with nuPlan-derived background-traffic densities
(p25/p50/p75 of vehicles per frame; see `lib/traffic_density_levels.py`).
Rows get `scene_id` suffixes `_td1`/`_td2`/`_td3` and distinct seeds. Set it
to `false` to emit a single row with `simulation.traffic_density`.

```bash
# No density-tier augmentation (single row per dual-path scenario, no NPCs):
python generate_manifest.py simulation.traffic_density_augment=false simulation.traffic_density=0.0

# Cap total dual-path scenarios (density tiers still multiply rows if augment=true):
python generate_manifest.py scenario.max_scenarios=10
```

Background traffic (when density > 0) is skill-bench friendly:
- `npc_ego_yield_radius=15` — NPCs brake for ego in a front hemisphere
- NPC↔NPC crashes remove both cars (no pile-up blocking the scene)
- spawn keep-out around ego is 30 m (`SumoTrafficManager.EGO_SAFE_RADIUS`)

## Output

```
benchmark_output/4_1_1/<experiment_name>/
├── real_manifest.jsonl
├── manifest.json
├── real_manifest_summary.json
├── config.yaml
└── gifs/                    # If gif.enabled=true
```
