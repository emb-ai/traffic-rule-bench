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
| `pdd_code` | `4.1.1` | Active member of 4.1.1–4.1.6 (`lib/direction_sign_spec.py`) |

### `paths`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scenes_dir` | `scenes` | Input scenes directory |
| `output_base` | `benchmark_output/4_1_1` | Base output directory |
| `experiment_name` | `<timestamp>` | Experiment folder name |

### `scenario` / `simulation` / `gif`

Same shapes as other junction benches (`main_sign`, `stop_sign`). This family
has **no auxiliary agents**: the 4.1.x task is route compliance, so scenes
contain only the ego vehicle plus background traffic.

`simulation.traffic_density_augment: true` (default) expands each dual-path
scenario into 3 rows with nuPlan-derived background-traffic densities
(p25/p50/p75 of vehicles per frame; see `lib/traffic_density_levels.py`).
Rows get `scene_id` suffixes `_td1`/`_td2`/`_td3` and distinct seeds. Set it
to `false` to emit a single row with `simulation.traffic_density`.

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
