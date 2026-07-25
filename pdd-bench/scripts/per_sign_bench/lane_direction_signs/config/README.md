# Configuration

Hydra config for **PDD 5.15.1** (lane direction board).

## Usage

```bash
python generate_manifest.py
python generate_manifest.py scenario.max_scenarios=20
python generate_manifest.py gif.enabled=true gif.policy=carl_rule
```

## `scenario`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_variants` | `1` | Repetitions per dual-path pick |
| `augment` | `false` | Extra ego/aux spawn enumeration (usually off) |
| `max_scenarios` | `null` | Cap **total** dual-path geometry picks (`null` = all). With `traffic_density_augment`, each pick expands into density tiers on top. |
| `max_scenarios_per_scene` | `5` | Cap dual-path picks **per cropped scene** |
| `respect_scene_selection` | `true` | Skip scenes marked reject / under `_rejected/` |
| `min_dual_path_gain_m` | `0.0` | Optional wrong-spur vs correct-path length gap |
| `min_ego_lane_m` | `21.0` | Min approach length for spawn ≥20 m before junction |
| `validate_metadrive_routes` | `true` | Keep only MetaDrive-routable target→dest paths |

## Output

```
benchmark_output/5_15_1/<experiment_name>/
├── real_manifest.jsonl
├── manifest.json
├── real_manifest_summary.json
├── config.yaml
└── gifs/                    # If gif.enabled=true
```
