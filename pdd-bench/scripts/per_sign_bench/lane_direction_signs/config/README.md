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
| `n_variants` | `1` | Dual-path index repetitions (usually 1) |
| `n_variations` | `5` | Like `sumo_space`: NPC/traffic seeds via nuPlan `sample_one_profile` |
| `augment` | `false` | Extra ego/aux spawn enumeration (usually off) |
| `max_scenarios` | `null` | Cap **total** dual-path geometry picks (`null` = all). Each pick expands by `n_variations` (or density tiers if enabled). |
| `max_scenarios_per_scene` | `5` | Cap dual-path picks **per cropped scene** |
| `respect_scene_selection` | `true` | Skip scenes marked reject / under `_rejected/` |
| `min_dual_path_gain_m` | `0.0` | Optional wrong-spur vs correct-path length gap |
| `min_ego_lane_m` | `21.0` | Min approach length for spawn ≥20 m before junction |
| `validate_metadrive_routes` | `true` | Keep only MetaDrive-routable target→dest paths |

## `simulation`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `traffic_density_augment` | `false` | If true: fixed td1/td2/td3. If false: use `n_variations` nuPlan profiles |
| `nuplan_density_cap` | `1.0` | Cap for sampled `traffic_density` (same as sumo_space) |
| `spawn_velocity_ms` | `2.5` | Ego initial speed |
| `spawn_distance_before_end` | `20.0` | Ego spawn distance before junction (m) |

```bash
# Cap scenarios + no NPC density tiers (single nuPlan profile / zero density):
python generate_manifest.py scenario.max_scenarios=10 scenario.n_variations=1 simulation.traffic_density=0.0

# Fixed density tiers instead of n_variations profiles:
python generate_manifest.py simulation.traffic_density_augment=true
```

Expert lane-change (5.15.1) is steering-only: no physical teleport onto the
allowed via after the peer LC; nav/checkpoints are updated without body snaps.

## Output

```
benchmark_output/5_15_1/<experiment_name>/
├── real_manifest.jsonl
├── manifest.json
├── real_manifest_summary.json
├── config.yaml
└── gifs/                    # If gif.enabled=true
```
