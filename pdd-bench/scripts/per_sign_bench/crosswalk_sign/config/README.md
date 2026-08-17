# Configuration

Hydra-based configuration for crosswalk (5.19) manifest generation.

## Usage

```bash
python generate_manifest.py
python generate_manifest.py pedestrian.initial_pedestrians=3 gif.enabled=true
python generate_manifest.py paths.experiment_name=my_experiment
```

## Configuration Groups

### `paths`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scenes_dir` | `scenes` | Input scenes directory |
| `output_base` | `benchmark_output/5_19` | Base output directory |
| `experiment_name` | `<timestamp>` | Experiment folder name |

### `scenario`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_variants` | `1` | Variants per crosswalk approach |
| `augment` | `true` | Enable variant enumeration |
| `max_scenarios_per_scene` | `null` | Limit crosswalk approaches per scene (before preset/density expansion) |
| `max_entries_per_scene` | `null` | Shuffle all expanded combos per scene and keep at most N manifest rows |

### `simulation`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `spawn_velocity_ms` | `2.5` | Ego initial velocity (m/s) |
| `traffic_density` | `0.0` | Background traffic density when augment is off |
| `traffic_density_augment` | `true` | Emit 3 nuPlan-derived density levels per scenario |
| `horizon` | `600` | Max simulation steps |
| `spawn_distance_before_end` | `20.0` | Ego spawn distance before crosswalk (m) |

### `pedestrian`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_pedestrians` | `2` | Pedestrians at episode start |
| `max_pedestrians` | `6` | Max simultaneous pedestrians |
| `spawn_probability` | `0.12` | Per-step spawn chance |
| `crossing_interval_range` | `[5, 10]` | Seconds between crossings |
| `yield_distance` | `12.0` | Yield zone distance (m) |
| `no_stop_before_crosswalk_m` | `3.0` | No-stop zone before crosswalk (m) |

### `gif`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Generate visualization GIFs |
| `policy` | `idm` | Policy for GIF rendering |
| `model_path` | `null` | Checkpoint override; `carl`/`plant2*` use `pdd-bench/checkpoints/` defaults |
| `max_scenes` | `null` | Limit GIFs to generate |

## Output Structure

```
benchmark_output/5_19/<experiment_name>/
├── real_manifest.jsonl
├── manifest.json
├── real_manifest_summary.json
├── config.yaml
└── gifs/                    # If gif.enabled=true
```
