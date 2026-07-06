# Configuration

Hydra-based configuration for manifest generation.

## Usage

```bash
# Default configuration
python generate_manifest.py

# Override parameters
python generate_manifest.py auxiliary.lanes_occupied=2 auxiliary.convoy_size=3 gif.enabled=true

# Custom experiment name
python generate_manifest.py paths.experiment_name=my_experiment
```

## Configuration Groups

### `paths`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scenes_dir` | `null` (= `./scenes`) | Input scenes directory |
| `output_base` | `benchmark_output/2_4` | Base output directory |
| `experiment_name` | `<timestamp>` | Experiment folder name |

### `scenario`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_variants` | `1` | Number of variants per scene |
| `augment` | `true` | Enable scenario augmentation |
| `max_scenarios_per_scene` | `null` | Limit scenarios per scene |

### `simulation`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `spawn_velocity_ms` | `2.5` | Ego initial velocity (m/s) |
| `traffic_density` | `0.0` | Background traffic density |
| `horizon` | `600` | Max simulation steps |
| `sign_distance_before_end` | `0.0` | Sign placement offset |
| `spawn_distance_before_end` | `20.0` | Ego spawn distance from lane end |

### `auxiliary`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable auxiliary (main road) agents |
| `distance_from_intersection` | `20.0` | Aux spawn distance from junction |
| `convoy_size` | `1` | Vehicles per lane (convoy depth) |
| `lanes_occupied` | `1` | Number of main lanes with aux agents |
| `convoy_gap_m` | `10.0` | Gap between convoy vehicles |

### `gif`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Generate visualization GIFs |
| `policy` | `idm` | Policy for GIF rendering |
| `max_scenes` | `null` | Limit GIFs to generate |
| `dry_run` | `false` | Skip actual rendering |
| `hide_signs` | `true` | Hide traffic signs in GIFs |
| `dir` | `null` (= `<exp>/gifs`) | GIF output directory |
| `run_name` | `null` | Custom run name |

## Output Structure

Each run creates a timestamped folder:

```
benchmark_output/2_4/<experiment_name>/
├── real_manifest.jsonl      # Scenario definitions
├── manifest.json            # Manifest metadata
├── real_manifest_summary.json
├── config.yaml              # Resolved config
├── .hydra/                  # Hydra metadata
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
└── gifs/                    # If gif.enabled=true
```
