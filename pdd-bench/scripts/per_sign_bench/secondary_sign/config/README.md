# Configuration

Hydra-based configuration for secondary-road sign manifest generation.

## Usage

```bash
python generate_manifest.py
python generate_manifest.py auxiliary.lanes_occupied=2 gif.enabled=true
```

## Key defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `paths.output_base` | `benchmark_output/2_3` | Base output directory |
| `scenario.max_scenarios_per_scene` | `100` | Cap manifest rows per scene |
| `auxiliary.enabled` | `true` | Spawn main-road auxiliary agents |

## Output

```
benchmark_output/2_3/<experiment_name>/
├── real_manifest.jsonl
├── manifest.json
└── config.yaml
```
