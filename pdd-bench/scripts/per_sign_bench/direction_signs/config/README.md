# Configuration

Hydra config for the **4.1.1–4.1.6** direction-sign family.

## Usage

```bash
# Default: 4.1.1
python generate_manifest.py

# Another family member
python generate_manifest.py sign.pdd_code=4.1.3 paths.output_base=benchmark_output/4_1_3

# Overrides
python generate_manifest.py auxiliary.enabled=true gif.enabled=true
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

### `scenario` / `simulation` / `auxiliary` / `gif`

Same shapes as other junction benches (`main_sign`, `stop_sign`). Auxiliary is
**off by default** until direction-aware scene generation exists.

## Output

```
benchmark_output/4_1_1/<experiment_name>/
├── real_manifest.jsonl
├── manifest.json
├── real_manifest_summary.json
├── config.yaml
└── gifs/                    # If gif.enabled=true
```
