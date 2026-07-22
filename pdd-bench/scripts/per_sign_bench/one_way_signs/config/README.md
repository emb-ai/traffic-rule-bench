Hydra config for the **5.7.1 / 5.7.2** one-way-entry family.

```bash
# Default: 5.7.1
python generate_manifest.py

python generate_manifest.py sign.pdd_code=5.7.2
```

| Key | Default | Notes |
|-----|---------|-------|
| `pdd_code` | `5.7.1` | Active member (`lib/one_way_sign_spec.py`). |
| `scenes_dir` | `null` → `scenes/<slug>` | Cropped scenes for the active `sign.pdd_code` |
| `output_base` | `benchmark_output/<slug>` | Via `${pdd_slug:...}` resolver |

Auxiliary traffic is **off**: the task is route compliance (forbidden vs allowed first exit).
