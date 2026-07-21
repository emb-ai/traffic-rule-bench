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

Auxiliary traffic is **off**: the task is route compliance (forbidden vs allowed first exit).
