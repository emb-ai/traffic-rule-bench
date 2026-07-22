Hydra config for the **3.1 / 3.2** no-entry family.

```bash
# Default: 3.1
python generate_manifest.py

python generate_manifest.py sign.pdd_code=3.2
```

| Key | Default | Meaning |
| --- | ------- | ------- |
| `pdd_code` | `3.1` | Active member (`lib/no_entry_sign_spec.py`). |
| `paths.scenes_dir` | auto `scenes/<slug>` | Per-member scene root. |
| `paths.output_base` | `benchmark_output/<slug>` | Per-member outputs. |
| `scenario.spawn_margin_before_sign_m` | `15` | Ego spawn distance before catalog sign. |
