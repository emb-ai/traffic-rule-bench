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
| `simulation.sign_distance_from_start` | `10` | Sign offset from start of forbidden (destination) lane (m). |
| `simulation.destination_past_sign_m` | `8` | Route end this far past the sign (short finish for non-compliant). Forbidden lane must be longer than `sign_distance_from_start + destination_past_sign_m`. |
| `simulation.spawn_distance_before_end` | `25` | Ego spawn offset from approach lane end (m). |
| `simulation.compliant_stop_success_seconds` | `3` | Stopped before sign this long → arrive_dest (**only** this override). |
