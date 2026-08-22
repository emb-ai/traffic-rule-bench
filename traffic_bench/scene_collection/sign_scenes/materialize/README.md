# sign_scenes/materialize/ — allocations → data/scenes/<sign>/

```bash
python -m traffic_bench.scene_collection materialize --sign yield
```

Default is a **relative symlink** into `maps/crops/…` (survives NFS remount).
`--mode copy` writes real directories. Signs with `prepare: crosswalk` in yaml
are always copied so later inject does not mutate the shared pool.

- `run.py` — CLI used by `materialize` / `--refill`
- `pool_index.py` — `moscow_pool.json` bookkeeping (filename kept for eval)
