# sign_scenes/materialize/ — allocations → data/scenes/<sign>/

```bash
python -m traffic_bench.scene_collection materialize --sign yield
```

Default is a **relative symlink** into `maps/crops/…` (survives NFS remount).
`--mode copy` writes real directories. Signs with `prepare: crosswalk` in yaml
are always copied so inject does not mutate the shared pool; `materialize`
then runs that hook itself.

- `run.py` — CLI used by `materialize` / `--refill`
- `pool_index.py` — `moscow_pool.json` bookkeeping (filename kept for eval)

`--refill` uses the same tiered place-reuse policy as `assign` (via
`assign/refill_pick.py`): unique place → same behavioral family → same semantic
group; never cross-semantic. Segment / dual_path refill from their own crop
pools (not junction `train_ids`).
