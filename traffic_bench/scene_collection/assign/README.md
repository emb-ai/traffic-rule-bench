# assign/

`signs.yaml` + train/test ids → `sign_allocations.json`.

```bash
python -m traffic_bench.scene_collection assign
```

Each sign samples on its own. The same map can appear under several signs.
