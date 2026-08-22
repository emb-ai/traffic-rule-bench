# assign/ — signs as queries over the map pool

Reads `maps/splits/signs.yaml` plus the global train/test ids and writes
`maps/splits/sign_allocations.json` (keyed by PDD code).

```bash
python -m traffic_bench.scene_collection assign
```

`crop_kind` is `junction` (default), `dual_path`, or `segment`. Signs sample
independently; the same map may appear under several signs. Place identity
(`junction_id` / `osm_way_id`) is stamped **before** this step.
