# collect/

OSM → [`../maps/`](../maps/README.md).

```bash
python -m traffic_bench.scene_collection collect --skip-existing
```

| | |
| --- | --- |
| `build_net.py` | download OSM, convert to SUMO |
| `make_split.py` | train / test on junction ids |
| [`enumerate/`](enumerate/README.md) | list junctions and corridors |
| [`junctions/`](junctions/README.md) | crop T / X / O |
| [`dual_path/`](dual_path/README.md) | crop two-route nets |
| [`segments/`](segments/README.md) | crop corridors |
| [`lib/`](lib/README.md) | shared helpers |
