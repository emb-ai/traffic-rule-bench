# analysis/ — harvest counts and diversity

Reads `maps/index/` + `maps/crops/`. Writes figures to [`figures/`](figures/).

```bash
python -m traffic_bench.scene_collection analysis
python -m traffic_bench.scene_collection analysis --pdf   # also write PDF
```

See [`figures/inventory.md`](figures/inventory.md) for the latest counts.

| File | Role |
| --- | --- |
| `inventory.py` | load indexes, count crops |
| `figures.py` | matplotlib |
| `report.py` | `figures/inventory.md` + `summary.json` |
| `run.py` | CLI |
