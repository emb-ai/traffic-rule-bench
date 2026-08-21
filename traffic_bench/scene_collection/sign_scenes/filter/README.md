# filter/

```bash
python -m traffic_bench.scene_collection reject --sign yield --apply --refill --loop
python -m traffic_bench.scene_collection review --scenes-dir data/scenes/yield
```

`reject` drops maps that cannot make a scenario. `review` is a keep/reject UI.
Verdicts live in `scene_selection.json`.
