# sign_scenes/filter/ — keep / reject after materialize

- `reject.py` — drop maps that cannot produce a manifest row; optional `--refill`
- `review.py` — local web UI over `custom_cropped.png`
- `selection.py` — `scene_selection.json` (`keep` / `reject` / `pending`)

```bash
python -m traffic_bench.scene_collection reject --sign yield --apply --refill --loop
python -m traffic_bench.scene_collection review --scenes-dir data/scenes/yield
```
