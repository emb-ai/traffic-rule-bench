# sign_scenes/filter/ — keep / reject after materialize

- `reject.py` — drop maps that cannot produce a manifest row; optional `--refill`
- `review.py` — local web UI over scene previews
- `selection.py` — `scene_selection.json` (`keep` / `reject` / `pending`)

```bash
python -m traffic_bench.scene_collection reject --sign yield --apply --refill --loop
python -m traffic_bench.scene_collection reject --all --apply --refill --loop
python -m traffic_bench.scene_collection review --scenes-dir data/scenes/yield

# Crosswalk defaults to metadrive_first.png (override with --preview-name)
python -m traffic_bench.scene_collection review --scenes-dir data/scenes/crosswalk

# After fixing viability: put falsely-rejected maps back, then re-run reject
python -m traffic_bench.scene_collection reject --sign crosswalk --restore-rejected
```

Segment signs (`crosswalk`, `speed_*`, `detour_*`) use corridor viability
(not junction T/X). See `manifest_viability._check_crosswalk_viability` /
`_check_speed_zone_viability` / `_check_detour_viability`.
