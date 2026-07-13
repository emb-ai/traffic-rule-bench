# Crosswalk scene pipeline

Full workflow (same structure as other per-sign benches):

```bash
# 1. Import catalog maps into scenes/core/
python tools/filter_scenes/import_catalog_scenes.py --limit 40

# 2. Crop each pedestrian crossing into scenes/sign_<id>_cw<rank>/
python tools/filter_scenes/build_scene_pool.py crop --target 100

# 3. Review previews (custom_cropped.png) and mark keep/reject
python tools/filter_scenes/review_junction_scenes.py

# 4. Add more crops if kept count < target
python tools/filter_scenes/build_scene_pool.py fill --target 100

# 5. Generate manifest (skips rejected scenes)
python generate_manifest.py
```

## Layout

```
scenes/
  core/sign_71853/          # full imported map
  sign_71853_cw0/           # crop around crossing #0
  sign_71853_cw1/           # crop around crossing #1
  scene_selection.json      # review verdicts
  _rejected/                # after review --apply
```

Crop one core scene manually:

```bash
python tools/filter_scenes/crop_crosswalk_scene.py sign_71853
# larger bbox (default): geo crop, radius 150 m
python tools/filter_scenes/crop_crosswalk_scene.py sign_71853 --radius 200
# old tight junction-only crop:
python tools/filter_scenes/crop_crosswalk_scene.py sign_71853 --crop-mode junction --radius 80
```

**Crop modes**

- `geo` (default) — square geo boundary around the crossing center. Produces much larger maps than junction-only crop.
- `junction` — keeps only junction arms up to `--radius` meters (often small when arms are short).

Default `--radius` is **150 m**. Increase it (e.g. `--radius 200`) for even larger scenes. Re-crop existing scenes with `--overwrite` after changing defaults.

`crop_junction_scene.py` is a backward-compatible alias for `crop_crosswalk_scene.py`.
