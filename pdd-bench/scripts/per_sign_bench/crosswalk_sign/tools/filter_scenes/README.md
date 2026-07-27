The sequence of commands that are needed to run the pedestrian crossing (5.19) scene pool workflow:

1. Import qualifying catalog scenes into scenes/core/ (SUMO `function="crossing"` filter required)
```
# Import core maps whose SUMO net contains pedestrian crossings
python tools/filter_scenes/import_catalog_scenes.py --limit 40
```

2. Build crosswalk scene pool up to 100 candidates, then review

For each core map, `crop_crosswalk_scene.py` finds SUMO pedestrian crossings with at
least one vehicle approach lane, ranks them by approach length, and crops a **100 m**
geo square around each crossing center. **Each cropped scene keeps exactly one target
crossing** — all other pedestrian crossings and foreign walking areas are removed from
the net so only a single active crosswalk remains for pedestrians and verification.

Cropping runs the same manifest-viability checks as `generate_manifest.py` (crosswalk
layout, routable ego/aux spawn scenarios). Invalid crossings are skipped before
review so you do not label scenes that would be dropped later.

```
# Crop until >= 100 manifest-viable crosswalk scenes exist
python tools/filter_scenes/build_scene_pool.py crop --target 100

# Review keep/reject in browser (crosswalk crops only)
python tools/filter_scenes/review_junction_scenes.py

# After review, add more crops from unused core maps if kept < target:
python tools/filter_scenes/build_scene_pool.py fill --target 100

# For initial bulk growth (no review yet), prefer crop — it loops until target candidates:
python tools/filter_scenes/build_scene_pool.py crop --target 100

# Check progress anytime (shows manifest-viable count among candidates)
python tools/filter_scenes/build_scene_pool.py status --target 100
```

To disable manifest filtering (old behavior): add `--no-require-manifest-viable` to crop/fill.

Analyze drop reasons on existing scenes:
```
python tools/analyze_manifest_drops.py
```

3. Optionally move rejected scenes aside
```
python tools/filter_scenes/review_junction_scenes.py --apply
```

and check how many scenes are generated in the result:
```
ls -1d scenes/sign_*_cw*/ 2>/dev/null | wc -l
```

Each core map can produce up to `--max-crosswalks` cropped scenes (default 8), one per
ranked crossing: `sign_<id>_cw0`, `sign_<id>_cw1`, … Picks and manifest status are
recorded in `scenes/core/sign_<id>/crosswalks.json`.


4. Generate manifest (rejected scenes in scene_selection.json are skipped automatically)
```
python generate_manifest.py
```

### Scene layout

```
scenes/
  core/sign_71853/          # full imported catalog map
    crosswalks.json         # ranked crossing picks + manifest viability
  sign_71853_cw0/           # crop around crossing #0 (single active crossing only)
  sign_71853_cw1/           # crop around crossing #1
  scene_selection.json      # review verdicts (keep / reject / pending)
  _rejected/                # after review --apply
```

### Lower-level commands (optional)

Crop one core scene manually:
```
python tools/filter_scenes/crop_crosswalk_scene.py sign_71853
```

Crop all uncropped cores:
```
python tools/filter_scenes/crop_crosswalk_scene.py
```

Larger or smaller bbox (default radius 100 m):
```
python tools/filter_scenes/crop_crosswalk_scene.py sign_71853 --radius 150
```

Legacy tight junction-only crop (short arms only):
```
python tools/filter_scenes/crop_crosswalk_scene.py sign_71853 --crop-mode junction --radius 80
```

Re-crop existing scenes after changing defaults:
```
python tools/filter_scenes/crop_crosswalk_scene.py sign_71853 --overwrite
```

Import only:
```
python tools/filter_scenes/import_catalog_scenes.py --limit 10
```

Check resulted GIFs after IDM running on scenes:
```
python tools/review_benchmark_gifs.py benchmark_output/5_19/2026-07-02_16-59-49
```

`crop_junction_scene.py` is a backward-compatible alias for `crop_crosswalk_scene.py`.
