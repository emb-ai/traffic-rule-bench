The sequence of commands that are needed to run the roundabout (4.3) scene pool workflow:

1. Import qualifying catalog scenes into scenes/core/ (SUMO ``<roundabout>`` filter required)
```
# Import core maps with SUMO roundabouts reachable from the sign road
python tools/filter_scenes/import_catalog_scenes.py --limit 30
```

2. Build roundabout scene pool up to 100 candidates, then review

Cropping runs the same manifest-viability checks as `generate_manifest.py` (O layout,
aux spawn on long ring arms or compact entry zones near ego, routable ego/aux scenarios).
Invalid roundabouts are skipped before review so you do not label scenes that would be
dropped later.

```
# Crop until >= 100 manifest-viable roundabout scenes exist
python tools/filter_scenes/build_scene_pool.py crop --target 100

# Re-try cores that failed earlier (duplicate/crop bugs) after fixes:
python tools/filter_scenes/build_scene_pool.py retry --target 150

# Add scenes on other spokes for roundabouts that already have sign_*_rb (reach 100+):
python tools/filter_scenes/build_scene_pool.py expand-spokes --target 100

# Review keep/reject in browser (roundabout crops only)
python tools/filter_scenes/review_junction_scenes.py

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
ls -1d sign*/ 2>/dev/null | wc -l
```

By default each core map produces one cropped scene (`sign_<id>_rb/`). Use
`--per-spoke` on crop/build_scene_pool for the legacy per-spoke layout (`_rb_s00`, …).

Duplicate roundabouts (same SUMO `<roundabout>` OSM node set) are tracked in
`scenes/roundabout_fingerprints.json` and skipped on import/crop. Rebuild after
manual edits:

```
python tools/filter_scenes/rebuild_roundabout_fingerprints.py
```


4. Generate manifest (rejected scenes in scene_selection.json are skipped automatically)
```
python generate_manifest.py
```

### Lower-level commands (optional)

Crop all uncropped cores manually:
```
python tools/filter_scenes/crop_junction_scene.py
```

Import only:
```
python tools/filter_scenes/import_catalog_scenes.py --limit 10
```

Check resulted GIFs after IDM running on scenes:
```
python tools/review_benchmark_gifs.py benchmark_output/4_3/2026-06-25_17-07-31
```
