The sequence of commands that are needed to run the scene pool workflow:

1. Import qualifying catalog scenes into scenes/core/ (3/4-arm junction check built in)
```
# Import enough core maps (e.g. 25 cores × up to 5 junctions ≈ 125 candidates)
python tools/filter_scenes/import_catalog_scenes.py --limit 30
```

2. Build junction scene pool up to 100 candidates, then review

Cropping runs the same manifest-viability checks as `generate_manifest.py` (junction layout,
aux lane length, routable ego/aux spawn scenarios). Invalid junctions are skipped before
review so you do not label scenes that would be dropped later.

```
# Crop until >= 100 manifest-viable junction scenes exist
python tools/filter_scenes/build_scene_pool.py crop --target 100

# Review keep/reject in browser
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
python tools/filter_scenes/import_catalog_scenes.py --limit 10 --arms 4 3
```

Check resulted GIFs after IDM running on scenes:
```
python tools/review_benchmark_gifs.py benchmark_output/2_5/2026-06-25_17-07-31
```