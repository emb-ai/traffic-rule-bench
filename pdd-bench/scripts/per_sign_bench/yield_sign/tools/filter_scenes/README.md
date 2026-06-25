The sequence of commands that are needed to run the scene pool workflow:

1. Import qualifying catalog scenes into scenes/core/ (3/4-arm junction check built in)
```
# Import enough core maps (e.g. 25 cores × up to 5 junctions ≈ 125 candidates)
python tools/filter_scenes/import_catalog_scenes.py --limit 30
```

2. Build junction scene pool up to 100 candidates, then review
```
# Crop until >= 100 junction scenes exist
python tools/filter_scenes/build_scene_pool.py crop --target 100

# Review keep/reject in browser
python tools/filter_scenes/review_junction_scenes.py

# After review: add more cores if kept < 100, then review again
python tools/filter_scenes/build_scene_pool.py fill --target 100

# Check progress anytime
python tools/filter_scenes/build_scene_pool.py status --target 100
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
