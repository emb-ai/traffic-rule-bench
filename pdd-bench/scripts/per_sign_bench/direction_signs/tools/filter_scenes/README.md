The sequence of commands for the direction-sign (4.1.x) scene pool workflow.

Default catalog is ``pdd-bench/scenes/4.1.1``. Override the source when working on
another family member:

```
python tools/filter_scenes/import_catalog_scenes.py --source ../../../../scenes/4.1.2 --limit 10
```

1. Import qualifying catalog scenes into scenes/core/ (3/4-arm junction check)
```
python tools/filter_scenes/import_catalog_scenes.py --limit 30
```

2. Build junction scene pool, then review

```
python tools/filter_scenes/build_scene_pool.py crop --target 100
python tools/filter_scenes/review_junction_scenes.py
python tools/filter_scenes/build_scene_pool.py status --target 100
```

Direction-aware route filtering is not applied yet — crops use the shared
junction viability checks (layout + spawn geometry).

Analyze drops:
```
python tools/analyze_manifest_drops.py
```

3. Optionally move rejected scenes aside
```
python tools/filter_scenes/review_junction_scenes.py --apply
```

4. Generate manifest (default sign 4.1.1)
```
python generate_manifest.py
python generate_manifest.py sign.pdd_code=4.1.2 paths.output_base=benchmark_output/4_1_2
```

### Lower-level

```
python tools/filter_scenes/crop_junction_scene.py
python tools/filter_scenes/import_catalog_scenes.py --limit 10 --arms 4 3
python tools/review_benchmark_gifs.py benchmark_output/4_1_1/<timestamp>
```
