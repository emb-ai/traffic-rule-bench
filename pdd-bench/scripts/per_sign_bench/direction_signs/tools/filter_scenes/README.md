The sequence of commands for the direction-sign (4.1.x) scene pool workflow.

Per-sign layout under ``direction_signs/scenes/``:

```
scenes/
├── 4_1_1/core/ … crops …
└── 4_1_2/core/ … crops …
```

Catalog source remains ``pdd-bench/scenes/<pdd_code>`` (e.g. ``4.1.2``).

```
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.2 --limit 10
# writes → scenes/4_1_2/core/
```

### Dual-path crop (4.1.1 / 4.1.2 / 4.1.3)

1. Import cores (4-arm preferred):

```
python tools/filter_scenes/import_catalog_scenes.py --arms 4 --limit 30
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.2 --arms 4 --limit 30
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.3 --arms 4 --limit 30
```

2. Select + crop scenes where the **same destination** is reachable by a
   shorter **baseline** (forbidden) first exit and a longer **compliant**
   (allowed) path. Roles from ``--pdd-code``:

   * 4.1.1: baseline ``l``/``r``, compliant ``s`` → ``scenes/4_1_1/``
   * 4.1.2: baseline ``s``/``l``, compliant ``r`` → ``scenes/4_1_2/``
   * 4.1.3: baseline ``s``/``r``, compliant ``l`` → ``scenes/4_1_3/``

```
python tools/filter_scenes/crop_junction_scene.py --limit 10
python tools/filter_scenes/crop_junction_scene.py --pdd-code 4.1.2 --limit 10 --overwrite
python tools/filter_scenes/crop_junction_scene.py --dry-run --limit 20
python tools/filter_scenes/crop_junction_scene.py sign_72915 --overwrite --min-gain 20 --margin 40
```

Each written scene stores canonical ``road_id`` (spawn), ``destination_edge_id``,
``pdd_code``, and a ``dual_path`` block in ``meta.json``. The preview
``custom_cropped.png`` overlays both paths (blue = compliant / longer,
orange = baseline / shorter) plus spawn and destination markers.

3. Manifest / eval:

```
python generate_manifest.py
# → reads scenes/4_1_1/
python generate_manifest.py sign.pdd_code=4.1.2 paths.output_base=benchmark_output/4_1_2
# → reads scenes/4_1_2/
```

Pool builder:

```
python tools/filter_scenes/build_scene_pool.py crop --pdd-code 4.1.2 --target 20
```

and check how many scenes are generated in the result:
```
ls -1d sign*/ 2>/dev/null | wc -l
```

4. Optionally move rejected scenes aside
```
python tools/filter_scenes/review_junction_scenes.py --apply
```

### Notes

Not every OSM extract has a reconverging baseline+compliant pair. Cores without a
dual-path hit are skipped (see ``junctions.json`` / console output).
