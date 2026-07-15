The sequence of commands for the direction-sign (4.1.x) scene pool workflow.

Default catalog is ``pdd-bench/scenes/4.1.1``. Override when working on another
family member:

```
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.2 --limit 10
```

### 4.1.1 dual-path crop (variant 1)

1. Import cores (4-arm preferred):

```
python tools/filter_scenes/import_catalog_scenes.py --arms 4 --limit 30
```

2. Select + crop scenes where the **same destination** is reachable by a
   shorter turn (l/r) and a longer straight path through an X junction. Crop
   bbox = union of both paths + margin (not junction-stub-only):

```
python tools/filter_scenes/crop_junction_scene.py --limit 10
python tools/filter_scenes/crop_junction_scene.py --dry-run --limit 20
python tools/filter_scenes/crop_junction_scene.py sign_72915 --overwrite --min-gain 20 --margin 40
```

Each written scene stores canonical ``road_id`` (spawn), ``destination_edge_id``,
and a ``dual_path`` block (straight + turn edge lists) in ``meta.json``. The
preview ``custom_cropped.png`` overlays both paths (blue = straight / longer,
orange = turn / shorter) plus spawn and destination markers.

3. Manifest / eval (reuses crop-time spawn/dest — does not rediscover routes):

```
python generate_manifest.py
```

### Notes

Not every OSM extract has a reconverging turn+straight pair. Cores without a
dual-path hit are skipped (see ``junctions.json`` / console output).
