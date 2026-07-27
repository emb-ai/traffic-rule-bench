Scene pool workflow for no-entry signs (3.1 / 3.2).

Per-sign layout under ``no_entry_signs/scenes/``:

```
scenes/
  3_1/
    core/           # imported catalog OSM cores
    sign_*_j*/      # ranked multi-arm junction crops
  3_2/
```

Catalog source: ``pdd-bench/scenes/<pdd_code>`` (e.g. ``3.1``, ``3.2``).

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.1 --arms 4 3 --limit 50 --no-simulation
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.2 --arms 4 3 --limit 50 --no-simulation
```

### Ranked multi-arm crop

Import stores full maps under ``scenes/<slug>/core/``. Crop picks up to
``--max-junctions`` 3-/4-arm junctions per core (4-arm preferred, then by lane
count) via ``find_ranked_intersection_junctions`` and writes siblings
``sign_*_j*`` under ``scenes/<slug>/``. Crop meta drops catalog
``distance_from_start``, sets ``sign_spawn_distance=30``, and fills ego
``road_id`` + destination.

```bash
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.1 --limit 40 --overwrite
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.2 --limit 40 --overwrite
```

```bash
python generate_manifest.py sign.pdd_code=3.1
python tools/filter_scenes/build_scene_pool.py crop --pdd-code 3.1 --target 20
python tools/filter_scenes/review_junction_scenes.py --pdd-code 3.1 --apply
```
