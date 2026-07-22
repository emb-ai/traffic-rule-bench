Scene pool workflow for one-way entry signs (5.7.1 / 5.7.2).

Per-sign layout under ``one_way_signs/scenes/``:

```
scenes/<slug>/core/     # imported catalog maps
scenes/<slug>/sign_*_j* # dual-path junction crops
```

Catalog source: ``pdd-bench/scenes/<pdd_code>`` (e.g. ``5.7.1``).

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 5.7.1 --arms 4 3 --limit 50 --no-simulation
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 5.7.2 --arms 4 3 --limit 50 --no-simulation
```

Dual-path roles (forbidden short / allowed long):

* 5.7.1: baseline ``l``, compliant ``s``/``r`` → ``scenes/5_7_1/``
* 5.7.2: baseline ``r``, compliant ``s``/``l`` → ``scenes/5_7_2/``

X (4-arm) and T (3-arm) junctions are both eligible. On a T, e.g. 5.7.1 is
kept only if a real left (forbidden) exit exists on the approach.

```bash
python tools/filter_scenes/crop_junction_scene.py --pdd-code 5.7.1 --limit 40 --overwrite
python tools/filter_scenes/crop_junction_scene.py --pdd-code 5.7.2 --limit 40 --overwrite
```

```bash
python generate_manifest.py sign.pdd_code=5.7.1
python tools/filter_scenes/build_scene_pool.py crop --pdd-code 5.7.1 --target 20
python tools/filter_scenes/review_junction_scenes.py --pdd-code 5.7.1 --apply
```
