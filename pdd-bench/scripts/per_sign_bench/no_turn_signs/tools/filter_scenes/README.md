Scene pool workflow for no-turn signs (3.18.1 / 3.18.2 / 3.19).

Per-sign layout under ``no_turn_signs/scenes/``:

```
scenes/
  3_18_1/
    core/           # imported catalog OSM cores
    sign_*_j*/      # dual-path crops
  3_18_2/
  3_19/
```

Catalog source: ``pdd-bench/scenes/<pdd_code>`` (e.g. ``3.18.1``).

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.18.1 --arms 4 --limit 30 --no-simulation
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.18.2 --arms 4 --limit 30 --no-simulation
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.19 --arms 4 --limit 40 --no-simulation
```

### Dual-path crop

Roles:

* 3.18.1: baseline ``r``, compliant ``s``/``l`` → ``scenes/3_18_1/``
* 3.18.2: baseline ``l``, compliant ``s``/``r`` → ``scenes/3_18_2/``
* 3.19: baseline ``t``, compliant ``s``/``r``/``l`` → ``scenes/3_19/``

```bash
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.18.1 --limit 20 --overwrite
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.18.2 --limit 20 --overwrite
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.19 --limit 20 --overwrite
```

Each crop stores ``pdd_code``, ``forbidden_dir``, and a ``dual_path`` block in
``meta.json``.

```bash
python generate_manifest.py sign.pdd_code=3.18.1
python tools/filter_scenes/build_scene_pool.py crop --pdd-code 3.18.1 --target 20
python tools/filter_scenes/review_junction_scenes.py --pdd-code 3.18.1 --apply
```
