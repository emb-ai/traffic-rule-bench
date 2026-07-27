Scene pool workflow for no-turn signs (3.18.1 / 3.18.2).

Per-sign layout under ``no_turn_signs/scenes/``:

```
scenes/
  3_18_1/
    core/           # imported catalog OSM cores
    sign_*_j*/      # dual-path crops
  3_18_2/
```

Catalog source: ``pdd-bench/scenes/<pdd_code>`` (e.g. ``3.18.1``).

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.18.1 --arms 4 3 --limit 50 --no-simulation
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.18.2 --arms 4 3 --limit 50 --no-simulation
```

### Dual-path crop

Roles (approach kept only if forbidden + allowed first exits both exist):

* 3.18.1: baseline ``r``, compliant ``s``/``l`` → ``scenes/3_18_1/``
* 3.18.2: baseline ``l``, compliant ``s``/``r`` → ``scenes/3_18_2/``

X (4-arm) and T (3-arm) junctions are both eligible. On a T, e.g. 3.18.1 is
skipped on approaches that have no right-turn connection.

```bash
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.18.1 --limit 40 --overwrite
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.18.2 --limit 40 --overwrite
```

Each crop stores ``pdd_code``, ``forbidden_dir``, and a ``dual_path`` block in
``meta.json``.

```bash
python generate_manifest.py sign.pdd_code=3.18.1
python tools/filter_scenes/build_scene_pool.py crop --pdd-code 3.18.1 --target 20
python tools/filter_scenes/review_junction_scenes.py --pdd-code 3.18.1 --apply
```
