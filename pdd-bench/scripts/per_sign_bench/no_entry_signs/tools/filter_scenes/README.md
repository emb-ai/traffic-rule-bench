Scene pool workflow for no-entry signs (3.1 / 3.2).

Per-sign layout under ``no_entry_signs/scenes/``:

```
scenes/
  3_1/
    sign_*/          # imported catalog extracts
  3_2/
```

Catalog source: ``pdd-bench/scenes/<pdd_code>`` (e.g. ``3.1``, ``3.2``).
Scenes are already local OSM extracts — **no dual-path crop step**.

### Import only

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.1 --limit 40 --no-simulation
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.2 --limit 40 --no-simulation
```

`crop_junction_scene.py` is a stub (exit 0): cropping is not required.

### Manifest + review

```bash
python generate_manifest.py sign.pdd_code=3.1
python tools/filter_scenes/review_junction_scenes.py --pdd-code 3.1 --apply
```
