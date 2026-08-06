# Legacy scene pool (catalog / Overpass)

Do **not** use for new 2.4/2.1 pools. Prefer
[`../materialize_scenes.py`](../materialize_scenes.py) +
[`../review_scenes.py`](../review_scenes.py).

These scripts assumed sign-catalog CSV → Overpass/OSM fragment → junction crop
into a local `scenes/` tree.

```bash
# Historical workflow only
python build_scenes/legacy/import_catalog_scenes.py --limit 30
python build_scenes/legacy/build_scene_pool.py crop --target 100
python build_scenes/review_scenes.py
python build_scenes/legacy/build_scene_pool.py fill --target 100
```
