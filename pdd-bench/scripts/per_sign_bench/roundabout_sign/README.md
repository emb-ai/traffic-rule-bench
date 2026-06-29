# Roundabout Sign (4.3) Benchmark

PDD **4.3 Круговое движение** — ego approaches a traffic circle on a **secondary spoke** and must **yield to vehicles on the ring** (main road).

## Scene pipeline

1. **Catalog** — OSM scenes in `pdd-bench/scenes/4.3/` (`build_sign_scenes_from_osm_async.py --sign-types "4.3"`).
2. **Import** — `tools/filter_scenes/import_catalog_scenes.py` copies qualifying maps into `scenes/core/` (SUMO net must contain a ``<roundabout>`` block reachable from the sign road).
3. **Crop** — `tools/filter_scenes/crop_junction_scene.py` keeps only the traffic circle + spokes, emitting one scene per attached road (`scenes/sign_<id>_rb_s00/`, `_rb_s01/`, …) with the 4.3 sign on that spoke.
4. **Manifest** — `generate_manifest.py` (Hydra config in `config/config.yaml`).
5. **Run** — `run_benchmark.py` places **4.3** icon on the ego spoke and an invisible **RoundaboutYieldSign** tracker (ring = main road).

## Layout rules

| Role | Roads |
|------|--------|
| Ego (secondary) | Spoke approaching the circle (catalog `road_id` chain) |
| Aux (main) | Traffic circle ring edges |
| Sign | 4.3 on ego approach; yield verified against ring lanes |

## Quick start

```bash
cd pdd-bench/scripts/per_sign_bench/roundabout_sign

# Import from catalog
python tools/filter_scenes/import_catalog_scenes.py --limit 5

# Crop roundabouts
python tools/filter_scenes/crop_junction_scene.py --limit 5

# Build manifest + optional GIFs
python generate_manifest.py gif.enabled=true
```
