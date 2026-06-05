# sumo_space/

SUMO scene materialization. Usually orchestrated by `per_sign_benchmark.py --materialize`
(backend `sumo`); run from `scripts/per_sign_bench/`. Direct calls:

### `sumo_scene_enumerator.py` – walk `scenes/<code>/sign_*/meta.json`
```
python sumo_space/sumo_scene_enumerator.py --scenes-root ../../scenes
```

### `sumo_catalog.py` – enumerate (scene, v_idx, var_idx) tuples
```
python sumo_space/sumo_catalog.py --scenes-root ../../scenes --output catalog.jsonl
```

### `sumo_pipeline.py` – catalog → materialize → `sumo_manifest.jsonl`
```
python sumo_space/sumo_pipeline.py --catalog catalog.jsonl \
    --scenes-root ../../scenes --output-dir out
```

### `sumo_zone_pairing.py` – pair start/end zone scenes
```
python sumo_space/sumo_zone_pairing.py --scenes-root ../../scenes \
    --out-root out --manifest paired.jsonl
```
