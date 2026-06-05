# citymap_space/

CityMap (looped grid, real alternative routes) scene generation. Run from
`scripts/per_sign_bench/`.

### `citymap_pipeline.py` – build maps → analyse → enumerate → materialize (run as module)
```
python -m citymap_space.citymap_pipeline --n-maps 40 --output-dir out
```
Or via the orchestrator: `per_sign_benchmark.py --materialize --backends citymap --citymap-maps 40`.

Library modules (no CLI): `citymap_env.py`, `citymap_analyzer.py`,
`citymap_scene_enumerator.py`, `citymap_runner.py`.
