# tools — ad-hoc debug helpers

Not part of the main pipeline. Scene pools: `traffic_bench/scene_collection/`.

| Script | Role |
|--------|------|
| `check_map_overlap.py` | Geometric / road-graph overlap of maps inside a split, within one sign and across signs |
| `fetch_hf_scenes.py` | Download the official scenes from HF (catalog only, aliases, `moscow_pool.json`) |
| `run_simulation.py` | One-off MetaDrive sim / GIF |
| `review_benchmark_gifs.py` | Browse GIFs after a run |
| `render_map.py` | CLI for a top-down PNG (library: `traffic_bench.scene_collection.preview`) |
| `build_scene.py` | OSM → SUMO for a single hand-built scene |
| `vis_sumo_map_traffic_sign.py` | Visualize signs on a SUMO map |
