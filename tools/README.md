# tools — ad-hoc debug helpers

Not part of the main pipeline. Scene pools: `traffic_bench/scene_collection/`.

| Script | Role |
|--------|------|
| `run_simulation.py` | One-off MetaDrive sim / GIF |
| `review_benchmark_gifs.py` | Browse GIFs after a run |
| `eval_progress.py` | Train-eval progress + ETA (`--watch`) for `run_signs_parallel.sh` |
| *(eval metrics)* | Baseline comparison plots: `python -m traffic_bench.eval metrics plot` |
| `render_map.py` | CLI for a top-down PNG (library: `traffic_bench.scene_collection.preview`) |
| `build_scene.py` | OSM → SUMO for a single hand-built scene |
| `vis_sumo_map_traffic_sign.py` | Visualize signs on a SUMO map |
