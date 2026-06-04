## Running instructions

```
conda activate zinkovich-plant2
```

### Syntetic scenes

1. Scene generation (```run_benchmark.py``` – visualization, main_road_traffic.py – for traffic spawn)
```
python yield_sign/generate_synthetic_scenes.py --n-scenes 5 --save-gifs
```

2. Baseline evaulation & Metrics computation
```
python yield_sign/eval_pipeline.py \
    --policies idm \
    --manifest yield_sign/benchmark_output/pgmaps/2_4/2026-05-29_16-44-48/pgmap_materialized.jsonl \
    --backends pgmap \
    --out-dir yield_sign/benchmark_output/pgmaps/2_4/2026-05-29_16-44-48
```

### Real-world maps




## Constructing scenes

1. Scene loading from OSM file and truncating it 
```
python build_single_sign_scene.py savvinskaya_3 --radius 100
```

2. Policy execution (IDM)
```
python run_simulation.py savvinskaya_3
```

 2*. Obtain the map view
```
python render_static_map.py savvinskaya_3
```