## Running instructions

1. Scene generation (```run_benchmark.py``` – visualization, main_road_traffic.py – for traffic spawn)
```
python yield_generate_fixed_scenes.py --n-scenes 5 --save-gifs
```

2. Baseline evaulation 
```
python yield_prepare_metrics.py \
    --manifest benchmark_output/fixed/2_4/2026-05-28_17-37-38/pgmap_materialized.jsonl \
    --out-dir benchmark_output/fixed/2_4/2026-05-28_17-37-38/ \
    --rerun-failed --emit-replay-sidecar
```

3. Metrics computation
```
python yield_sign/eval_pipeline.py \
    --policies idm \
    --manifest yield_sign/benchmark_output/pgmaps/2_4/2026-05-29_16-44-48/pgmap_materialized.jsonl \
    --backends pgmap \
    --out-dir yield_sign/benchmark_output/pgmaps/2_4/2026-05-29_16-44-48
```
