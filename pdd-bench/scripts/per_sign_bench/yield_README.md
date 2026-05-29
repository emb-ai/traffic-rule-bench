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
python yield_aggregate_metrics.py \
    --runs-dir benchmark_output/fixed/2_4/2026-05-28_17-37-38 \
    --out-md benchmark_output/fixed/2_4/2026-05-28_17-37-38/report_cumulative.md

bash yield_run_metrics.sh --runs-dir benchmark_output/fixed/2_4/2026-05-28_17-37-38 --out-dir benchmark_output/fixed/2_4/2026-05-28_17-37-38/reports
```
