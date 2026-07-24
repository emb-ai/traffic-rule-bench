# per_sign_bench

Run from `scripts/per_sign_bench/`. `PER_SIGN_COMPLIANT_NPC=1` enables rule-compliant NPCs.

### `per_sign_benchmark.py` – materialize scene manifests (per sign/backend)
```
python per_sign_benchmark.py --materialize --only-codes 3.24 \
    --output-dir benchmark_output/per_sign_n_v0_1
```

### `eval_pipeline.py` – run policies on a manifest + build report
```
PER_SIGN_COMPLIANT_NPC=1 python eval_pipeline.py \
    --policies idm --ego-variants default \
    --manifest benchmark_output/per_sign_n_v0_1/3_24/sumo/sumo_manifest.jsonl \
    --scenes-root ../../scenes --backends sumo --out-dir eval_out --save-gifs
```

### `run_benchmark.py` – run ONE policy on a manifest (`episodes_*.jsonl`)
```
PER_SIGN_COMPLIANT_NPC=1 python run_benchmark.py \
    --policy idm --ego-variant default --run-name idm_default \
    --manifest <manifest.jsonl> --scenes-root ../../scenes --backends sumo \
    --benchmark-output eval_out/benchmark --max-steps 800 --save-gifs
```

### `build_episode_metrics_csv.py` – episodes → per-episode CSV
```
python build_episode_metrics_csv.py \
    --episodes-root eval_out/benchmark/policy_eval --out eval_out/metrics_per_episode.csv
```

### `aggregate_episode_metrics.py` – CSV → per-baseline/sign/var rollups
```
python aggregate_episode_metrics.py \
    --csv eval_out/metrics_per_episode.csv --out-dir eval_out/agg
```

### `expert_replay.py` – record expert rollouts (`--aggregate` to summarize)
```
python expert_replay.py --manifest <manifest.jsonl> --backend sumo \
    --output-dir eval_out/expert --save-gifs
```

### `consolidate_replays.py` – merge per-scene `replay.json` → one jsonl/baseline
```
python consolidate_replays.py --runs-var <var_dir> --baseline idm_default --out <out.jsonl>
```

Subfolders: `bench/` (runner helpers), `sumo_space/` `factorized_space/` `citymap_space/`
(per-backend materialization), `filtered_metrics/` (category reports). Sign 2.4 (Yield) has
its own flow — see `yield_README.md`.
