# priority_bench — unified 2.1 (main road) + 2.4 (yield)

Shared junction-priority evaluation bench. Sign-specific behavior lives in
`signs/` profiles; shared engine in `core/`.

## Layout

```
priority_bench/
├── core/                 # shared libs (layout, aux, augmentation, viability, …)
├── signs/                # SignProfile registry (main_road, yield)
├── configs/              # Hydra (configs/sign/{main_road,yield}.yaml)
├── data/
│   ├── main_road/{scenes,output}   # symlinks → former main_sign trees
│   └── yield/{scenes,output}       # symlinks → former yield_sign trees
├── tools/
├── generate_manifest.py
├── run_benchmark.py
└── eval_pipeline.py
```

## Commands

```bash
cd priority_bench

# Equal-priority / main road (2.1)
python generate_manifest.py sign=main_road

# Yield (2.4)
python generate_manifest.py sign=yield

# Eval (row already carries pdd_code / sign_type)
python eval_pipeline.py --policies idm --manifest <run_dir> --scenes-root data/yield/scenes
```

`main_sign/` and `yield_sign/` remain as thin compatibility shims.
