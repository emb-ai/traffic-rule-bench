# build_scenes — sign scene pool (materialize → review)

First-class pipeline stage for `priority_bench`. Pulls allocated junctions from
the shared **`moscow_junctions`** harvest into `data/<sign>/scenes/`, then
review keep/reject.

Ad-hoc debug helpers stay under [`../tools/`](../tools/).

## Layout

```
build_scenes/
├── materialize_scenes.py   # allocations → data/<sign>/scenes/
├── review_scenes.py      # browser keep/reject + --apply
├── README.md
└── legacy/                        # old catalog / Overpass flow (do not use)
    ├── import_catalog_scenes.py
    ├── crop_junction_scene.py
    └── build_scene_pool.py
```

## Flow (example: 2.4 yield; same for 2.1 / 2.5)

Prereq: `moscow_junctions` has `nets/moscow.net.xml`, `index/junctions.jsonl`,
and `splits/sign_allocations.json` (see `../moscow_junctions/README.md`).
Map quotas live in `moscow_junctions/splits/signs.yaml` (`n_train` / `n_test`).

```bash
cd traffic-rule-bench/pdd-bench/scripts/per_sign_bench/priority_bench

# 1) Link allocated maps (train+test) into data/<sign>/scenes/
python build_scenes/materialize_scenes.py --sign 2.4   # or 2.1 / 2.5

# 2) Review keep/reject
python build_scenes/review_scenes.py --scenes-dir data/yield/scenes
# stop:  --scenes-dir data/stop/scenes
# Optional: mark every pending scene as keep, then reject only the bad ones
python build_scenes/review_scenes.py --scenes-dir data/stop/scenes --mark-all-keep
# (or click "Keep all pending" in the UI)

# 3) Apply rejects → data/<sign>/scenes/_rejected/
python build_scenes/review_scenes.py --scenes-dir data/yield/scenes --apply

# 4) Top up kept counts to signs.yaml quotas (new scenes = pending)
python build_scenes/materialize_scenes.py --sign 2.4 --refill
# Repeat review → --apply → --refill until quotas met or pool exhausted

# 5) Manifest / bench (filter train or test via Hydra)
python generate_manifest.py sign=yield paths.split=train
python generate_manifest.py sign=stop paths.split=test
```

### Materialize flags

| Flag | Meaning |
|------|---------|
| `--split train\|test\|all` | Which half of the allocation (default `all`) |
| `--refill` | Add unused maps until kept train/test hit `signs.yaml` quotas |
| `--mode symlink\|copy` | Symlink into sign pool (default) or full copy |
| `--crop-missing` / `--no-crop-missing` | Crop from city net if missing under `moscow_junctions/scenes` (default: crop) |
| `--force-preview` | Rebuild `custom_cropped.png` for the review UI |

Pool bookkeeping: `data/yield/scenes/moscow_pool.json` (per-scene `split`).
Keep train+test in one `scenes/` folder; choose half at manifest time with
`paths.split`.

## Relation to moscow_junctions

```
moscow_junctions/scenes/{T,X,O}/junc_*   ← city harvest (shared)
        │  allocate (splits/sign_allocations.json)
        ▼
build_scenes/materialize_scenes.py --sign 2.4
        ▼
data/yield/scenes/junc_*                 ← symlink/copy + review (+ --refill)
        ▼
generate_manifest.py sign=yield paths.split=train|test|all
        ▼
output/<ts>/{real_manifest.jsonl, repro/}
```

## Legacy

`legacy/` assumed **sign-catalog → Overpass fragment → crop**. Kept for
reference only. Old yield folders: `data/yield/_old_scenes`.
