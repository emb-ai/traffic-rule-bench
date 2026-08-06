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

## Flow (2.4 yield)

Prereq: `moscow_junctions` has `nets/moscow.net.xml`, `index/junctions.jsonl`,
and `splits/sign_allocations.json` (see `../moscow_junctions/README.md`).

```bash
cd traffic-rule-bench/pdd-bench/scripts/per_sign_bench/priority_bench

# 1) Link allocated 2.4 maps (train+test) into data/yield/scenes/
python build_scenes/materialize_scenes.py --sign 2.4

# 2) Review keep/reject
python build_scenes/review_scenes.py
# default --scenes-dir = data/yield/scenes

# 3) Apply rejects → data/yield/scenes/_rejected/
python build_scenes/review_scenes.py --apply

# 4) Manifest / bench
python generate_manifest.py sign=yield
```

### Materialize flags

| Flag | Meaning |
|------|---------|
| `--split train\|test\|all` | Which half of the allocation (default `all`) |
| `--mode symlink\|copy` | Symlink into sign pool (default) or full copy |
| `--crop-missing` / `--no-crop-missing` | Crop from city net if missing under `moscow_junctions/scenes` (default: crop) |
| `--force-preview` | Rebuild `custom_cropped.png` for the review UI |

Pool bookkeeping: `data/yield/scenes/moscow_pool.json`.

## Relation to moscow_junctions

```
moscow_junctions/scenes/{T,X,O}/junc_*   ← city harvest (shared)
        │  allocate (splits/sign_allocations.json)
        ▼
build_scenes/materialize_scenes.py --sign 2.4
        ▼
data/yield/scenes/junc_*                 ← symlink/copy + review
        ▼
generate_manifest.py sign=yield
```

## Legacy

`legacy/` assumed **sign-catalog → Overpass fragment → crop**. Kept for
reference only. Old yield folders: `data/yield/_old_scenes`.
