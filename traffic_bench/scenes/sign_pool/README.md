# sign_pool — per-sign scene pool (materialize → viability reject → review)

Pulls allocated maps from `traffic_bench/scenes/map_pool` into `data/scenes/<sign>/`.
Debug helpers: repo-root `tools/`.

## Layout

```
build_scenes/
├── materialize_scenes.py      # allocations → data/<sign>/scenes/
├── reject_unusable_scenes.py  # viability reject → _rejected/ (+ optional --refill)
├── review_scenes.py           # browser keep/reject + --apply
├── README.md
└── legacy/                    # old catalog / Overpass flow (do not use)
    ├── import_catalog_scenes.py
    ├── crop_junction_scene.py
    └── build_scene_pool.py
```

## Flow (example: 2.4 yield; same for 2.1 / 2.3 / 2.5 / 3.2 / 4.3)

Prereq: `moscow_scenes` has `nets/moscow.net.xml`, `index/junctions.jsonl`,
and `splits/sign_allocations.json` (see `../moscow_scenes/README.md`).
Map quotas live in `moscow_scenes/splits/signs.yaml` (`n_train` / `n_test`).

```bash
cd traffic-rule-bench/traffic-bench/sign_bench

# 1) Link allocated maps (train+test) into data/<sign>/scenes/
python build_scenes/materialize_scenes.py --sign 2.4   # or 2.1 / 2.3 / 2.5 / 3.2 / 4.3
# Dual-path: 5.7 / 4.1 / 3.18 / 3.1 (slot-balanced in allocate)
#   python build_scenes/materialize_scenes.py --sign 3.18.1
#   python build_scenes/materialize_scenes.py --sign 3.1
# Roundabout: moscow scenes/O/ → data/roundabout/scenes

# 2) Drop maps that cannot produce scenarios; refill to signs.yaml quotas
python build_scenes/reject_unusable_scenes.py --sign 4.3 --apply --refill --loop
# (works for other signs too; for 2.4 use --sign 2.4)

# 3) Optional visual review keep/reject
python build_scenes/review_scenes.py --scenes-dir data/yield/scenes
# stop:  --scenes-dir data/stop/scenes
# Optional: mark every pending scene as keep, then reject only the bad ones
python build_scenes/review_scenes.py --scenes-dir data/stop/scenes --mark-all-keep
# (or click "Keep all pending" in the UI)

# 4) Apply visual rejects → data/<sign>/scenes/_rejected/
python build_scenes/review_scenes.py --scenes-dir data/yield/scenes --apply

# 5) Top up again if review removed maps
python build_scenes/materialize_scenes.py --sign 2.4 --refill
# Or re-run reject_unusable_scenes --apply --refill --loop

# 6) Manifest / bench (filter train or test via Hydra) — no scene pool edits
python generate_manifest.py sign=yield paths.split=train
python generate_manifest.py sign=roundabout gif.enabled=false
python generate_manifest.py sign=blocked_road paths.split=train
python generate_manifest.py sign=one_way_right paths.split=train
python generate_manifest.py sign=one_way_left paths.split=train
python generate_manifest.py sign=direction_straight paths.split=train
python generate_manifest.py sign=direction_right paths.split=train
python generate_manifest.py sign=direction_left paths.split=train
python generate_manifest.py sign=direction_straight_right paths.split=train
python generate_manifest.py sign=direction_straight_left paths.split=train
python generate_manifest.py sign=direction_left_right paths.split=train
python generate_manifest.py sign=no_turn_right paths.split=train
python generate_manifest.py sign=no_turn_left paths.split=train
python generate_manifest.py sign=no_entry paths.split=train
```

### reject_unusable_scenes flags


| Flag           | Meaning                                                    |
| -------------- | ---------------------------------------------------------- |
| `--dry-run`    | Print which live scenes would be rejected                  |
| `--apply`      | Move rejects to `scenes/_rejected/`                        |
| `--refill`     | Call `materialize_scenes.py --refill` after apply          |
| `--loop`       | Repeat reject→apply→refill until live pool is fully viable |
| `--audit PATH` | Write rejected rows as JSONL                               |


Reasons are stored in `scenes/scene_selection.json` under `reject_reasons`.

### Materialize flags


| Flag                                   | Meaning                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `--split train|test|all`               | Which half of the allocation (default `all`)                               |
| `--refill`                             | Add unused maps until kept train/test hit `signs.yaml` quotas              |
| `--mode symlink|copy`                  | Symlink into sign pool (default) or full copy                              |
| `--crop-missing` / `--no-crop-missing` | Crop from city net if missing under `moscow_scenes/scenes` (default: crop) |
| `--force-preview`                      | Rebuild `custom_cropped.png` for the review UI                             |


Pool bookkeeping: `data/yield/scenes/moscow_pool.json` (per-scene `split`).
Keep train+test in one `scenes/` folder; choose half at manifest time with
`paths.split`.

## Relation to moscow_scenes

```
moscow_scenes/scenes/{T,X,O}/…   ← city harvest (shared)
        │  allocate (splits/sign_allocations.json)
        ▼
build_scenes/materialize_scenes.py --sign 4.3
        ▼
build_scenes/reject_unusable_scenes.py --apply --refill --loop
        ▼
data/roundabout/scenes/…            ← viable pool (+ _rejected/)
        ▼
generate_manifest.py sign=roundabout   ← read-only w.r.t. scenes/
        ▼
output/<ts>/{real_manifest.jsonl, repro/}
```

## Legacy

`legacy/` assumed **sign-catalog → Overpass fragment → crop**. Kept for
reference only. Old yield folders: `data/yield/_old_scenes`.



Useful command:

```markdown
ls -1d sign*/ 2>/dev/null | wc -l
```

