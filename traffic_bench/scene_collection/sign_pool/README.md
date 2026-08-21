# sign_pool — per-sign maps (materialize → viability reject → review)

Pulls allocated maps from `map_pool` into `data/scenes/<sign>/`.
`--sign` is the **eval profile id** (same as `generate_manifest.py sign=...`),
not the PDD code. Debug helpers: repo-root `tools/`.

## Layout

```
sign_pool/
├── materialize_scenes.py      # allocations → data/scenes/<sign>/
├── reject_unusable_scenes.py  # viability reject → _rejected/ (+ optional --refill)
├── review_scenes.py           # browser keep/reject + --apply
└── README.md
```

## Flow (example: yield / PDD 2.4)

Prereq: `map_pool` has `nets/moscow.net.xml`, `index/junctions.jsonl`,
and `splits/sign_allocations.json`. Quotas live in `map_pool/splits/signs.yaml`.
Eval ids for every harvested sign: `../README.md`.

```bash
cd traffic-rule-bench

# 1) Link allocated maps (train+test) into data/scenes/<sign>/
python traffic_bench/scene_collection/sign_pool/materialize_scenes.py --sign yield
# Dual-path: no_entry / no_turn_* / direction_* / one_way_*
#   python .../materialize_scenes.py --sign no_turn_right
# Segment (speed / detour):
#   python .../materialize_scenes.py --sign speed_limit
#   python .../materialize_scenes.py --sign detour_right
# crosswalk: injects zebra into data/scenes/crosswalk/
#   python .../materialize_scenes.py --sign crosswalk

# 2) Drop maps that cannot produce scenarios; refill to signs.yaml quotas
python traffic_bench/scene_collection/sign_pool/reject_unusable_scenes.py \
  --sign roundabout --apply --refill --loop

# 3) Optional visual review
python traffic_bench/scene_collection/sign_pool/review_scenes.py --scenes-dir data/scenes/yield
python traffic_bench/scene_collection/sign_pool/review_scenes.py --scenes-dir data/scenes/yield --apply

# 4) Top up again if review removed maps
python traffic_bench/scene_collection/sign_pool/materialize_scenes.py --sign yield --refill
```

Manifest / bench (no pool edits):

```bash
python traffic_bench/eval/generate_manifest.py sign=yield paths.split=train
python traffic_bench/eval/generate_manifest.py sign=crosswalk paths.split=train
```

### reject_unusable_scenes flags

| Flag           | Meaning                                                    |
| -------------- | ---------------------------------------------------------- |
| `--dry-run`    | Print which live scenes would be rejected                  |
| `--apply`      | Move rejects to `scenes/_rejected/`                        |
| `--refill`     | Call `materialize_scenes.py --refill` after apply          |
| `--loop`       | Repeat reject→apply→refill until live pool is fully viable |
| `--audit PATH` | Write rejected rows as JSONL                               |

`crosswalk` derived maps live under `data/scenes/crosswalk/` (`prepare: crosswalk`
in `signs.yaml`). Junction / dual_path / plain segment maps are symlinked (or
copied) as-is. Detour signs use the same segment crop; obstacle-lane choice is
left to eval.
