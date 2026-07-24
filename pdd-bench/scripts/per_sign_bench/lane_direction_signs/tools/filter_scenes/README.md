# Filter / crop workflow — PDD 5.15.1

Lane-direction board («Направления движения по полосам»). Run from
`lane_direction_signs/`:

```bash
cd pdd-bench/scripts/per_sign_bench/lane_direction_signs
```

Layout:

```
scenes/
└── 5_15_1/
    ├── core/                 # imported catalog maps
    ├── sign_*_j*/            # lane-change crops
    ├── scene_selection.json  # review keep/reject
    └── _rejected/            # after --apply
```

Catalog source: `pdd-bench/scenes/5.15.2` (same family; see
`lib/direction_sign_spec.py` → `catalog_subdir`). Default `--pdd-code` is
`5.15.1` → writes under `scenes/5_15_1/`.

## What a crop is

Not the 4.1.x “forbidden first exit vs longer compliant loop” dual-path.

Here:

1. Junction has ≥1 **multi-lane** approach (≥2 lanes).
2. Ego spawns on a **wrong** lane with **no** first-exit route to dest.
3. A **target** peer lane on the same approach can reach dest (prefer adjacent).
4. Preview `custom_cropped.png`:
   - per-lane allowed-direction arrows
   - **orange dashed** = wrong path (stay on spawn lane)
   - **blue** = correct path (lane-change → dest)

`meta.json` stores `road_id`, `spawn_lane_num`, `target_lane_num`,
`destination_*`, and `dual_path` with `kind: lane_change`
(`turn_*` = wrong, `straight_*` = correct).

At eval, injection adds an illegal connector from the spawn lane so baseline
`idm` can take the wrong exit; rule policies are expected to lane-change first.

## 1) Import cores

Keep scenes with a 3- and/or 4-arm junction **and** a multi-lane approach:

```bash
python tools/filter_scenes/import_catalog_scenes.py --arms 4 3 --limit 30
python tools/filter_scenes/import_catalog_scenes.py --arms 4 --limit 20
python tools/filter_scenes/import_catalog_scenes.py sign_79054 --no-simulation
```

Cores land in `scenes/5_15_1/core/`.

## 2) Crop lane-change scenes

```bash
python tools/filter_scenes/crop_junction_scene.py --limit 10
python tools/filter_scenes/crop_junction_scene.py --overwrite --min-gain 0
python tools/filter_scenes/crop_junction_scene.py --dry-run --limit 20
python tools/filter_scenes/crop_junction_scene.py sign_71895 --overwrite
```

Useful flags:

- `--min-ego-lane-m` (default 21) — approach must fit spawn ≥20 m before junction
- `--min-gain` (default 0) — optional wrong-spur vs correct-path length gap
- `--max-scenarios` (default 5) — crops per core
- `--skip-metadrive-check` — skip MetaDrive routability filter on target→dest

Dedup: at most one crop per `(junction_id, ego approach)` across all cores.

## 3) Pool builder + review

```bash
python tools/filter_scenes/build_scene_pool.py crop --target 20
python tools/filter_scenes/build_scene_pool.py status --target 20
python tools/filter_scenes/review_junction_scenes.py
python tools/filter_scenes/review_junction_scenes.py --apply   # → scenes/5_15_1/_rejected/
python tools/filter_scenes/build_scene_pool.py fill --target 20
```

Count crops:

```bash
ls -1d scenes/5_15_1/sign*/ 2>/dev/null | wc -l
```

## 4) Manifest / eval

```bash
python generate_manifest.py sign.pdd_code=5.15.1
# optional GIF smoke:
python generate_manifest.py gif.enabled=true gif.policy=carl_rule

python eval_pipeline.py \
    --policies idm modified_idm carl_rule \
    --manifest benchmark_output/5_15_1/<timestamp> \
    --scenes-root scenes/5_15_1
```

## Notes

- Cores without a wrong-spawn / correct-peer pair are skipped (console /
  `junctions.json`).
- Package overview: `../README.md` and `config/config.yaml`.
