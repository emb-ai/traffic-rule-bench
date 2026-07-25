# Lane Direction Signs (5.15.1)

PDD **5.15.1** — «Направления движения по полосам». Package mirrors
`direction_signs/` but selects multi-lane approaches and forces a peer
lane-change before the junction.

| Code   | Title                              | Scenes folder   |
|--------|------------------------------------|-----------------|
| 5.15.1 | Directions of movement by lanes    | `scenes/5_15_1/` |

```
scenes/
└── 5_15_1/
    ├── core/              # imported from pdd-bench/scenes/{5.15.2,4.1.x,…}
    └── sign_*_j*/         # lane-change crops
```

## Scenario design

1. Junction has ≥1 approach with **≥2 lanes**.
2. Ego spawns only on such an approach, on a **wrong** lane that has **no**
   first-exit route to the destination.
3. A **target** peer lane on the same approach can reach the destination.
4. Crop preview draws:
   - per-lane allowed-direction **arrows**
   - **orange dashed** = wrong path (stay on spawn lane)
   - **blue** = correct path (after lane-change → dest)

At eval, plain `idm` tends to stay on the spawn lane and miss dest / take the
wrong exit; sign-compliant experts lane-change to `target_lane_num` then follow
the correct exit (`SignComplianceMixin`).

## Setup

```bash
conda activate zinkovich-plant2
cd pdd-bench/scripts/per_sign_bench/lane_direction_signs
```

## Workflow

```bash
# 1) Import cores (catalog 5.15.2, keep junctions with a multi-lane arm)
python tools/filter_scenes/import_catalog_scenes.py --arms 4 3 --limit 20

# 2) Crop lane-change scenarios (arrows + wrong/correct paths)
python tools/filter_scenes/crop_junction_scene.py --limit 5 --overwrite

# 3) Manifest + eval
python generate_manifest.py sign.pdd_code=5.15.1
# optional global cap on dual-path scenarios:
python generate_manifest.py scenario.max_scenarios=20
python eval_pipeline.py \
    --policies idm modified_idm \
    --manifest benchmark_output/5_15_1/<timestamp> \
    --scenes-root scenes/5_15_1
```

Crop writes `road_id`, `spawn_lane_num`, `target_lane_num`, `destination_*`,
and `dual_path` (wrong=`turn_*`, correct=`straight_*`) into `meta.json`.
