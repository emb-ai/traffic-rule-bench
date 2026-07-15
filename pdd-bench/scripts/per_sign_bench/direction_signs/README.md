# Direction Signs (4.1.1–4.1.6)

One package for the whole family of mandatory movement-direction signs.
Members differ only by allowed directions (`allowed_dirs`); scene / manifest /
benchmark scaffolding is shared.

Family registry: `lib/direction_sign_spec.py`.

| Code  | Title                | `allowed_dirs`   |
|-------|----------------------|------------------|
| 4.1.1 | Proceed straight     | `s`              |
| 4.1.2 | Turn right           | `r`              |
| 4.1.3 | Turn left            | `l` (+ U-turn)   |
| 4.1.4 | Straight or right    | `s`, `r`         |
| 4.1.5 | Straight or left     | `s`, `l`         |
| 4.1.6 | Right or left        | `l`, `r`         |

Default active member is **4.1.1**. For 4.1.1, crop selects dual-path
spawn/dest (shorter turn vs longer straight) and stores them in scene meta;
manifest reuses those endpoints. At eval, baseline ``idm`` follows the short
turn (violation); ``modified_idm`` / ``carl_rule`` / ``comprehensive_rule_expert``
replan via ``SignComplianceMixin`` onto the straight first exit to the same dest.

## Setup

```bash
conda activate zinkovich-plant2
cd pdd-bench/scripts/per_sign_bench/direction_signs
```

## Folder structure

```
direction_signs/
├── build_scene.py
├── generate_manifest.py      # Hydra; sign.pdd_code selects family member
├── eval_pipeline.py
├── run_benchmark.py          # Places LaneAllowedDirectionSign4_1_* on ego
├── lib/
│   ├── direction_sign_spec.py   # registry for 4.1.1–4.1.6
│   ├── junction_*.py            # shared junction topology (like main/stop)
│   └── …
├── tools/filter_scenes/      # import/crop from pdd-bench/scenes/<code>/
├── config/config.yaml
├── scenes/
└── benchmark_output/
```

## Workflow (scaffold)

1. Import from catalog (prefer 4-arm junctions):

```bash
python tools/filter_scenes/import_catalog_scenes.py --arms 4 --limit 20
```

2. Crop **dual-path** scenes for 4.1.1 (variant 1):

   - Find an X approach where the **same** destination is reachable via a
     shorter left/right turn **and** a longer straight path.
   - Crop to the XY bbox of both paths (+ margin), not a tight stub around the
     junction alone.

```bash
python tools/filter_scenes/crop_junction_scene.py --limit 5
python tools/filter_scenes/crop_junction_scene.py sign_72915 --overwrite --min-gain 20 --margin 40
```

3. Manifest reuses crop-time spawn/dest from ``meta.json`` (no rediscovery):

```bash
python generate_manifest.py
# another family member (not dual-path yet):
python generate_manifest.py sign.pdd_code=4.1.2 paths.output_base=benchmark_output/4_1_2
```

Crop writes ``road_id``, ``destination_*``, and ``dual_path`` (both edge lists);
``custom_cropped.png`` overlays full spawn→dest routes (blue = straight / longer,
orange = turn / shorter; laterally offset so the shared final edges stay
visible). Manifest copies those endpoints into each row. Eval places
``LaneAllowedDirectionSign4_1_1`` on the ego approach.

4. Eval:

```bash
python eval_pipeline.py \
    --policies idm \
    --manifest benchmark_output/4_1_1/<timestamp> \
    --scenes-root scenes
```

## Next

- Extend dual-path selection to other 4.1.x ``allowed_dirs``
- Split catalogs / seeds per code if needed
