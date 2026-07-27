# Direction Signs (4.1.1–4.1.6)

One package for the whole family of mandatory movement-direction signs.
Members differ only by allowed directions (`allowed_dirs`); scene / manifest /
benchmark scaffolding is shared.

Family registry: `lib/direction_sign_spec.py`.

| Code  | Title                | `allowed_dirs`   | Scenes folder   |
|-------|----------------------|------------------|-----------------|
| 4.1.1 | Proceed straight     | `s`              | `scenes/4_1_1/` |
| 4.1.2 | Turn right           | `r`              | `scenes/4_1_2/` |
| 4.1.3 | Turn left            | `l` (+ U-turn)   | `scenes/4_1_3/` |
| 4.1.4 | Straight or right    | `s`, `r`         | `scenes/4_1_4/` |
| 4.1.5 | Straight or left     | `s`, `l`         | `scenes/4_1_5/` |
| 4.1.6 | Right or left        | `l`, `r`         | `scenes/4_1_6/` |

Each member keeps **separate** scene trees:

```
scenes/
├── 4_1_1/
│   ├── core/              # imported catalog cores
│   └── sign_*_j*/         # dual-path crops
└── 4_1_2/
    ├── core/
    └── sign_*_j*/
```

Default active member is **4.1.1**. Dual-path crop (4.1.1–4.1.6) stores
spawn/dest in scene meta: **baseline** = shorter forbidden first exit,
**compliant** = longer allowed first exit. Manifest reuses those endpoints.

| Sign  | Baseline (short) | Compliant (long) |
|-------|------------------|------------------|
| 4.1.1 | `l` or `r`       | `s`              |
| 4.1.2 | `s` (prefer) / `l` | `r`            |
| 4.1.3 | `s` (prefer) / `r` | `l`            |
| 4.1.4 | `l`              | `s` (prefer) / `r` |
| 4.1.5 | `r`              | `s` (prefer) / `l` |
| 4.1.6 | `s`              | `r` (prefer) / `l` |

At eval, baseline ``idm`` tends to take the short forbidden exit (violation);
``modified_idm`` / ``carl_rule`` / ``comprehensive_rule_expert`` replan via
``SignComplianceMixin`` onto the compliant first exit to the same dest.

## Setup

```bash
conda activate zinkovich-plant2
cd pdd-bench/scripts/per_sign_bench/direction_signs
```

## Workflow

### 4.1.1 (straight only)

```bash
python tools/filter_scenes/import_catalog_scenes.py --arms 4 --limit 20
python tools/filter_scenes/crop_junction_scene.py --limit 5
python generate_manifest.py sign.pdd_code=4.1.1 paths.output_base=benchmark_output/4_1_1
# → scenes/4_1_1/, benchmark_output/4_1_1/<timestamp>/
python eval_pipeline.py \
    --policies idm \
    --manifest benchmark_output/4_1_1/<timestamp> \
    --scenes-root scenes/4_1_1
```

### 4.1.2 (right only)

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.2 --arms 4 --limit 20
python tools/filter_scenes/crop_junction_scene.py --pdd-code 4.1.2 --limit 5 --overwrite
python generate_manifest.py sign.pdd_code=4.1.2
# → scenes/4_1_2/, benchmark_output/4_1_2/<timestamp>/
```

### 4.1.3 (left only)

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.3 --arms 4 --limit 20
python tools/filter_scenes/crop_junction_scene.py --pdd-code 4.1.3 --limit 5 --overwrite
python generate_manifest.py sign.pdd_code=4.1.3
# → scenes/4_1_3/, benchmark_output/4_1_3/<timestamp>/
```

### 4.1.4 (straight or right)

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.4 --arms 4 --limit 20
python tools/filter_scenes/crop_junction_scene.py --pdd-code 4.1.4 --limit 5 --overwrite
python generate_manifest.py sign.pdd_code=4.1.4
# → scenes/4_1_4/, benchmark_output/4_1_4/<timestamp>/
```

### 4.1.5 (straight or left)

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.5 --arms 4 --limit 20
python tools/filter_scenes/crop_junction_scene.py --pdd-code 4.1.5 --limit 5 --overwrite
python generate_manifest.py sign.pdd_code=4.1.5
# → scenes/4_1_5/, benchmark_output/4_1_5/<timestamp>/
```

### 4.1.6 (right or left)

```bash
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.6 --arms 4 --limit 20
python tools/filter_scenes/crop_junction_scene.py --pdd-code 4.1.6 --limit 5 --overwrite
python generate_manifest.py sign.pdd_code=4.1.6
# → scenes/4_1_6/, benchmark_output/4_1_6/<timestamp>/
```

Crop writes ``road_id``, ``destination_*``, and ``dual_path`` (both edge lists);
``custom_cropped.png`` overlays full spawn→dest routes (blue = compliant / longer,
orange = baseline / shorter). Manifest copies those endpoints into each row.
Eval places the matching ``LaneAllowedDirectionSign4_1_*`` on the ego approach.

**Invalid dual-path (auto-skipped):** the compliant route must not return onto
the *signed approach edge* after the first allowed exit (e.g. right → loop →
same approach → through the X under the still-active sign). Re-entering the
junction from a *different* arm is fine — that arm has no 4.1.x sign.
Check: ``ego_edge_id in straight_path`` via ``path_revisits_signed_approach``.

## Next

- Split catalogs / seeds per code if needed
