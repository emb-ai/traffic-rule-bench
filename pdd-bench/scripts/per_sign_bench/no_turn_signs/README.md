# No-Turn Signs (3.18.1 / 3.18.2)

Family of **prohibited-maneuver** signs. Same dual-path incentive as 4.1.x:
baseline = shorter *forbidden* first exit, compliant = longer *allowed* path.

Registry: `lib/no_turn_sign_spec.py`. Sign classes live in
`traffic_signs/no_turn_allowed.py`.


| Code   | Title         | Forbidden | Allowed  | Scenes folder    |
| ------ | ------------- | --------- | -------- | ---------------- |
| 3.18.1 | No right turn | `r`       | `s`, `l` | `scenes/3_18_1/` |
| 3.18.2 | No left turn  | `l`       | `s`, `r` | `scenes/3_18_2/` |


## Dual-path roles


| Code   | Baseline (short) | Compliant (long) |
| ------ | ---------------- | ---------------- |
| 3.18.1 | `r`              | `s` / `l`        |
| 3.18.2 | `l`              | `s` / `r`        |


## Quick start

```bash
cd pdd-bench/scripts/per_sign_bench/no_turn_signs
conda activate zinkovich-plant2   # or your env

# 3.18.1 (no right)
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.18.1 --arms 4 --limit 30 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.18.1 --limit 20 --overwrite
python generate_manifest.py sign.pdd_code=3.18.1

# 3.18.2 (no left)
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.18.2 --arms 4 --limit 30 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.18.2 --limit 20 --overwrite
python generate_manifest.py sign.pdd_code=3.18.2
```

Eval places the matching `NoRightTurnSign` / `NoLeftTurnSign` on the ego
approach. Smoke: `idm` should take the short forbidden exit; `modified_idm`
should take the long allowed route.

To obtain metrics:

```bash
python generate_manifest.py sign.pdd_code=3.18.1 paths.output_base=benchmark_output/3_18_1/
gif.enabled=true gif.policy=modified_idm
```
