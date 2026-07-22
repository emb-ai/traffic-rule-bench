# One-Way Entry Signs (5.7.1 / 5.7.2)

Family of **exit onto a one-way road** signs. Same dual-path incentive as
no-turn / 4.1.x: baseline = shorter *forbidden* first exit, compliant = longer
*allowed* path.

Registry: `lib/one_way_sign_spec.py`. Sign classes live in
`traffic_signs/one_way_entry_sign.py`.


| Code  | Title                          | Forbidden | Allowed  | Scenes folder  |
| ----- | ------------------------------ | --------- | -------- | -------------- |
| 5.7.1 | Exit onto one-way (right)      | `l`       | `s`, `r` | `scenes/5_7_1/` |
| 5.7.2 | Exit onto one-way (left)       | `r`       | `s`, `l` | `scenes/5_7_2/` |


## Dual-path roles


| Code  | Baseline (short) | Compliant (long) |
| ----- | ---------------- | ---------------- |
| 5.7.1 | `l`              | `s` / `r`        |
| 5.7.2 | `r`              | `s` / `l`        |


## Quick start

```bash
cd pdd-bench/scripts/per_sign_bench/one_way_signs
conda activate zinkovich-plant2   # or your env

# 5.7.1 — X and T junctions (T only if left/forbidden exit exists)
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 5.7.1 --arms 4 3 --limit 50 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 5.7.1 --limit 40 --overwrite
python generate_manifest.py sign.pdd_code=5.7.1

# 5.7.2
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 5.7.2 --arms 4 3 --limit 50 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 5.7.2 --limit 40 --overwrite
python generate_manifest.py sign.pdd_code=5.7.2
```

Eval places `OneWayEntrySignR` (5.7.1) / `OneWayEntrySignL` (5.7.2) on the ego
approach. Smoke: `idm` should take the short forbidden exit; `modified_idm`
should take the long allowed route.

```bash
python generate_manifest.py sign.pdd_code=5.7.1 gif.enabled=true gif.policy=modified_idm
```
