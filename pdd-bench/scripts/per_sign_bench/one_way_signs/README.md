# One-Way Entry Signs (5.7.1 / 5.7.2)

Family of **exit onto a one-way road** signs. Related to no-turn 3.18.x
(**5.7.1 ≅ 3.18.2 no-left-turn**) — baseline = shorter *forbidden* first exit,
compliant = longer *allowed* path — but with **one-way-road semantics** that
make it stricter than a plain no-turn (see below).

Registry: `lib/one_way_sign_spec.py`. Sign classes / icons:
`traffic_signs/one_way_entry_sign.py`, `traffic_signs/icons/5.7.1.png`.

## One-way semantics (how 5.7.x differs from 3.18.x)

The crossing road the ego turns onto is **one-way**. Scene generation
(`lib/direction_dual_path.py`) enforces:

1. **Stem entry.** The ego approach must be able to turn onto the *same*
   crossing OSM way on **both** carriageways (the forbidden and the allowed
   turn share the crossing road's base id). On a T this forces the ego onto the
   **stem**; approaches sitting *on* the one-way road itself are rejected.
2. **Compliant route fully detours.** The forbidden-flow ("wrong-way")
   carriageway of the crossing road is removed from the compliant Dijkstra, so
   the allowed route turns the legal way and **loops around via other roads** —
   it never re-enters the one-way road against its flow and never U-turns back
   onto it. (This fixes the old `sign_100357_j0` case that drove back down the
   one-way.) The destination is only kept if it is reachable this way.
3. **One-directional background.** The wrong-way carriageway edges are recorded
   in `meta.json` / the manifest as `background_excluded_edges` (also under
   `dual_path.wrong_dir_edges`). `OneWaySumoTrafficManager` (in
   `run_benchmark.py`) never spawns NPCs on them or routes NPCs through them, so
   the crossing road shows traffic in **one direction only**.

The wrong-way carriageway is *kept in the net* so a non-compliant ego **can**
illegally enter it and be flagged by `OneWayEntrySign`. Sign-compliant planners
receive `sign.one_way_forbidden_edges` and replan a detour that avoids the whole
wrong-way carriageway (not just the first forbidden exit).


| Code  | Title                          | Forbidden | Allowed  | Like     | Scenes folder  |
| ----- | ------------------------------ | --------- | -------- | -------- | -------------- |
| 5.7.1 | Exit onto one-way (right)      | `l`       | `s`, `r` | 3.18.2   | `scenes/5_7_1/` |
| 5.7.2 | Exit onto one-way (left)       | `r`       | `s`, `l` | 3.18.1   | `scenes/5_7_2/` |


## Dual-path roles


| Code  | Baseline (short) | Compliant (long) |
| ----- | ---------------- | ---------------- |
| 5.7.1 | `l`              | `s` / `r`        |
| 5.7.2 | `r`              | `s` / `l`        |


## Quick start

```bash
cd pdd-bench/scripts/per_sign_bench/one_way_signs
conda activate zinkovich-plant2   # or your env

# 5.7.1 — X and T junctions (default --arms 4 3; T kept only if left exit exists)
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 5.7.1 --arms 4 3 --limit 50 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 5.7.1 --limit 40 --overwrite
python generate_manifest.py sign.pdd_code=5.7.1

# 5.7.2
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 5.7.2 --arms 4 3 --limit 50 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 5.7.2 --limit 40 --overwrite
python generate_manifest.py sign.pdd_code=5.7.2
```

Eval places `OneWayEntrySignR` (5.7.1, icon `5.7.1.png`) / `OneWayEntrySignL`
(5.7.2) on the ego approach. Smoke: `idm` takes the short forbidden exit onto
the wrong-way carriageway and is flagged (`OneWayEntrySignR: 1`,
`driving_score 0`); sign-compliant planners replan the long legal detour.

```bash
python generate_manifest.py sign.pdd_code=5.7.1 gif.enabled=true gif.policy=modified_idm
```
