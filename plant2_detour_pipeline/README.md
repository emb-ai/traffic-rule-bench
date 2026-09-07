# Detour signs (4.2.1 / 4.2.2 / 4.2.3) fine-tune

Obstacle-detour signs: a cone/obstacle sits on the road; SUMO routing does
**not** account for it (routing goes straight through), so the model must
predict waypoints that swerve around it — left or right depending on which
of the three signs is posted — purely from what it perceives, not from the
route.

## Data source

Dumps and splits were pre-collected separately (not by this pipeline) under
`$SM = /home/jovyan/shares/SR006.nfs2/smirnova`:

| What | Path |
|---|---|
| Dumps | `$SM/plant2_fix/plant2_l1_from_experts_signs/data/sumo_4.2.{1,2,3}_*` (475/384/510 routes) |
| Original split | `$SM/plant2_fix/detour_split/` (train 1164, val 205) |
| Maps | `$SM/sdc/pdd-bench/scenes/4.2.{1,2,3}/` |
| Test catalog | `$SM/traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl` (122 scenes: 43/33/46) |

## Data verification (done before spending GPU time)

Per the task's own concern list — checked and confirmed correct except one
issue:

1. **Dump counts match exactly** (475/384/510).
2. **Cones appear correctly in x_objs** at plausible coordinates (`class:
   'static'`, `type_id: 'static.prop.constructioncone'`, positions tens of
   meters ahead of ego, monotonically approaching across frames as ego
   drives up to them) — verified against known-good schema.
3. **Routing does *not* curve around the obstacle.** `measurements/*.json.gz`
   `route`/`route_original` are the resampled *actual* future ego trajectory
   (BC regression targets), not a SUMO-graph path; lane `.net.xml` shapes
   near the obstacle's `s` position are smooth with no localized bulge —
   confirms the routing graph is unaware of the obstacle, as intended. The
   actual recorded trajectory does deviate laterally (~3m clearance,
   consistently negative offset for 4.2.1 / positive for 4.2.2 — correct
   mirror-opposite "pass left" vs "pass right" semantics).
4. **Splits correct**: symlink targets resolve to the right sign type, no
   4.2.1/4.2.2/4.2.3 mixups, counts match (`detour_split` train
   1164=405+327+432, val 205=70+57+78; `detour_tiny` 36/36; `detour_one`
   1/1).

### Issue found and fixed: obstacle-absent routes

**~20-26% of routes across all three signs never show the obstacle in any
frame** despite `results.json.gz` reporting `status: Completed,
score_composed: 100.0` — the episode is just ordinary lane-following
labeled as a "detour" scene, giving zero avoidance signal. Confirmed two
ways: an independent scan of `boxes/*.json.gz` for `class=='static'`, and
the split's own `sample_weights.json` (`"cones": []` for the exact same
routes — root cause is presumably a routing/obstacle-placement coupling
issue upstream in the collection tool, not investigated further here since
the fix is the same either way: drop them).

| sign | affected (of split total) |
|---|---:|
| 4.2.1 | 98/405 train (24.2%), 14/70 val |
| 4.2.2 | 26/327 train (7.9%), 9/57 val |
| 4.2.3 | 115/432 train (26.6%), 21/78 val |

**Fix**: [`scripts/plant2_ft_pipeline/data/filter_detour_split.py`](../scripts/plant2_ft_pipeline/data/filter_detour_split.py)
drops these routes from the original train/val partition (excludes, doesn't
reshuffle — same seed/assignment as `detour_split` otherwise) using the
`sample_weights.json` `cones` field, symlinking the rest into
[`detour_split_filtered/`](detour_split_filtered) in this directory:

```bash
python scripts/plant2_ft_pipeline/data/filter_detour_split.py \
  --src $SM/plant2_fix/detour_split \
  --out plant2_detour_pipeline/detour_split_filtered
```

Result: **train 925** (307/301/317), **val 161** (56/48/57).

The 122-scene test catalog was *not* filtered — its `detour_cones` field
(True for ~53%, False for ~47% of rows) is a deliberate, labeled test design
(both "must avoid" and "no obstacle present, don't swerve unnecessarily"
cases), unrelated to the training-data collection issue above.

## Fine-tune

`run_plant2_finetune.py` on `detour_split_filtered`, same pretrain
checkpoint as the stop-sign experiments
(`stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`), LR=1e-4,
20 epochs, batch_size=512 (larger split than stop-sign's 294 routes, so a
lower LR / longer budget than the 3e-4/10-epoch stop-sign setting).
`checkpoint-addon = detour_4212223_lr1e4_ep20`.

## Eval

[`run_eval.sh <tag> <ckpt> [gpu]`](run_eval.sh):

1. Full FV-fast metrics on the 122-scene test catalog (no gifs) →
   `eval/<tag>/fv_fast/reports/report_cumulative.md`.
2. **5 sample GIFs** (one per sign with `detour_cones=True`, plus two
   `detour_cones=False` false-positive spot checks for 4.2.1/4.2.3) →
   `eval/<tag>/gifs/`. Full-catalog gifs weren't generated (122 scenes ×
   ~3-5x slowdown per the harness's own `--save-gifs` warning) — this is a
   qualitative spot-check, not the metrics run.
