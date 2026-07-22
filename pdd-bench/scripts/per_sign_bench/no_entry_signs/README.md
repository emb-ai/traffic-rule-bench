# No-Entry Signs (3.1 / 3.2)

Family of **no-entry / movement-prohibited** signs.

| Code | Title                 | Sign class      | Scenes folder |
| ---- | --------------------- | --------------- | ------------- |
| 3.1  | No entry              | `NoEntrySign`   | `scenes/3_1/` |
| 3.2  | Movement prohibited   | `NoTrafficSign` | `scenes/3_2/` |

Incentive (same for both members):

* **Baseline (`idm`)** drives past the sign into the junction (ignores it).
* **Experts** (`modified_idm`, rule experts, …) stop **strictly before** the sign
  when the destination requires crossing it.

Signs are placed artificially at the **start of the forbidden (destination)
lane** (`sign_distance_from_start`), not at the end of the approach and not at
catalog `distance_from_start`.

Registry: `lib/no_entry_sign_spec.py`. Classes live in
`traffic_signs/no_entry_sign.py` and `traffic_signs/no_traffic_sign.py`.


## Quick start

```bash
cd pdd-bench/scripts/per_sign_bench/no_entry_signs
conda activate zinkovich-plant2   # or your env

# 3.1 — import → core → crop → review → generate_manifest
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.1 --arms 4 3 --limit 40 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.1 --limit 40 --overwrite
python tools/filter_scenes/review_junction_scenes.py --pdd-code 3.1

# Manifest + smoke GIFs
python generate_manifest.py sign.pdd_code=3.1
python generate_manifest.py sign.pdd_code=3.1 gif.enabled=true gif.policy=idm gif.max_scenes=3
python generate_manifest.py sign.pdd_code=3.1 gif.enabled=true gif.policy=modified_idm gif.max_scenes=3

# Same for 3.2
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.2 --arms 4 3 --limit 40 --no-simulation
python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.2 --limit 40 --overwrite
python generate_manifest.py sign.pdd_code=3.2
```

Crops target multi-arm (≥3) junctions (X/T). The sign sits on the forbidden
destination lane past the junction (`sign_distance_from_start`, default 10 m)
so a compliant stop is not right in the intersection. Ego spawns on the
approach arm (`spawn_distance_before_end`, default 25 m).

If ego stays stopped before the sign for
`compliant_stop_success_seconds` (default 3 s) with 0 violations, the episode
ends early and counts as `reached_dest` (**only** this compliant-stop case
overrides arrive_dest; timeout/crash do not).

The route endpoint is shortly past the sign (`destination_past_sign_m`,
default 8 m): non-compliant agents that cross the line finish there instead of
driving to the far end of a long forbidden edge.

Eval:

```bash
python eval_pipeline.py \
  --policies idm,modified_idm \
  --manifest benchmark_output/3_1/<run_dir> \
  --scenes-root scenes/3_1
```

Expected smoke: `idm` accumulates no-entry violations (drives into the junction);
`modified_idm` stops before the sign with 0 violations (and may not reach dest —
`select_experts` treats 3.1/3.2 as success without arrive_dest when clean).
