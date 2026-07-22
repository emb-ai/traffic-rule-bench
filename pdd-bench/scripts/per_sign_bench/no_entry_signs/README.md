# No-Entry Signs (3.1 / 3.2)

Family of **no-entry / movement-prohibited** signs.

| Code | Title                 | Sign class      | Scenes folder |
| ---- | --------------------- | --------------- | ------------- |
| 3.1  | No entry              | `NoEntrySign`   | `scenes/3_1/` |
| 3.2  | Movement prohibited   | `NoTrafficSign` | `scenes/3_2/` |

Incentive (same for both members):

* **Baseline (`idm`)** drives onto the forbidden road past the sign (ignores it).
* **Experts** (`modified_idm`, rule experts, …) stop **strictly before** the sign
  line when the destination requires crossing it.

Signs are placed at the catalog `distance_from_start` on `road_id` — the same
map location as in `pdd-bench/scenes/3.1` / `3.2`.

Registry: `lib/no_entry_sign_spec.py`. Classes live in
`traffic_signs/no_entry_sign.py` and `traffic_signs/no_traffic_sign.py`.


## Quick start

```bash
cd pdd-bench/scripts/per_sign_bench/no_entry_signs
conda activate zinkovich-plant2   # or your env

# 3.1 — import catalog scenes (already local OSM extracts)
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.1 --limit 40 --no-simulation

# Optional visual review
python tools/filter_scenes/review_junction_scenes.py --pdd-code 3.1

# Manifest + smoke GIFs
python generate_manifest.py sign.pdd_code=3.1
python generate_manifest.py sign.pdd_code=3.1 gif.enabled=true gif.policy=idm gif.max_scenes=3
python generate_manifest.py sign.pdd_code=3.1 gif.enabled=true gif.policy=modified_idm gif.max_scenes=3

# Same for 3.2
python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.2 --limit 40 --no-simulation
python generate_manifest.py sign.pdd_code=3.2
```

Eval:

```bash
python eval_pipeline.py \
  --policies idm,modified_idm \
  --manifest benchmark_output/3_1/<run_dir> \
  --scenes-root scenes/3_1
```

Expected smoke: `idm` accumulates no-entry violations; `modified_idm` stops
before the sign with 0 violations (and may not reach dest — `select_experts`
treats 3.1/3.2 as success without arrive_dest when clean).
