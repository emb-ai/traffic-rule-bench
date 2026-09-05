# segments/

Corridor map harvest for speed / detour / crosswalk signs.

**Pipeline:** `enumerate` → `select` (train/test split only) → `crop` **all** candidates

```bash
python -m traffic_bench.scene_collection.collect.segments.enumerate
python -m traffic_bench.scene_collection.collect.segments.select
python -m traffic_bench.scene_collection.collect.segments.crop --skip-existing --workers 8
```

Or via collect (net already built):

```bash
python -m traffic_bench.scene_collection collect \
  --skip-download --skip-netconvert --skip-crop --skip-dual-path \
  --skip-existing
```

## Design

1. **Enumerate** full Moscow net → mid-corridor windows (≥150 m, one per `osm_way_id`).
2. **Select** = place-disjoint train/test split only (no harvest quota).
3. **Crop** every candidate (~13k) into `crops/segment/`.
4. **Assign** balances `(straight|curved) × (lanes 1|2|3plus)` **per sign** when picking 80/20.

Same philosophy as junctions: large city pool H≈P, protocol N=80/20 sampled later with topology balance.

## Modules

| File | Role |
| --- | --- |
| `metrics.py` | length / straightness / window / lane buckets |
| `enumerate.py` | full-net candidates → `index/segments.jsonl` |
| `select.py` | stratified way split → `segment_{train,test}_ids.json` |
| `crop.py` | XY neighborhood crop → `crops/segment/<id>/` |

## After crop: assign + materialize only corridor signs

`assign` always rewrites the full allocation file (no `--sign` filter). Then
materialize only the eight segment eval ids:

```bash
PY=/home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python

# 1) harvest segments only
$PY -m traffic_bench.scene_collection collect \
  --skip-download --skip-netconvert --skip-crop --skip-dual-path \
  --skip-existing --workers 8

# 2) refresh inventory stats
$PY -m traffic_bench.scene_collection analysis inventory

# 3) re-allocate all signs (segment pool used only by corridor signs)
$PY -m traffic_bench.scene_collection assign

# 4) rematerialize corridor signs only
for sign in speed_limit min_speed residential_zone zone_speed_limit \
            detour_right detour_left detour_either crosswalk; do
  $PY -m traffic_bench.scene_collection materialize --sign "$sign"
done

# 5) verify allocations + overlap after materialize
$PY -m traffic_bench.scene_collection analysis assign_verify
$PY -m traffic_bench.scene_collection analysis overlap
```
