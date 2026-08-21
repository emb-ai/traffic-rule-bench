# map_pool — sign-free Moscow harvest

Shared pool of maps. Signs do not own crop folders; they query a family in
`splits/signs.yaml`. Lives at `traffic_bench/scene_collection/map_pool/`.

Junction / dual_path maps are keyed by SUMO `junction_id`. Segment maps are
keyed by incoming `edge_id` and split by OSM way id.

## Layout

```
map_pool/
├── raw/  nets/  index/
├── lib/                 # dual_path, segment, roles, stem
├── crops/
│   ├── {T,X,O}/                    # junction-only (~80 m arms)
│   ├── dual_path/{T,X}/{slot}/     # path-union bbox
│   └── segment/<scene_id>/         # corridor; type is meta.segment_type
├── splits/
│   ├── signs.yaml / signs.json
│   ├── train_ids.json / test_ids.json
│   └── sign_allocations.json
└── scripts/
```

Unused leftovers that may still sit on disk: `crops/lane_direction`,
`crops/segment_detour`, `crops/segment_crosswalk`.

## Crop families

| Kind        | Path                               | Used by                                      |
| ----------- | ---------------------------------- | -------------------------------------------- |
| `junction`  | `crops/{T,X,O}/`                   | 2.x, 3.2, 4.3                                |
| `dual_path` | `crops/dual_path/{T,X}/{slot}/`    | 3.1, 3.18, 4.1, 5.7                          |
| `segment`   | `crops/segment/<scene_id>/`        | 3.24, 4.6, 5.21, 5.31, 4.2.x, 5.19           |

**Why dual_path is a family and “2 lanes” is not:** T/X is an ~80 m junction
crop. dual_path is a different net (union of two routes). Lane count and
straightness are columns on the same segment crop.

### Dual-path slots

Atomic slot = `(baseline_dir, compliant_dir)` among `{l,s,r}`:
`l_s` `l_r` `r_s` `r_l` `s_l` `s_r`.

No per-slot cap in this iteration (H = P). At most one atom per junction per
slot. Sign → slots: `lib/roles.py`. Extra gates: **5.7** → T only, stem /
carriageway.

### Segments

Incoming edges from `index/junctions.jsonl`, cropped so the scene **ends
before the junction** (10 m margin). Inclusion gates (`lib/segment.py`):

- length ≥ **150 m** — braking 60→20 km/h plus a compliance zone
- **straight** — chord/arc ≥ 0.99 (speed signs; curve-aware IDM must not brake)
- **curved** — 0.97 ≤ chord/arc < 0.99 (4.2 / 5.19; slight bend OK)

Written on each `meta.json` / index row: `length_m`, `straightness`,
`segment_type`, `lane_count` (passenger lanes only), `vehicle_lane_indices`
(SUMO 0 = rightmost), `pass_right_ok`, `pass_left_ok`, `osm_way_id`.

Not written at harvest: obstacle lane, `sign_s`, zebra (5.19 injects at
materialize).

## Sign queries (`splits/signs.yaml`)

- 3.24 / 4.6 / 5.21 / 5.31 → `segment_type: straight`
- 4.2.1 → `lane_count_min: 2` and `pass_right_ok`
- 4.2.2 → `lane_count_min: 2` and `pass_left_ok`
- 4.2.3 → `lane_count_min: 2`
- 5.19 → any segment, then `prepare: crosswalk` at materialize
- 5.15 is **not** listed (no harvest)

4.2 maps in the pool are the **same segment crops** as speed signs. Obstacle
lane placement is an eval concern, not a harvest column.

## Pipeline

T/X/O are already cropped for the full junction index. Overnight, finish P for
dual_path and segment (~5 s/netconvert, independent of family):

```bash
cd traffic-rule-bench

# dual_path: resume past the old 500-per-slot cap
python traffic_bench/scene_collection/map_pool/scripts/crop_dual_path_scenes.py \
  --max-per-slot 0 --skip-existing

# segment: all remaining index rows → crops/segment/<id>/
python traffic_bench/scene_collection/map_pool/scripts/crop_segment_scenes.py \
  --max-scenes 0 --skip-existing

python traffic_bench/scene_collection/map_pool/scripts/allocate_sign_scenes.py
```

`crop_segment_scenes.py` flattens leftover `segment/{straight,curved}/` dirs
and backfills `pass_*` on existing metas (no recrop).

Smoke:

```bash
python traffic_bench/scene_collection/map_pool/scripts/crop_dual_path_scenes.py --max-per-slot 5 --discover-only
python traffic_bench/scene_collection/map_pool/scripts/crop_segment_scenes.py --max-scenes 10
```

Full pipeline (download → net → enumerate → all crops):

```bash
python traffic_bench/scene_collection/map_pool/scripts/run_pipeline.py --skip-download --skip-netconvert --skip-existing
```

## Train / test

1. Shared pool — not owned by any sign.
2. Stamp place ids **before** allocating signs (`make_junction_split.py` for
   junctions; md5 of `osm_way_id` for segments).
3. Signs sample independently; the same map may appear under several signs.

Quotas: `splits/signs.yaml` (`n_train`, `n_test`, `seed`, query fields).

## Provenance

BBBike Moscow OSM extract → `nets/moscow.net.xml`. See `raw/DOWNLOAD_META.json`.
