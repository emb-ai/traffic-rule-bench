# maps/ — on-disk harvest (data only)

Sign-free Moscow maps. Code lives under `collect/`, `assign/`, `sign_scenes/`.
Signs query this pool via `splits/signs.yaml`.

| Path | Role |
| --- | --- |
| [`raw/`](raw/README.md) | OSM extract (gitignored) |
| [`nets/`](nets/README.md) | City `moscow.net.xml` (gitignored) |
| [`index/`](index/README.md) | Junction / segment / dual-path JSONL catalogs |
| [`crops/`](#layout) | Cropped scene folders (gitignored; regenerate with `collect`) |
| [`splits/`](splits/README.md) | `signs.yaml`, train/test ids, allocations |
| [`previews/`](previews/README.md) | City overview PNG |

Junction / dual_path maps are keyed by SUMO `junction_id`. Segment maps are
keyed by incoming `edge_id` and split by OSM way id.

## Layout

```
maps/
├── raw/  nets/  index/
├── previews/junctions_overview.png
├── crops/
│   ├── junction/{T,X,O}/           # junction-only (~80 m arms)
│   ├── dual_path/{T,X}/{slot}/     # path-union bbox
│   └── segment/<scene_id>/         # corridor; type is meta.segment_type
└── splits/
    ├── signs.yaml
    ├── train_ids.json / test_ids.json
    └── sign_allocations.json
```

`crops/` is gitignored (regenerate with `python -m traffic_bench.scene_collection collect`).

## Crop families

| Kind        | Path                               | Used by                                      |
| ----------- | ---------------------------------- | -------------------------------------------- |
| `junction`  | `crops/junction/{T,X,O}/`          | 2.x, 3.2, 4.3                                |
| `dual_path` | `crops/dual_path/{T,X}/{slot}/`    | 3.1, 3.18, 4.1, 5.7                          |
| `segment`   | `crops/segment/<scene_id>/`        | 3.24, 4.6, 5.21, 5.31, 4.2.x, 5.19           |

### Dual-path slots

Atomic slot = `(baseline_dir, compliant_dir)` among `{l,s,r}`:
`l_s` `l_r` `r_s` `r_l` `s_l` `s_r`. Sign → slots: `collect/dual_path/roles.py`.
Extra gates: **5.7** → T only, stem / carriageway.

### Segments

Incoming edges from `index/junctions.jsonl`, cropped so the scene **ends
before the junction** (10 m margin). Inclusion gates (`collect/segments/metrics.py`):

- length ≥ **150 m** — braking 60→20 km/h plus a compliance zone
- **straight** — chord/arc ≥ 0.99 (speed signs)
- **curved** — 0.97 ≤ chord/arc < 0.99 (4.2 / 5.19)

Written on each `meta.json` / index row: `length_m`, `straightness`,
`segment_type`, `lane_count`, `vehicle_lane_indices` (SUMO 0 = rightmost),
`pass_right_ok`, `pass_left_ok`, `osm_way_id`.

Not written at harvest: obstacle lane, `sign_s`, zebra (5.19 injects in
`data/scenes/crosswalk/` via `prepare`).

## Sign queries (`splits/signs.yaml`)

- 3.24 / 4.6 / 5.21 / 5.31 → `segment_type: straight`
- 4.2.1 → `lane_count_min: 2` and `pass_right_ok`
- 4.2.2 → `lane_count_min: 2` and `pass_left_ok`
- 4.2.3 → `lane_count_min: 2`
- 5.19 → any segment, then `prepare: crosswalk` (copy + zebra, middle only)
- 5.15 is **not** listed (no harvest)

4.2 maps in the pool are the **same segment crops** as speed signs. Obstacle
lane placement is an eval concern (`detour_expansion.py`).

## Provenance

BBBike Moscow OSM extract → `nets/moscow.net.xml`. See `raw/DOWNLOAD_META.json`.
