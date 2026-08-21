# map_pool — Moscow map harvest (junction + dual_path + segment)

Sign-free scene harvest for Moscow. Lives at `traffic_bench/scene_collection/map_pool/`.

Sign-free scene harvest for Moscow. Junction / dual_path scenes are keyed by
**SUMO `junction_id`** (or roundabout fingerprint). Segment scenes are keyed by
incoming **edge_id** and split by **OSM way id**.

Formerly `moscow_junctions`. Renamed because the pool now includes **dual_path**
crops (path-union bbox) and **segment** crops (straight-road stubs), not only
junction-only maps.

> Naming: use **dual_path**, not “detour” (PDD **4.2.1** is the detour sign).

## Layout

```
moscow_scenes/
├── README.md
├── raw/ … nets/ … index/
├── lib/                         # dual_path, lane_direction, segment, roles, stem
├── crops/
│   ├── {T,X,O}/                 # junction-only (~80 m arms)
│   ├── dual_path/{T,X}/{slot}/  # path-union crops (l_s, l_r, …)
│   ├── lane_direction/{T,X}/    # 5.15.1 multi-lane LC crops
│   └── segment/{straight,curved}/  # straight-road stubs (no intersection)
├── splits/
│   ├── signs.yaml / signs.json
│   ├── train_ids.json / test_ids.json
│   └── sign_allocations.json
└── scripts/
    ├── build_net.py / enumerate_junctions.py / crop_scenes.py
    ├── crop_dual_path_scenes.py / crop_lane_direction_scenes.py
    ├── enumerate_segments.py / crop_segment_scenes.py
    ├── make_junction_split.py / allocate_sign_scenes.py
    └── run_pipeline.py
```

## Crop kinds


| Kind             | Path                                  | Used by                         |
| ---------------- | ------------------------------------- | ------------------------------- |
| `junction`       | `crops/{T,X,O}/`                     | 2.x, 3.2, 4.3, …                |
| `dual_path`      | `crops/dual_path/{T,X}/{slot}/`      | 5.7, 3.18, 4.1, 3.1             |
| `lane_direction` | `crops/lane_direction/{T,X}/`        | **5.15.1** (multi-lane LC)      |
| `segment`        | `crops/segment/{straight,curved}/`   | **3.24**, **5.19**, **4.2.x**   |


### Dual-path slots

Atomic slot = exact `(baseline_dir, compliant_dir)` among `{l,s,r}`:

`l_s` `l_r` `r_s` `r_l` `s_l` `s_r`

**Pool size:** at most **500** scenes per `(shape, slot)` (default `--max-per-slot 500`).

Same shared-pool idea as junction `T/X/O` (many signs reuse train/test maps).
Do **not** size the pool as one sign’s `n_train+n_test`. Cap exists only because
each dual_path crop is a path-union netconvert (heavier than junction-only); 500
is roughly X-inventory scale per bucket and leaves headroom under 80/20 split and
slot/stem filters. At most **one** atom per junction per slot; shuffle `--seed 42`;
early-stop when buckets are full.

Sign → slots (allocate filter): see `lib/roles.py` (`SIGN_TO_SLOTS`).
Extra gates: **5.7** → T only, `ego_is_t_stem`, `carriageway_pair`.

### Segments (straight-road signs)

These signs are tested **on a road, not at an intersection**. Source is
`incoming_edge_ids` from `index/junctions.jsonl`: the approach is cropped so the
scene **ends before the junction** (10 m margin). That reuses the existing
junction harvest and keeps the map a plain corridor.

Filters (`lib/segment.py`):

- length ≥ **150 m** — covers braking from ego spawn 60 km/h to a 20 km/h limit
  (`d_brake ≈ 47 m` at 3.5 m/s²) plus a ~60 m compliance zone and buffers
- **straight** — chord/arc ≥ 0.99 → **3.24** (curve-aware IDM must not brake
  from geometry)
- **curved** — chord/arc ≥ 0.97 → **5.19**, **4.2.x** (slight curvature OK)

Each scene: `map.net.xml`, `meta.json`, `center.json`, `custom_cropped.png`.

## Pipeline

```bash
cd traffic-rule-bench
# from repo root; map_pool scripts use ROOT = this package dir
python traffic_bench/scene_collection/map_pool/scripts/run_pipeline.py --skip-download --skip-netconvert

# Junction-only harvest (existing)
python scripts/run_pipeline.py --skip-download --skip-netconvert

# Dual-path harvest (among the same enumerated T/X junctions)
# Default: 500 scenes per (shape, slot) — shared pool, not one-sign quota.
# Crops incrementally (each atom → disk + candidates flush); Ctrl+C safe with
# --skip-existing (resume recounts on-disk scenes toward the cap).
python scripts/crop_dual_path_scenes.py --max-per-slot 500 --skip-existing
# smoke:  python scripts/crop_dual_path_scenes.py --max-per-slot 5 --discover-only
# subset: python scripts/crop_dual_path_scenes.py --max-per-slot 500 --slots l_s,l_r

# Lane-direction harvest for 5.15.1 (multi-lane approach + exclusive L/R peer exit)
python scripts/crop_lane_direction_scenes.py --max-per-shape 500 --skip-existing
# smoke:  python scripts/crop_lane_direction_scenes.py --max-junctions 50 --max-per-shape 5

# Segment harvest for 3.24 / 5.19 / 4.2.x (incoming edges, cropped before junction)
python scripts/enumerate_segments.py
python scripts/crop_segment_scenes.py --max-per-type 500 --skip-existing
# smoke:  python scripts/crop_segment_scenes.py --max-per-type 10

python scripts/make_junction_split.py  # global train/test split among X/T/O
python scripts/allocate_sign_scenes.py
```

## Train / test

1. Shared pool — not owned by any sign.
2. Junction / dual_path / lane_direction: prefer split by `**junction_id**` so
  crops of the same junction stay on the same side (no road leakage across
  train/test).
3. Segments: split by `**osm_way_id**` so consecutive SUMO edges of one OSM way
  cannot land in both train and test.
4. Signs sample from the matching `crop_kind` with shape / slot / segment_type
  filters.

Quotas: `splits/signs.yaml` (`n_train`, `n_test`, `seed`, `crop_kind`, …).

## Provenance

BBBike Moscow OSM extract → `nets/moscow.net.xml`. See `raw/DOWNLOAD_META.json`.
