# moscow_junctions — T / X / O scenes from the Moscow OSM map

Sign-free junction harvest for Moscow. Scenes are keyed by **SUMO `junction_id`**
(or roundabout node-set fingerprint), not by traffic-sign database IDs.

## Layout

```
moscow_junctions/
├── README.md
├── raw/ … nets/ … index/ … scenes/{T,X,O}/
├── previews/moscow_net_overview.png
├── splits/
│   ├── signs.yaml / signs.json # per-sign shape quotas (JSON is what scripts read)
│   ├── train_ids.json          # global split (by scene_id)
│   ├── test_ids.json
│   ├── split_summary.json
│   └── sign_allocations.json   # shared-pool samples per sign
└── scripts/
    ├── build_net.py / enumerate_junctions.py / crop_scenes.py / run_pipeline.py
    ├── make_junction_split.py
    └── allocate_sign_scenes.py
```

## Train / test + per-sign allocation

**Logic (brief):**

1. **One shared scene pool** — `scenes/{T,X,O}/` is not owned by any sign.
2. **One global split** by `scene_id` (`junc_*` / `rb_*`), stratified by shape, **80/20**, **seed=42**. A junction never appears in both train and test.
3. **Shared pool across signs** — each sign independently samples from the same `train_ids` / `test_ids` (the same map may be used for 2.4 and 2.1).
4. **Quotas**
   - Most signs: **~115 train maps**, **X/T = 50/50** (`x_share: 0.5`).
   - **4.3**: only **O**.
   - **5.7.1 / 5.7.2**: only **T**.
   - Test ≈ `115 * 0.2/0.8 ≈ 29` maps per sign (same shape mix).

**What is `x_share`?** (informal synonym: `x_frac`)  
Fraction of allocated maps that are **X** when the sign allows both T and X.  
`x_share: 0.5` → half X, half T. Ignored for T-only or O-only signs.

```bash
python scripts/make_junction_split.py          # → splits/train_ids.json, test_ids.json
python scripts/allocate_sign_scenes.py        # → splits/sign_allocations.json
# edit quotas in splits/signs.json (keep signs.yaml in sync) then re-run allocate
```

Each scene folder contains:

| File | Role |
|------|------|
| `map.net.xml` | Cropped SUMO net around the junction / roundabout |
| `meta.json` | `junction_id`, `shape` ∈ {T,X,O}, lat/lon, crop radius, source net |
| `center.json` | `{lat, lon}` of the crop center |

## Provenance (where the map came from)

| Field | Value |
|-------|--------|
| Source | [BBBike Moscow extract](https://download.bbbike.org/osm/bbbike/Moscow/) |
| Files | `Moscow.osm.gz` → `Moscow.osm` (XML for netconvert); `Moscow.osm.pbf` archival |
| URL (XML) | `https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.gz` |
| URL (PBF) | `https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.pbf` |
| Coverage | BBBike city clip; OSM header bounds ≈ `55.566–55.916°N`, `37.322–37.881°E` (MKAD-ish) |
| Downloaded | see `raw/DOWNLOAD_META.json` |
| Note | SUMO `netconvert` needs OSM **XML**, not PBF |
| Overview PNG | `previews/moscow_net_overview.png` (full net + T/X/O markers) |

**Why BBBike (not Geofabrik CFD / Overpass)?** Ready-made *city* extract (~150 MB `.osm.gz`), already clipped near MKAD — no need for a multi‑GB Central Federal District PBF or `osmium extract`. Overpass city-wide downloads are fragile/timeout-prone. Swap later if you want a different bbox.

This replaces the old flow `sign CSV → Overpass bbox → fragment net`. Junctions
are discovered on the **full city network**, then cropped.

## How junctions are classified

| Shape | Rule |
|-------|------|
| **X** | Intersection junction with **4** incoming vehicle arms (lane length ≥ threshold) |
| **T** | Same with **3** incoming arms |
| **O** | SUMO `<roundabout nodes="…" edges="…">` blocks produced by `netconvert` |

Intersection junction types kept: `priority`, `right_before_left`, `allway_stop`,
`traffic_light` (same set as `priority_bench`).

Roundabouts are **not** guessed from geometry; only explicit SUMO roundabout
tags after `netconvert`.

## Pipeline

```bash
cd traffic-rule-bench/pdd-bench/scripts/per_sign_bench/moscow_junctions

# Full run (download if missing → netconvert → index → crop)
python scripts/run_pipeline.py

# Or step by step:
python scripts/build_net.py                 # raw/Moscow.osm.pbf → nets/moscow.net.xml
python scripts/enumerate_junctions.py       # → index/junctions.jsonl
python scripts/crop_scenes.py               # → scenes/{T,X,O}/…

# Limits for smoke tests:
python scripts/run_pipeline.py --max-per-shape 20 --skip-download
```

Useful flags on `crop_scenes.py` / `run_pipeline.py`:

- `--max-per-shape N` — cap crops per T/X/O (smoke / disk)
- `--shapes T,X` — skip roundabouts
- `--radius-m 80` — arm trim for T/X crops
- `--workers 8` — parallel `netconvert` crops (default 4)
- `--skip-existing` — do not re-crop folders that already have `map.net.xml`

## Current harvest stats

After `enumerate_junctions.py` on BBBike Moscow → `moscow.net.xml`:

| Shape | Count |
|-------|------:|
| T | 5181 |
| X | 1052 |
| O | 224 |
| **total** | **6457** |

Lat/lon in the index uses UTM zone 37 (`EPSG:32637`) + SUMO `netOffset`
(see `scripts/geo_utils.py`), not naive linear conv→orig mapping.

Full crop of all index rows:

```bash
python scripts/crop_scenes.py --skip-existing --workers 8
# progress:  ls scenes/T | wc -l ;  tail -f index/crop_full.log
```

Expect ~hours on the full city net (each crop runs `netconvert` on `moscow.net.xml`).
Smoke test with `--max-per-shape 20` first if you only need a sample.

## Relation to priority_bench

- `priority_bench` can later point `paths.scenes_dir` at
  `moscow_junctions/scenes/T` + `scenes/X` (O stays with roundabout bench).
- Layout / yield-vs-main rules are applied at **manifest** time, not at harvest.
- Legacy `sign_*` pools remain untouched.

## Dependencies

- `netconvert` (SUMO) on `PATH` or `~/.local/bin/netconvert`
- Python 3.10+ with stdlib only for these scripts (reuses
  `priority_bench.core.junction_crop` for T/X crops)
