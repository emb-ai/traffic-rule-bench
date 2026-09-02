# Scene Collection

Scene collection pipeline for **TrafficSignBench**. It crops road scenes from Moscow OpenStreetMap (OSM) data, assigns them to sign-specific pools, and materializes the official benchmark scenes.

**Pipeline:**

`OSM → collect → assign → materialize → review/reject → pack → publish`

## Quick start

Run all commands from the repository root (`traffic-rule-bench`).

```
# 1. Harvest Moscow maps: net + index + train/test split + crops
python -m traffic_bench.scene_collection collect --skip-existing

# 2. Allocate harvested maps to signs
python -m traffic_bench.scene_collection assign

# 3. Materialize scenes for all signs
python -m traffic_bench.scene_collection materialize --all
```

*Optionally:*

```
# 4. Review and reject scenes
python -m traffic_bench.scene_collection review --scenes-dir data/scenes/yield
python -m traffic_bench.scene_collection reject --sign yield --apply --refill --loop

# 5. Pack scenes and optional upload Hugging Face
python -m traffic_bench.scene_collection pack --all
python -m traffic_bench.scene_collection publish

# 6. Generate statistics and diversity figures
python -m traffic_bench.scene_collection analysis
python -m traffic_bench.scene_collection analysis overlap   # cross-sign map reuse + train/test leakage
```

*Prebuilt official scenes are available on Hugging Face:*

```
huggingface-cli download emb-ai/traffic-sign-bench \
    --repo-type dataset \
    --local-dir data
```

## Details

### 1. Collect

`collect` builds the shared map pool:

- download and convert OSM data;
- build the city network and index;
- create the train/test split;
- crop junction, dual-path, and segment maps.

Useful stage controls:

```
--skip-download
--skip-netconvert
--skip-enumerate
--skip-split
--skip-crop
--skip-dual-path
--skip-segment
```

### 2. Assign

`assign` queries the harvested pool using `maps/splits/signs.yaml` and creates:

```
maps/splits/sign_allocations.json
```

Allocation is **tiered by physical place** within each split (train/test separately):

1. unused place in this split
2. place already used by the same behavioral family (e.g. 4.1.1–4.1.6)
3. place used elsewhere in the same semantic group (priority / speed / obstacle / reroute)

Cross-semantic reuse is rejected. Signs are processed in taxonomy order so early
signs get unique places first. Topology quotas (T/X 50/50, segment straight/curved)
are preserved per sign.

CLI `--sign` and Hydra `sign=` use **eval profile IDs** (`yield`, `roundabout`, `crosswalk`, ...), matching `python -m traffic_bench.eval manifest`.

### 3. Materialize

`materialize` places allocated maps into:

```
data/scenes/<sign>/
```

The default mode creates **relative symlinks** into `maps/crops/`. This keeps the materialized dataset lightweight and survives remounting the NFS share.

Useful stage controls:

```
--skip-download   # for a single sign
```

Signs with a `prepare:` hook are automatically prepared after materialization. Currently this applies to `crosswalk` (5.19).

## Data layout

```
maps/
├── crops/
│   ├── junction/{T,X,O}/
│   ├── dual_path/{T,X}/{slot}/
│   └── segment/<scene_id>/
└── splits/
    ├── signs.yaml
    └── sign_allocations.json

data/
└── scenes/
    ├── yield/
    ├── stop/
    ├── crosswalk/
    └── ...
```

`straight`, `curved`, and lane count are **metadata tags** stored in `meta.json`; they are not separate crop families.

Shared path constants are defined in `paths.py`. `preview.py` generates a top-down PNG from a cropped SUMO network.

## Dataset scale


| Symbol | Meaning                             | This iteration                                          |
| ------ | ----------------------------------- | ------------------------------------------------------- |
| **P**  | Full Moscow population in the index | junctions 6457 (T 5181 / X 1052 / O 224); segments 7620 |
| **H**  | Cropped nets on disk                | **H = P** (crop until the city runs out)                |
| **N**  | Official maps per sign              | 80 train + 20 test                                      |


The independent benchmark unit is a **map**. Scenario augmentations (≤10 per map) are correlated. The train/test split is assigned by **place identity before sign allocation**: 1) junctions / dual-path: `junction_id`, and 2) segments: `osm_way_id`. Thus, the same street cannot appear in both train and test.

The `50/50` T/X split in `signs.yaml` ensures balanced sampling across T- and X-junction topologies, rather than reflecting Moscow's natural ~83% T-junction distribution.

## Sign mapping


| `--sign` / `sign=`         | Sign ID (Moscow Traffic Rules) | Family                             |
| -------------------------- | ------------------------------ | ---------------------------------- |
| `main`                     | 2.1                            | junction                           |
| `secondary`                | 2.3                            | junction                           |
| `yield`                    | 2.4                            | junction                           |
| `stop`                     | 2.5                            | junction                           |
| `blocked_road`             | 3.2                            | junction                           |
| `roundabout`               | 4.3                            | junction                           |
| `no_entry`                 | 3.1                            | dual_path                          |
| `no_turn_right`            | 3.18.1                         | dual_path                          |
| `no_turn_left`             | 3.18.2                         | dual_path                          |
| `direction_straight`       | 4.1.1                          | dual_path                          |
| `direction_right`          | 4.1.2                          | dual_path                          |
| `direction_left`           | 4.1.3                          | dual_path                          |
| `direction_straight_right` | 4.1.4                          | dual_path                          |
| `direction_straight_left`  | 4.1.5                          | dual_path                          |
| `direction_left_right`     | 4.1.6                          | dual_path                          |
| `one_way_right`            | 5.7.1                          | dual_path                          |
| `one_way_left`             | 5.7.2                          | dual_path                          |
| `speed_limit`              | 3.24                           | segment                            |
| `min_speed`                | 4.6                            | segment                            |
| `residential_zone`         | 5.21                           | segment                            |
| `zone_speed_limit`         | 5.31                           | segment                            |
| `detour_right`             | 4.2.1                          | segment                            |
| `detour_left`              | 4.2.2                          | segment                            |
| `detour_either`            | 4.2.3                          | segment                            |
| `crosswalk`                | 5.19                           | segment + `prepare` zebra (middle) |


