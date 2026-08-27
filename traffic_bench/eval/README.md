# `eval/` — closed-loop evaluation

Evaluation pipeline for TrafficSignBench:

`scenes → manifest → closed-loop runs → metrics`

Scenes must be available under:

```
data/scenes/<sign>/
```

They can be generated with `scene_collection/` or downloaded from the official `emb-ai/traffic-sign-bench` dataset.

Outputs are written under:

```
data/runs/<sign>/<split>/
```

The CLI has three commands:

```
manifest → run → metrics
```

## Quick start

### 1. Build a manifest

Debug manifest (test scenes):

```
python -m traffic_bench.eval manifest sign=yield
```

*or:*

```
python -m traffic_bench.eval manifest sign=yield paths.split=test    # generate test
python -m traffic_bench.eval manifest sign=yield paths.split=train   # generate train
```

### 2. Run closed-loop evaluation

Single policy:

```
python -m traffic_bench.eval run policy=idm sign=yield
```

*or multiple policies:*

```
python -m traffic_bench.eval run \
    policies=[idm,idm_rule,plant2_ft] \
    sign=yield
```

*or all registered policies:*

```
python -m traffic_bench.eval run policies=all sign=yield
```

*or run all signs:*

```
python -m traffic_bench.eval run policies=all sign=all
```

### 4. Compute metrics

```
python -m traffic_bench.eval metrics combine sign=all
```

## Evaluation workflow

### Manifest

`manifest` discovers scenes and expands each scene into scenario rows.

Output:

```
data/runs/<sign>/<split>/
├── real_manifest.jsonl
├── config.yaml
├── repro/
└── gifs/                    # optional
```

Each manifest row contains the information required to reproduce one scenario.

Scenario augmentation is controlled by:

- `augmentation.layout` — ego/arm/lane/destination variants;
- `augmentation.auxiliary` — convoy size × occupied lanes;
- `scenario.max_scenarios` — final per-scene cap.

`scenario.max_scenarios` is applied **after** augmentation, filtering, geometry deduplication, and shuffling.

### Run

`run` loads a policy, applies a manifest row, places the required signs, and executes the scenario in closed loop.

One policy:

```
python -m traffic_bench.eval run policy=idm sign=yield
```

Several policies:

```
python -m traffic_bench.eval run \
    policies=[idm,plant2] \
    sign=yield
```

`policy=` accepts a single policy name; use `policies=[...]` for multiple policies.

`policies=all` runs the complete registered policy set.

If `manifest=` is not provided, `run` uses:

```
data/runs/<sign>/test/
```

when available, otherwise `debug/latest`.

### Metrics

Metrics operate on closed-loop episode outputs:

```
episode JSONL → per-episode CSV → aggregate/report
```

Use `metrics combine sign=all` for one overall report across signs.



## Run folders

`paths.split` controls the output folder:


| Split   | Command                                 | Purpose                                        |
| ------- | --------------------------------------- | ---------------------------------------------- |
| `debug` | `manifest sign=yield`                   | Timestamped test-scene snapshot for inspection |
| `train` | `manifest sign=yield paths.split=train` | Stable train manifest                          |
| `test`  | `manifest sign=yield paths.split=test`  | Stable evaluation manifest                     |


Debug manifests are written to:

```
data/runs/<sign>/debug/<timestamp>/
```

and `debug/latest` points to the newest snapshot.

`train/` and `test/` are immutable snapshots: remove the directory before rebuilding an existing manifest.

## Configuration

Hydra configuration is split into:

```
configs/
├── config.yaml          # manifest defaults
├── run.yaml             # closed-loop defaults
├── sign/                # sign-specific configuration
└── shared/              # configuration shared by multiple signs
```

Common overrides:

```
# Limit scenarios
python -m traffic_bench.eval manifest \
    sign=yield scenario.max_scenarios=20

# Disable auxiliary augmentation
python -m traffic_bench.eval manifest \
    sign=yield augmentation.auxiliary=false

# GIFs during manifest generation
python -m traffic_bench.eval manifest \
    sign=yield gif.enabled=true gif.max_scenes=8
```

Nested Hydra groups use paths such as:

```
sign=direction/right
sign=detour/right
```

## Signs

Sign-specific logic lives under `signs/`.


| `sign=`                            | Sign code       | Group                                    | On-disk folder     |
| ---------------------------------- | --------------- | ---------------------------------------- | ------------------ |
| `main_road`                        | 2.1             | [junction](signs/junction/README.md)     | `main_road`        |
| `secondary`                        | 2.3             | [junction](signs/junction/README.md)     | `secondary_road`   |
| `yield`                            | 2.4             | [junction](signs/junction/README.md)     | `yield`            |
| `stop`                             | 2.5             | [junction](signs/junction/README.md)     | `stop`             |
| `roundabout`                       | 4.3             | [roundabout](signs/roundabout/README.md) | `roundabout`       |
| `blocked_road`                     | 3.2             | [blocked](signs/blocked/README.md)       | `blocked_road`     |
| `no_entry`                         | 3.1             | [dual_path](signs/dual_path/README.md)   | `no_entry`         |
| `no_turn_right` / `no_turn_left`   | 3.18.1 / 3.18.2 | [dual_path](signs/dual_path/README.md)   | same as id         |
| `direction_*`                      | 4.1.1–4.1.6     | [dual_path](signs/dual_path/README.md)   | same as id         |
| `one_way_right` / `one_way_left`   | 5.7.1 / 5.7.2   | [dual_path](signs/dual_path/README.md)   | same as id         |
| `crosswalk`                        | 5.19            | [crosswalk](signs/crosswalk/README.md)   | `crosswalk`        |
| `detour_right` / `left` / `either` | 4.2.1–4.2.3     | [detour](signs/detour/README.md)         | same as id         |
| `speed_limit`                      | 3.24            | [speed](signs/speed/README.md)           | `speed_limit`      |
| `min_speed`                        | 4.6             | [speed](signs/speed/README.md)           | `min_speed`        |
| `residential_zone`                 | 5.21            | [speed](signs/speed/README.md)           | `residential_zone` |
| `zone_speed_limit`                 | 5.31            | [speed](signs/speed/README.md)           | `zone_speed_limit` |




