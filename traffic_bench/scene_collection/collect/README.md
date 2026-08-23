# collect/

Harvests the Moscow OSM map pool:

`OSM → SUMO network → index → train/test split → crops`

All generated data is written under `../maps/`.

## Quick start

### Full collection

Resume-safe: existing crops are kept and only missing scenes are generated.

```
python -m traffic_bench.scene_collection collect --skip-existing
```

Use this as the default command, especially after an interrupted run.

### Crop from an existing network

If `maps/nets/moscow.net.xml` and indexes already exist:

```
python -m traffic_bench.scene_collection collect \
    --skip-existing \
    --skip-download \
    --skip-netconvert
```

`--skip-existing` applies only to crop stages (junction, dual-path, and segment). A crop is considered complete when `map.net.xml` exists.

## Pipeline


| Step                         | Module                                      | Writes                                                  |
| ---------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| 1. Download + convert        | `build_net.py`                              | `maps/raw/`, `maps/nets/moscow.net.xml`                 |
| 2. Enumerate junctions       | `enumerate/junctions.py`                    | `maps/index/junctions.jsonl`                            |
| 3. Train / test split        | `make_split.py`                             | `maps/splits/train_ids.json`, `test_ids.json`           |
| 4. Crop junctions            | `junctions/crop.py`                         | `maps/crops/junction/{T,X,O}/<id>/`                     |
| 5. Crop dual-path            | `dual_path/crop.py`                         | `maps/crops/dual_path/{T,X}/<slot>/<id>/`               |
| 6. Enumerate + crop segments | `enumerate/segments.py`, `segments/crop.py` | `maps/index/segments.jsonl`, `maps/crops/segment/<id>/` |


`--skip-existing` is passed only to crop steps (4–6). It skips a scene when `map.net.xml` is already on disk. Download / netconvert / enumerate / split still run unless you pass the matching `--skip-*` flags.

## Useful flags


| Flag                         | Effect                                         |
| ---------------------------- | ---------------------------------------------- |
| `--skip-existing`            | Do not re-crop scenes that already exist       |
| `--skip-download`            | Reuse OSM under `maps/raw/`                    |
| `--skip-netconvert`          | Reuse `maps/nets/moscow.net.xml`               |
| `--skip-enumerate`           | Skip junction (and segment) enumeration        |
| `--skip-split`               | Skip train/test id files                       |
| `--skip-crop`                | Skip junction crops                            |
| `--skip-dual-path`           | Skip dual_path crops                           |
| `--skip-segment`             | Skip segment enumerate + crops                 |
| `--workers N`                | Parallel workers for junction crop (default 8) |
| `--shapes T,X,O`             | Junction shapes to enumerate/crop              |
| `--max-per-shape N`          | Cap junction crops per shape                   |
| `--dual-path-max-per-slot N` | Cap dual_path per slot (`0` = no cap)          |


## Crop only one family

```bash
# junctions only
python -m traffic_bench.scene_collection collect \
  --skip-existing --skip-download --skip-netconvert \
  --skip-dual-path --skip-segment

# dual_path only
python -m traffic_bench.scene_collection collect \
  --skip-existing --skip-download --skip-netconvert \
  --skip-crop --skip-segment

# segments only
python -m traffic_bench.scene_collection collect \
  --skip-existing --skip-download --skip-netconvert \
  --skip-crop --skip-dual-path
```



