# collect/

OSM → city SUMO net → index → train/test split → crops under [`../maps/`](../maps/README.md).

## Usual command

```bash
# Resume-safe: skip crops that already have map.net.xml
python -m traffic_bench.scene_collection collect --skip-existing
```

Use this when a previous run was interrupted, or when you are not sure
junction / dual_path / segment crops finished. Existing scenes are kept;
only missing ones are cropped.

If the city net and indexes are already there and you only want to
continue cropping (faster):

```bash
python -m traffic_bench.scene_collection collect \
  --skip-existing \
  --skip-download \
  --skip-netconvert
```

## Pipeline steps

| Step | Module | Writes |
| --- | --- | --- |
| 1. Download + convert | `build_net.py` | `maps/raw/`, `maps/nets/moscow.net.xml` |
| 2. Enumerate junctions | `enumerate/junctions.py` | `maps/index/junctions.jsonl` |
| 3. Train / test split | `make_split.py` | `maps/splits/train_ids.json`, `test_ids.json` |
| 4. Crop junctions | `junctions/crop.py` | `maps/crops/junction/{T,X,O}/<id>/` |
| 5. Crop dual-path | `dual_path/crop.py` | `maps/crops/dual_path/{T,X}/<slot>/<id>/` |
| 6. Enumerate + crop segments | `enumerate/segments.py`, `segments/crop.py` | `maps/index/segments.jsonl`, `maps/crops/segment/<id>/` |

`--skip-existing` is passed only to crop steps (4–6). It skips a scene
when `map.net.xml` is already on disk. Download / netconvert / enumerate /
split still run unless you pass the matching `--skip-*` flags.

## Useful flags

| Flag | Effect |
| --- | --- |
| `--skip-existing` | Do not re-crop scenes that already exist |
| `--skip-download` | Reuse OSM under `maps/raw/` |
| `--skip-netconvert` | Reuse `maps/nets/moscow.net.xml` |
| `--skip-enumerate` | Skip junction (and segment) enumeration |
| `--skip-split` | Skip train/test id files |
| `--skip-crop` | Skip junction crops |
| `--skip-dual-path` | Skip dual_path crops |
| `--skip-segment` | Skip segment enumerate + crops |
| `--workers N` | Parallel workers for junction crop (default 8) |
| `--shapes T,X,O` | Junction shapes to enumerate/crop |
| `--max-per-shape N` | Cap junction crops per shape |
| `--dual-path-max-per-slot N` | Cap dual_path per slot (`0` = no cap) |

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

## Layout of code

| | |
| --- | --- |
| `build_net.py` | download OSM, convert to SUMO |
| `make_split.py` | train / test on junction ids |
| [`enumerate/`](enumerate/README.md) | list junctions and corridors |
| [`junctions/`](junctions/README.md) | crop T / X / O |
| [`dual_path/`](dual_path/README.md) | crop two-route nets |
| [`segments/`](segments/README.md) | crop corridors |
| [`lib/`](lib/README.md) | shared helpers |
