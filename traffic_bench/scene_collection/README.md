# scene_collection

Shared Moscow maps, then one folder per sign under `data/scenes/<sign>/`.

`--sign yield` is the same name as in `generate_manifest.py`. YAML still uses
numeric sign codes (`2.4`, `5.19`, …).

## Folders

| | |
| --- | --- |
| [`collect/`](collect/README.md) | OSM → city net → crops |
| [`assign/`](assign/README.md) | pick maps per sign |
| [`sign_scenes/`](sign_scenes/README.md) | put them in `data/scenes/<sign>/` |
| [`maps/`](maps/README.md) | the harvested files |

## Commands

```bash
python -m traffic_bench.scene_collection collect --skip-existing
python -m traffic_bench.scene_collection assign

python -m traffic_bench.scene_collection materialize --sign yield
python -m traffic_bench.scene_collection reject --sign yield --apply --refill --loop

# crosswalk: copy corridors, then add a zebra
python -m traffic_bench.scene_collection materialize --sign crosswalk
python -m traffic_bench.scene_collection prepare --sign crosswalk

python -m traffic_bench.scene_collection review --scenes-dir data/scenes/yield
python -m traffic_bench.scene_collection pack --sign yield --out dist/yield
```

Materialize uses **relative symlinks** into `maps/crops/` (NFS-safe).
`pack` copies a sign out as a standalone folder. `--mode copy` if you want
real files in-repo.

Then eval:

```bash
python traffic_bench/eval/generate_manifest.py sign=yield paths.split=train
```

## Signs

80 train + 20 test maps per sign. Same map may serve several signs.

| `--sign` | code | maps |
| --- | --- | --- |
| `main` | 2.1 | junction |
| `secondary` | 2.3 | junction |
| `yield` | 2.4 | junction |
| `stop` | 2.5 | junction |
| `blocked_road` | 3.2 | junction |
| `roundabout` | 4.3 | junction |
| `no_entry` | 3.1 | dual_path |
| `no_turn_right` / `no_turn_left` | 3.18.1 / 3.18.2 | dual_path |
| `direction_*` | 4.1.1–4.1.6 | dual_path |
| `one_way_right` / `one_way_left` | 5.7.1 / 5.7.2 | dual_path |
| `speed_limit` / `min_speed` | 3.24 / 4.6 | segment |
| `residential_zone` / `zone_speed_limit` | 5.21 / 5.31 | segment |
| `detour_right` / `detour_left` / `detour_either` | 4.2.1–4.2.3 | segment |
| `crosswalk` | 5.19 | segment + zebra |
