# maps/

Data only. Code is in `collect/`, `assign/`, `sign_scenes/`.

| | |
| --- | --- |
| [`raw/`](raw/README.md) | OSM extract |
| [`nets/`](nets/README.md) | city SUMO net |
| [`index/`](index/README.md) | place catalogs |
| `crops/` | cropped scenes (gitignored) |
| [`splits/`](splits/README.md) | quotas, train/test, allocations |
| [`previews/`](previews/README.md) | overview PNG |

```
crops/junction/{T,X,O}/<id>/
crops/dual_path/{T,X}/{slot}/<id>/
crops/segment/<id>/
```

Junction and dual-path keys: `junction_id`. Segments: `osm_way_id`.
`straight` / `curved` are tags in `meta.json`, not folders.
