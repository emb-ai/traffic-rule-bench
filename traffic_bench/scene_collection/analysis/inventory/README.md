# Harvest inventory

Sign-free cropped SUMO nets on disk (`maps/crops/`).
Per-sign quota **N** = 80 train + 20 test is sampled later (`assign`);
it is not the harvest size.

## Reproduce

```bash
python -m traffic_bench.scene_collection analysis inventory
```

Outputs: this README, `summary.json`, PNGs under [`figures/`](figures/).

## Inventory

| Family | On disk |
| --- | --- |
| junction | 6457 |
| dual_path | 6507 |
| segment | 13056 |
| total | 26020 |

Segment index (candidates to crop): **13056** ways · cropped on disk: **13056**.

![Harvest inventory](figures/inventory.png)

## Junctions (T / X / O)

| Shape | On disk | Train ids | Test ids |
| --- | --- | --- | --- |
| T | 5181 | 4145 | 1036 |
| X | 1052 | 842 | 210 |
| O | 224 | 179 | 45 |

Place identity (`junction_id` / `scene_id`) is split 80/20 *before*
allocation to signs, stratified by shape.

![Junction topology and split](figures/junction_shapes.png)

## Dual-path atoms

Each cell is one `(baseline, compliant)` slot among {l, s, r}.
The same junction may contribute at most one atom per slot.

![Dual-path slot counts and detour gain](figures/dual_path.png)

## Segments (corridors)

Full-net mid-corridor windows (`harvest: diverse_segment_v2`), not junction
approaches. Gates: length ≥ 150 m; **straight** chord/arc ≥ 0.99; **curved**
in [0.97, 0.99). One map per `osm_way_id`. Train/test split is place-disjoint
and stratified by subtype; per-sign lane/curve balance is applied at `assign`.

- distinct OSM ways (index): 13056
- by type: {'curved': 1082, 'straight': 11974}
- `pass_right_ok`: 6312
- `pass_left_ok`: 6312
- segment split totals: train 10446 / test 2610

| Subtype | Index | Train ways | Test ways |
| --- | --- | --- | --- |
| curved|1 | 782 | 626 | 156 |
| curved|2 | 225 | 180 | 45 |
| curved|3plus | 75 | 60 | 15 |
| straight|1 | 5962 | 4770 | 1192 |
| straight|2 | 3082 | 2466 | 616 |
| straight|3plus | 2930 | 2344 | 586 |

![Segment length, straightness, lanes](figures/segment_diversity.png)

## Geographic coverage

Points are cropped nets on disk (segments fall back to the index when crops
are incomplete). Dual-path locations are unique parent junctions.

![Geographic coverage](figures/geo_coverage.png)

## Example crops

One net per cell. Labels are `family/group` and `scene_id`.

![Example cropped maps](figures/examples.png)
