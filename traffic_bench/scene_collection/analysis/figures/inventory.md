# Harvest inventory

Sign-free crops from the BBBike Moscow OSM extract. **P** is the catalog
(`maps/index/`). **H** is cropped SUMO nets on disk (`maps/crops/`).
Per-sign quota **N** = 80 train + 20 test is sampled later (`assign`);
it is not the harvest size.

Regenerate:

```bash
python -m traffic_bench.scene_collection analysis
```

## Inventory

| Family | P (index) | H (on disk) | H / P |
| --- | --- | --- | --- |
| junction | 6457 | 6458 | 100.0% |
| dual_path | 6502 | 6507 | 100.1% |
| segment | 7620 | 7620 | 100.0% |

![Harvest inventory](inventory.png)

## Junctions (T / X / O)

| Shape | Index | On disk | Train ids | Test ids |
| --- | --- | --- | --- | --- |
| T | 5181 | 5182 | 4145 | 1036 |
| X | 1052 | 1052 | 842 | 210 |
| O | 224 | 224 | 179 | 45 |

Place identity (`junction_id` / `scene_id`) is split 80/20 *before*
allocation to signs, stratified by shape.

![Junction topology and split](junction_shapes.png)

## Dual-path atoms

Each cell is one `(baseline, compliant)` slot among {l, s, r}.
The same junction may contribute at most one atom per slot.

![Dual-path slot counts and detour gain](dual_path.png)

## Segments (corridors)

Incoming edges cropped so the scene ends 10 m before the junction.
Gates: length ≥ 150 m; **straight** chord/arc ≥ 0.99; **curved** in [0.97, 0.99).

- distinct OSM ways: 5544
- `pass_right_ok`: 2044
- `pass_left_ok`: 2044

![Segment length, straightness, lanes](segment_diversity.png)

## Geographic coverage

Points are harvest candidates (index), not the N-per-sign protocol sample.
Dual-path locations are unique parent junctions.

![Geographic coverage](geo_coverage.png)

## Example crops

One net per cell. Labels are `family/group` and `scene_id`.

![Example cropped maps](examples.png)
