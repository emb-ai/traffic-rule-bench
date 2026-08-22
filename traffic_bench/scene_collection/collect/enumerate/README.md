# collect/enumerate/ — index places in the city net

Writes JSONL indexes under `maps/index/`. No cropped nets yet.

- `junctions.py` — T / X / O junctions → `junctions.jsonl`
- `segments.py` — long incoming edges → `segments.jsonl` (split later by `osm_way_id`)
