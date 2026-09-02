# assign/

Reads `maps/splits/signs.yaml` plus global train/test place splits and writes
`maps/splits/sign_allocations.json` using **tiered place reuse within each split**.

```bash
python -m traffic_bench.scene_collection collect   # includes make_split
python -m traffic_bench.scene_collection assign
```

## Policy

1. **Train/test** — place-disjoint (`junction_id` / `osm_way_id`), stratified by
   topology (T/X/O for junctions; straight/curved for segments).
2. **Sign order** — taxonomy order (roundabout → priority → speed → obstacle → reroute).
3. **Per pick** (within train or test):
   - tier 1 — unused physical place in this split
   - tier 2 — same behavioral family
   - tier 3 — same semantic group, different behavioral family
   - cross-semantic reuse → shortfall error

Behavioral families and compatible topologies live in `taxonomy.py`.
