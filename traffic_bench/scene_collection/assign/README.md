# assign/

Reads `maps/splits/signs.yaml` plus global train/test place splits and writes
`maps/splits/sign_allocations.json` using **tiered place reuse within each split**.

```bash
python -m traffic_bench.scene_collection collect   # includes make_split
python -m traffic_bench.scene_collection assign
```

## Policy

1. **Train/test** — place-disjoint (`junction_id` / `osm_way_id`), stratified by
   topology (T/X/O for junctions; `(straight|curved)×(1|2|3plus)` subtypes for
   segments). Per-sign quotas then balance those subtypes within each sign’s query.
2. **Sign order** — taxonomy order (roundabout → priority → speed → obstacle → reroute).
3. **Per pick** (within train or test):
   - tier 1 — unused physical place in this split
   - tier 2 — same behavioral family
   - tier 3 — same semantic group, different behavioral family
   - cross-semantic reuse → shortfall error

Behavioral families and compatible topologies live in `taxonomy.py`.

## Refill (`materialize --refill` / `reject --refill --loop`)

Same tiered pick as assign. For **segments**, refill slots go to subtypes that are
below the per-sign half quota first (`kept` vs target), not as a fresh even split
of only the missing headcount. If a rare subtype is exhausted, leftover need is
redistributed to still-open buckets.
