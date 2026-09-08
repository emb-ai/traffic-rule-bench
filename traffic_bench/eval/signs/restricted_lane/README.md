# signs/restricted_lane — reserved lane (5.14.1, 5.14.2, 5.11.1, 5.11.2)

Multi-lane segment crops (two or more vehicle lanes, rightmost lane = SUMO
lane 0). One lane becomes the reserved lane from the plate on:

* 5.14.1 / 5.14.2 — the rightmost lane, buses / cyclists drive WITH the ego;
* 5.11.1 / 5.11.2 — a counter-flow lane on a true one-way street (the leftmost
  lane), buses / cyclists come TOWARDS the ego and turn off at the plate.

The ego spawns on that lane `approach_before_sign_m` before the plate and has to
move to a neighbouring lane before the zone: every zone step spent on the
reserved lane is a violation (`RestrictedLaneSign._is_violating`; the vehicle centre has to be
inside the reserved lane's width, which filters MetaDrive's lane-attribution
flicker at polygon seams of long curved lanes), so a
lane-keeping baseline fails the row and the rule expert, which pre-empts
40–60 m ahead of the zone (`SignComplianceMixin._handle_restricted_lane`),
passes it.

| Eval id | Sign code | Plate class | Lane | Flow | Lane users |
| --- | --- | --- | --- | --- | --- |
| `bus_lane` | 5.14.1 | `BusLaneSign` | rightmost | same | buses, 8.5 m/s |
| `bike_lane` | 5.14.2 | `BikeLaneSign` | rightmost | same | cyclists, 4.5 m/s |
| `bus_lane_road` | 5.11.1 | `BusLaneRoadSign` | leftmost | opposite | buses, 8.5 m/s |
| `bike_lane_road` | 5.11.2 | `BikeLaneRoadSign` | leftmost | opposite | cyclists, 4.5 m/s |

| File | Role |
| --- | --- |
| `expand.py` | segment scene × (world grid × family axes) → manifest rows |
| `place.py` | `RestrictedLaneSign` subclass on the reserved lane at `sign_s` (SUMO→MetaDrive remapped) with `zone_length` |
| `traffic_bench/envs/reserved_lane.py` | `ReservedLaneAgentManager`: the buses / cyclists on the lane |
| `traffic_bench/envs/lane_stream.py` | `LaneStream`: ring stream with a headway, no overtaking, gated re-entry |

## Sampling (`expand.py`)

Shared world grid (`engine/expand/world_axes.py`): traffic density probe
p25/p50/p75 × ego spawn speed 3.61/7.75/11.05 m/s × 3 NPC profile variants
(route budget is a single 220 m level: it must cover approach + zone + tail/2).
Family axes from `configs/shared/restricted_lane.yaml`:

| Axis | Levels | Nominal | Notes |
| --- | --- | --- | --- |
| `restricted_zone_levels_m` | 60 / 80 / 100 | 60 | plate at `L − (zone + tail)`, levels that do not fit the edge are skipped per map |
| `approach_levels_m` | 60 / 80 / 100 | 60 | ego spawn `approach` before the plate; admissible only when `approach ≥ lc_planning_time_s (6 s) × spawn speed` |
| `reserved_agents_n_levels` | bus 1 / 2, bicycle 1 / 2 / 3 | 1 | number of lane users |
| plate slide | hidden, ≤ `sign_jitter_m` = 15 m upstream | 0 | seeded per (scene, NPC variant) |

All combinations that fit the map are enumerated, shuffled with
`stable_hash(scene, "restricted_lane_world_cap", cap)` and the first
`max_scenarios − 1` (= 9) rows are built after the nominal row (no background
traffic, nominal geometry, one user, 5 m/s, `is_nominal`). Two runs produce
identical manifests. Row columns added on top of the shared ones:
`reserved_agents_n`, `approach_before_sign_m`, `zone_length_m`, `zone_end_s`,
`sign_s_nominal`, `zone_level_id`, `approach_level_id`, `agents_level_id`,
`corridor_*`, `edge_length_m` (full SUMO edge, used for the runtime remap).

## Lane users (`envs/reserved_lane.py`)

`reserved_agents_n` users move along the lane as a stream: one every 6 s
(buses, ≥ 51 m) or 4 s (cyclists, ≥ 18 m), ±15 % speed jitter, a faster user
closes up to one headway behind a slower one and matches its speed. Same flow:
the stream starts 25 m past the plate and re-enters there after leaving the
edge end; counter flow: the stream comes from the far end, the first user is
timed to meet a lane-keeping ego 30 m inside the zone (ego pace model from the
row's `spawn_velocity_ms`) and turns off at the plate. Re-entry is gated by a
lane-frame clearance test (cars / ego on THIS lane only), parked users wait
off-scene on separate spots. Background cars never use the reserved lane and
the ego-edge ladder starts `max(30, 3 s × spawn speed + 10)` m past the ego.

## Scenes

`data/scenes/<sign>/` (100 maps per sign, 80 train / 20 test) are built by
`python -m traffic_bench.scene_collection reserved-lane {select,crop,materialize,report}`
(`scene_collection/assign/reserved_lane.py`): multi-lane segments of the Moscow
index, split inherited from the global by-`osm_way_id` split, lane counts
balanced 2 / 3 / 4+, places not used by any official sign, crop windows
disjoint, centres ≥ 500 m apart inside the family and ≥ 300 m from other-split
official maps, one-way streets (`collect/segments/oneway.py`) for 5.11.x. After
cropping, every map's corridor (`resolve_segment_corridor`) must be ≥ 150 m; the
`all` command re-selects without the failed ids until the set is clean.
`moscow_pool.json` carries the split; `reports/restricted_lane_scenes_v3/`
the histograms and distances.

The set is fixed at 100 maps per sign (80 train / 20 test) and every map keeps
its 10 manifest rows. The eval report (`reports/restricted_lane_eval_v3/`)
labels a map GOOD when the sign-blind baselines fail it (mean compliance over
zone-reaching episodes <= 0.15) and the rule experts pass it (mean success
> 0.5), but that label is **diagnostic only**: no map is dropped from the
dataset and the GIF selection does not use it either.
