# signs/restricted_lane — reserved lane (5.14.1, 5.14.2, 5.11.1, 5.11.2)

Same multi-lane segment crops as the detour and speed families (two or more
vehicle lanes, rightmost lane = SUMO lane 0). The plate stands on lane 0 at
`sign_s`; from there on lane 0 is reserved for `zone_length_m` metres. The ego
spawns on lane 0 `approach_before_sign_m` before the plate and has to move to a
neighbouring lane before the zone: every zone step spent on lane 0 is a
violation (`RestrictedLaneSign._is_violating`), so a lane-keeping baseline fails
the row and the rule expert, which pre-empts 50 m ahead
(`SignComplianceMixin._handle_restricted_lane`), passes it.

| Eval id | Sign code | Plate class |
| --- | --- | --- |
| `bus_lane` | 5.14.1 | `BusLaneSign` |
| `bike_lane` | 5.14.2 | `BikeLaneSign` |
| `bus_lane_road` | 5.11.1 | `BusLaneRoadSign` |
| `bike_lane_road` | 5.11.2 | `BikeLaneRoadSign` |

| File | Role |
| --- | --- |
| `expand.py` | segment scene × NPC profile → manifest rows (spawn on lane 0, plate, zone, finish past the zone) |
| `place.py` | `RestrictedLaneSign` subclass on lane 0 at `sign_s` with `zone_length` |

Scenes: `data/scenes/<id>/` are the multi-lane segment maps of the official
detour / speed scene set (80 train + 20 test per sign, split kept from the
source scene); `moscow_pool.json` carries the split.
