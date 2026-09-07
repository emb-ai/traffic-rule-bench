# signs/roundabout — 4.3

Ego on a spoke (secondary); ring edges are main. Visible RoundaboutSign on
the ego spoke; invisible RoundaboutYieldSign on the conflict-arc ring.

Yield semantics (4.3 tracker):
- Arm when aux is in the left entry-conflict arc (presence only; no path cross).
- Conflict lanes = topological ring edges into ego's entry (plus 2 upstream hops)
  **union** ring lanes that approach the geometric spoke/ring meeting XY.
- Conflict point = mean end of **entry** ring lanes (immediate incoming), not the
  spoke tip and not the mean of upstream-hop ends.
- Longitudinal window is centered on the closest point of each lane to that
  conflict XY (so long/split edges still count mid-lane).
- Release when that aux has passed the conflict XY.

| Eval id | Sign code |
| --- | --- |
| `roundabout` | 4.3 |

| File | Role |
| --- | --- |
| `spawn.py` | Spoke-in / ring-aux combinations + default spawn + ring meta kwargs |
| `place.py` | Plate on ego spoke + yield tracker; rebuild O-layout if missing |
| `aux.py` | Ring-chain aux placement |

Ring/spoke geometry stays in `engine/map/roundabout_topology.py` and
`roundabout_yield_zone.py`. Manifest rows use `signs/junction/expand.generate`
(same layout × aux product).
