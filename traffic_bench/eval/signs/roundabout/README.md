# signs/roundabout — 4.3

Ego on a spoke (secondary); ring edges are main. Visible RoundaboutSign on
the ego spoke; invisible RoundaboutYieldSign on the conflict-arc ring.

| Eval id | Sign code |
| --- | --- |
| `roundabout` | 4.3 |

| File | Role |
| --- | --- |
| `spawn.py` | Spoke-in / ring-aux combinations + default spawn + ring meta kwargs |
| `place.py` | Plate on ego spoke + yield tracker; rebuild O-layout if missing |

Ring/spoke geometry stays in `core/layout/roundabout_topology.py` and
`roundabout_yield_zone.py`. Manifest rows still use
`signs/junction/expand.py` (same layout × aux product).
