# signs/roundabout — 4.3

Ego on a spoke (secondary); ring edges are main. Visible RoundaboutSign on
the ego spoke; invisible RoundaboutYieldSign on the conflict-arc ring.

Yield semantics (4.3 tracker):
- At episode start, lock the **nearest** main conflict **edge** to ego (head XY
  of each candidate; pick once, never change). All **parallel lanes** on that
  edge are included in the yellow zone.
- Agents in the ego **yield zone** are ignored (no meet / not conflicting).
- Track every aux/NPC inside that locked edge; magenta heading ray **only**
  while they are geometrically on that edge (strict road-id match).
- Aux with an approaching ego×aux ray meet → block (expert waits / violations
  latch).
- The **first** time the meet disappears → drop that foe immediately (no hold,
  no post-exit tracking). No re-arm until the foe leaves the arc.

Debug GIFs (`gif.draw_path_conflict`, on by default for roundabout):
- **green** ribbon = ego yield zone
- **yellow** ribbon = locked nearest conflict edge (all parallel lanes)
- cyan = ego heading ray (always; not a conflict track)
- magenta = foe ray only while on the locked main edge and not yet released
- **yellow** X = approaching ray meet while blocking

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

Background NPCs (traffic density): never spawn on the **circular ring**
(``main_edge_ids`` / SUMO ``<roundabout>`` edges). Spoke arms remain allowed.
Aux conflict agents are unchanged and still place on the conflict arc.
