# signs/junction — main, secondary, yield, stop (2.1 / 2.3 / 2.4 / 2.5)

T/X priority junctions. Shared arm geometry stays in `core/layout/`; this
folder owns plate placement for the four eval ids.

| Eval id | Sign code | Ego arm | Plates |
| --- | --- | --- | --- |
| `main` | 2.1 | any | MainRoadSign on every approach; invisible right-hand yield for metrics |
| `secondary` | 2.3 | secondary | 2.3.x on main arms; YieldSign on secondary |
| `yield` | 2.4 | secondary | YieldSign on ego; MainRoadSign on main |
| `stop` | 2.5 | secondary | StopSign on ego; YieldSign on opposite secondary (X only) |

| File | Role |
| --- | --- |
| `expand.py` | Layout × aux cartesian product + row builder (also used by 4.3) |
| `spawn.py` | Equal-priority / yield ego×aux combinations + default spawn |
| `place.py` | Where plates go on main vs secondary arms |

`generate_manifest()` still discovers scenes and writes jsonl; it calls
`expand_scene_entries` here. Shared lane-parse helpers stay in
`core/scenarios/scene_augmentation.py`.
