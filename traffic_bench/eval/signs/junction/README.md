# signs/junction — main, secondary, yield, stop (2.1 / 2.3 / 2.4 / 2.5)

T/X priority junctions. Shared arm geometry stays in `lib/layout/`; this
folder owns spawn enumeration and plate placement for the four eval ids.

| Eval id | Sign code | Ego arm | Plates |
| --- | --- | --- | --- |
| `main` | 2.1 | any | MainRoadSign on every approach; invisible right-hand yield for metrics |
| `secondary` | 2.3 | secondary | 2.3.x on main arms; YieldSign on secondary |
| `yield` | 2.4 | secondary | YieldSign on ego; MainRoadSign on main |
| `stop` | 2.5 | secondary | StopSign on ego; YieldSign on opposite secondary (X only) |

## Today → tomorrow

| Today | Tomorrow |
| --- | --- |
| `generate_manifest.py` (`generate_*` for 2.1 / 2.3 / 2.4 / 2.5) | `expand.py` |
| `core/manifest/manifest_expansion.py` | `expand.py` (layout × aux product) |
| `core/scenarios/scene_augmentation.py` (`equal_priority`, `yield`) | `spawn.py` |
| `run_benchmark.py` `_place_equal_priority_*`, `_place_secondary_*`, `_place_yield_*`, `_place_stop_*` | `place.py` |
| `core/layout/junction_priority_layout.py` | stays in `lib/layout/` |
| `core/layout/junction_sign_placement.py` | stays in `lib/layout/` |
