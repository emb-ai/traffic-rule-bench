# signs/roundabout — 4.3

Ego on a spoke (secondary); ring edges are main. Visible RoundaboutSign on
the ego spoke; invisible RoundaboutYieldSign on the conflict-arc ring.

| Eval id | Sign code |
| --- | --- |
| `roundabout` | 4.3 |

## Today → tomorrow

| Today | Tomorrow |
| --- | --- |
| `generate_manifest.py` (roundabout branch) | `expand.py` |
| `core/scenarios/scene_augmentation.py` (`roundabout`) | `spawn.py` |
| `core/scenarios/roundabout_aux.py` | `spawn.py` / `place.py` |
| `core/layout/roundabout_topology.py` | `spec.py` (ring + spoke) |
| `core/layout/roundabout_yield_zone.py` | `place.py` (entry conflict arcs) |
| `run_benchmark.py` `_place_roundabout_*` | `place.py` |
