# signs/blocked — no through traffic (3.2)

Junction crop; ego must not take the forbidden through path.

| Eval id | Sign code |
| --- | --- |
| `blocked_road` | 3.2 |

## Today → tomorrow

| Today | Tomorrow |
| --- | --- |
| `core/manifest/blocked_road_expansion.py` | `expand.py` |
| `core/scenarios/scene_augmentation.py` (`blocked_road`) | `spawn.py` |
| `core/scenarios/blocked_road_route.py` | `spec.py` (forbidden-lane geometry) |
| `run_benchmark.py` blocked-road placement | `place.py` |
