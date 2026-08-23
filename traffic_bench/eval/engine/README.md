# `engine/`

Shared evaluation components used across sign families. **No sign-specific rules live here.**


| Path                   | Role                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `[map/](map/)`         | Map representation: SUMO keys, T/X arms, roundabout topology                                     |
| `[traffic/](traffic/)` | Traffic sources and profiles: nuPlan / IDM / density                                             |
| `[spawn/](spawn/)`     | Shared spawn types and utilities; sign-specific dispatch lives in `signs/<group>/spawn.py`       |
| `[expand/](expand/)`   | Shared scenario row/augmentation types; sign-specific builders live in `signs/<group>/expand.py` |
| `[sim/](sim/)`         | MetaDrive runtime, checkpoints, HUD/GIF patches                                                  |


Sign-specific runtime stays with the corresponding sign family:

```text
signs/dual_path/nav.py
signs/roundabout/aux.py
```

