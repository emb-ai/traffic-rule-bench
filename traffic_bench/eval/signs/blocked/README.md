# signs/blocked — no through traffic (3.2)

Junction crop; ego must not take the forbidden through path.

| Eval id | Sign code |
| --- | --- |
| `blocked_road` | 3.2 |

| File | Role |
| --- | --- |
| `spec.py` | Forbidden-lane length check (sign + dest cap) |
| `spawn.py` | Ego approach × forbidden-exit combinations |
| `expand.py` | through-path × NPC → manifest rows |
| `place.py` | `NoTrafficSign` at the start of the forbidden exit |

`generate_blocked_road_manifest` lives in `manifest/run.py`.
Old imports (`core.manifest.blocked_road_expansion`,
`core.scenarios.blocked_road_route`) are shims.
