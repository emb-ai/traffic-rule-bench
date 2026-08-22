# signs/detour — obstacle keep-right / keep-left / either (4.2.x)

Same segment crops as speed signs. Eval picks the obstacle lane from
`vehicle_lane_indices` plus `pass_right_ok` / `pass_left_ok`.

| Eval id | Sign code |
| --- | --- |
| `detour_right` | 4.2.1 |
| `detour_left` | 4.2.2 |
| `detour_either` | 4.2.3 |

| File | Role |
| --- | --- |
| `expand.py` | corridor × density → manifest rows |
| `place.py` | DetourSign on the obstacle lane at `sign_s` |

`expand.generate` writes the detour manifest.
