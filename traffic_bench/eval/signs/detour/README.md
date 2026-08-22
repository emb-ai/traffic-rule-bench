# signs/detour — obstacle keep-right / keep-left / either (4.2.x)

Same segment crops as speed signs. Eval picks the obstacle lane from
`vehicle_lane_indices` plus `pass_right_ok` / `pass_left_ok`.

| Eval id | Sign code |
| --- | --- |
| `detour_right` | 4.2.1 |
| `detour_left` | 4.2.2 |
| `detour_either` | 4.2.3 |

## Today → tomorrow

| Today | Tomorrow |
| --- | --- |
| `core/manifest/detour_expansion.py` | `expand.py` |
| `run_benchmark.py` detour placement | `place.py` |
