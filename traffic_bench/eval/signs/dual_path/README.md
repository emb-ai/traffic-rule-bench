# signs/dual_path — 4.1.x, 5.7.x, 3.18.x, 3.1

Same crop family (path-union bbox). Four near-copies of expansion + bridge +
sign spec collapse into one expand/spec parameterized by the eval id.

| Eval id | Sign code |
| --- | --- |
| `direction_straight` … `direction_left_right` | 4.1.1–4.1.6 |
| `one_way_right` / `one_way_left` | 5.7.1 / 5.7.2 |
| `no_turn_right` / `no_turn_left` | 3.18.1 / 3.18.2 |
| `no_entry` | 3.1 |

## Today → tomorrow

| Today | Tomorrow |
| --- | --- |
| `core/manifest/{one_way,direction,no_turn,no_entry}_expansion.py` | `expand.py` |
| `core/manifest/dual_path_budget.py` | `expand.py` |
| `core/scenarios/dual_path_scene.py` | `spec.py` |
| `core/scenarios/{one_way,direction,no_turn,no_entry}_bridge.py` | `spec.py` |
| `core/scenarios/{one_way,direction,no_turn,no_entry}_sign_spec.py` | `spec.py` |
| `core/scenarios/scene_augmentation.py` (`one_way`, `direction`, `no_turn`, `no_entry`) | `spawn.py` |
| `run_benchmark.py` dual-path `_place_*` | `place.py` |
