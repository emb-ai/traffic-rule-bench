# signs/dual_path — 4.1.x, 5.7.x, 3.18.x, 3.1

Same crop family (path-union bbox). One spec table and one expander,
parameterized by sign code.

| Eval id | Sign code |
| --- | --- |
| `direction_straight` … `direction_left_right` | 4.1.1–4.1.6 |
| `one_way_right` / `one_way_left` | 5.7.1 / 5.7.2 |
| `no_turn_right` / `no_turn_left` | 3.18.1 / 3.18.2 |
| `no_entry` | 3.1 |

| File | Role |
| --- | --- |
| `spec.py` | Plate table, role dirs, crop-meta discover |
| `scene.py` | `DualPathScenario` from `meta.json` |
| `budget.py` | Truncate both routes to a shared meter budget |
| `expand.py` | dual-path × spawn lane × NPC → manifest rows |

Old imports (`core.manifest.one_way_expansion`, `core.scenarios.one_way_bridge`, …)
are shims. `generate_manifest` / `run_benchmark` still call those names.
`place.py` is still inside `run_benchmark.py`.
