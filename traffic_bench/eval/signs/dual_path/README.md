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
| `expand.py` | `generate` + dual-path × spawn lane × NPC → manifest rows |
| `place.py` | Plate on ego approach (or 3.1 exit); `resolve_row_for_policy` |
| `nav.py` | Compliant dual-path route at episode start |
