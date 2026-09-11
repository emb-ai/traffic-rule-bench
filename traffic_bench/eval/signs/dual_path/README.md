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
| `route_probe.py` | Headless MetaDrive loop check (baseline + compliant dest) |

## Filling train/test map quotas

Eval `ok=false` on dual-path is usually MetaDrive `[RouteValidation] INVALID: route loops back to spawn`, not SUMO. Official `reject --apply --refill --loop` only runs SUMO viability, so looping maps stay live.

`materialize --refill` never redraws a **live** directory. If looping maps stay in `data/scenes/...` (dry-run / `--no-apply`), refill cannot replace them and the same reject list repeats.

Move looping dirs, then refill:

```bash
# One command: probe → move to _rejected/ → materialize --refill → repeat
# until train=80 / test=20 and every live map is routable.
python -m traffic_bench.eval.signs.dual_path.route_probe \
  --sign no_turn/left --loop

# Direction 4.1.1–4.1.6 (all six), or one plate:
python -m traffic_bench.eval.signs.dual_path.route_probe \
  --sign direction --loop
python -m traffic_bench.eval.signs.dual_path.route_probe \
  --sign direction/straight --loop

# One-way 5.7.1 / 5.7.2:
python -m traffic_bench.eval.signs.dual_path.route_probe \
  --sign one_way --loop
python -m traffic_bench.eval.signs.dual_path.route_probe \
  --sign one_way/right --loop

# Dry-run only (does not write selection or move dirs):
python -m traffic_bench.eval.signs.dual_path.route_probe \
  --sign no_turn/left --no-apply
```

Then rebuild manifests (`rm -rf data/runs/<sign>/<split>` first if the folder already exists).

`--loop` only checks the crop dest. Expand may still drop a map after route-budget trim (`no_expand_dests`). Manifest generation now rejects those, refills new maps, and caps **by unique map** (`n_test` × `max_scenarios`), so leftover combos cannot pad 17 maps up to 200 rows.
