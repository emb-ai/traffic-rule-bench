# signs/

Runtime plates: zone, violation, icon. Benchmark expand/place/spawn lives in
[`eval/signs/`](../eval/signs/), not here.

| Folder | Role |
|---|---|
| `base.py` / `manager.py` / `outgoing.py` | Shared plate, manager, SUMO outgoing mixin |
| `junction/` | 2.1 / 2.3 / 2.4 / 2.5 / 4.3 |
| `dual_path/` | 4.1.x / 5.7 / 3.18 / 3.1 + PG direction |
| `speed/` | 3.24 / 4.6 / 5.21 / 5.31 + zone ends |
| `detour/` | 4.2.x plate + cone obstacle |
| `crosswalk/` | 5.19 plate + yield rule |
| `blocked/` | 3.2 |
| `extra/` | Plates still used by `envs/sumo` / compliance, not official eval ids |
| `icons/` | Top-down PNGs named after the plate (`yield.png`, `speed_limit_20.png`) |
