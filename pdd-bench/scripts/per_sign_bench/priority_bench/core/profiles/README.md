# profiles/

nuPlan-derived driving profiles used by manifest expansion and `run_benchmark`.

Vendored from colleague `factorized_space/` (+ `stable_hash` from `sumo_space`).

| Module | Role |
|--------|------|
| `agent_profile_bank.py` | NPC `sample_one_profile`, speed v0 / braking helpers |
| `ego_defaults.py` | Fixed / sampled ego IDM params (`apply_ego_defaults`, s1–s4) |
| `stable_hash.py` | Deterministic seed helper for expansions |

```python
from core.profiles import sample_one_profile, apply_ego_defaults, stable_hash
```
