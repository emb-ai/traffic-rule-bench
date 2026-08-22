# profiles/

nuPlan-derived driving profiles used by manifest expansion and `run_benchmark`.

| Module / asset | Role |
|----------------|------|
| `nuplan_sampler.py` | KDE / empirical draws from precomputed CSV stats |
| `nuplan_statistics/` | speeds, densities, following, routes, … |
| `agent_profile_bank.py` | NPC `sample_one_profile`, speed v0 / braking helpers |
| `ego_defaults.py` | Fixed / sampled ego IDM params (`apply_ego_defaults`, s1–s4) |
| `stable_hash.py` | Deterministic seed helper for expansions |

```python
from traffic_bench.eval.lib.profiles import sample_one_profile, apply_ego_defaults, stable_hash
from traffic_bench.eval.lib.profiles.nuplan_sampler import NuPlanSampler
```

Stats used to live at `traffic-rule-bench/nuPlan/`; they now sit next to the sampler.
