# engine/traffic/ — who drives

nuPlan-derived driving profiles used by manifest expansion and closed-loop
`run`.

```python
from traffic_bench.eval.engine.traffic.agent_profile_bank import sample_one_profile
from traffic_bench.eval.engine.traffic.ego_defaults import apply_ego_defaults
from traffic_bench.eval.engine.traffic.stable_hash import stable_hash
from traffic_bench.eval.engine.traffic.nuplan_sampler import NuPlanSampler
```

CSV tables live in `nuplan_statistics/`.
