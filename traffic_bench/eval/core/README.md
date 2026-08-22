# `eval/core/` — compatibility shims

Implementation lives in [`../lib/`](../lib/README.md) (shared engine) and
[`../signs/`](../signs/README.md) (per-family rules).

`traffic_bench.eval.core.<subpackage>.<module>` still works. New eval code
should import `traffic_bench.eval.lib.*` or `traffic_bench.eval.signs.*`.

Family leftovers that never moved into `lib/` stay here as shims to
`signs/` (`*_expansion.py`, `*_bridge.py`, `*_sign_spec.py`,
`crosswalk_layout.py`, `blocked_road_route.py`, …).

Scene-pool bookkeeping lives in `traffic_bench.scene_collection.sign_scenes`,
not here.

```python
from traffic_bench.eval.core.sumo.lane_keys import make_lane_key
from traffic_bench.eval.core.scenarios.scene_augmentation import SpawnStrategy
from traffic_bench.eval.core.manifest.manifest_config import DEFAULT_SPAWN_DISTANCE_BEFORE_END
from traffic_bench.eval.core import SpawnScenario, build_junction_priority_layout
```
