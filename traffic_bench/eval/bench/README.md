# bench/ — one policy, one manifest, closed-loop episodes

Shared runner. Not multi-policy orchestration (`pipeline/`) and not CSV
rollups (`metrics/`).

What belongs here (today inside [`../run_benchmark.py`](../run_benchmark.py)):

- wrap the MetaDrive / SUMO env
- load IDM / PPO / CaRL / PlanT2
- apply a manifest row (spawn lane, destination, aux convoy)
- call `signs/<family>/place.py` to put plates in the world
- step the episode; record violations, crash, route completion
- optional top-down GIF
- write `episodes_<policy>.jsonl` and `replay.json`

What does **not** belong here: `_place_yield_*` / `_place_stop_*` / other
per-family plate logic — those go to `signs/<family>/place.py`.

Command: `python -m traffic_bench.eval run --policy idm --manifest data/runs/yield/<ts>/real_manifest.jsonl`.
