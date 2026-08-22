# bench/ — one policy, one manifest, closed-loop episodes

Shared runner. Not multi-policy orchestration (`pipeline/`) and not CSV
rollups (`metrics/`).

| File | Role |
| --- | --- |
| `episode.py` | wrap env, load policy, apply row, step, GIF, write episodes |
| `place.py` | dispatch to `signs/<family>/place.py` |

CLI still lives in [`../run_benchmark.py`](../run_benchmark.py)
(`python -m traffic_bench.eval run …`). Oracle and
`python -m traffic_bench.eval.run_benchmark` keep working: they import
`run_one_episode` from that module.

What does **not** belong here: per-family plate logic — that stays in
`signs/<family>/place.py`.
