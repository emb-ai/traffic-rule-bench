# pipeline/ — many policies + report

Today: [`../eval_pipeline.py`](../eval_pipeline.py).

Loops policies (IDM × ego variants s1–s4, then NN), invokes `bench/`, writes
`eval_out/`, then runs `metrics/` (CSV → aggregations → markdown).

`--manifest` can be a run folder (`<folder>/real_manifest.jsonl` →
`<folder>/eval_out/`) or one or more `.jsonl` files.

Command: `python -m traffic_bench.eval pipeline --policies idm --manifest data/runs/yield/<ts>`.
