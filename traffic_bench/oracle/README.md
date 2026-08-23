# oracle/

Collect expert trajectories, pick the best run per scene, report policy-vs-oracle.

| Verb | Role |
|---|---|
| [`collect/`](collect/README.md) | Run policies on a train manifest → `all_runs.jsonl` + replays |
| [`select/`](select/) | Filter + F1 pick (`filter.py`); coverage CLI; optional complete-scenes |
| [`report/`](report/) | Policy-vs-oracle markdown/CSV; post-hoc `oracle_rule` baseline |

```bash
SIGN=yield MANIFEST=data/runs/yield/train/real_manifest.jsonl \
  ./traffic_bench/oracle/collect/collect.sh

python -m traffic_bench.oracle.select.coverage \
    --root data/trajectories/yield/trajectories_<ts> \
    --catalog data/trajectories/yield/trajectories_<ts>/catalog.jsonl \
    --signs 2.4 --horizon 1500 \
    --out-dir data/trajectories/yield/trajectories_<ts>/experts

SIGN=yield ./traffic_bench/oracle/report/table.sh \
    data/trajectories/yield/trajectories_<ts>
```

Standalone: `python -m traffic_bench.oracle.report.baseline --csv …`
and `python -m traffic_bench.oracle.select.complete`.
