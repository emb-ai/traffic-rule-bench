# oracle/

Collect expert trajectories, pick the best run per scene, report policy-vs-oracle.


| Verb                            | Role                                                                   |
| ------------------------------- | ---------------------------------------------------------------------- |
| `[collect/](collect/README.md)` | Run policies on a train manifest → `all_runs.jsonl` + replays          |
| `[select/](select/)`            | Filter + F1 pick (`filter.py`); coverage CLI; optional complete-scenes |
| `[report/](report/)`            | Policy-vs-oracle markdown/CSV; post-hoc `oracle_rule` baseline         |


```bash
SIGN=yield ./traffic_bench/oracle/collect/collect.sh
SIGN=yield,stop,direction/right SMOKE=1 \
  ./traffic_bench/oracle/collect/collect.sh

python -m traffic_bench.oracle.select.coverage \
    --root data/trajectories/crosswalk/final \
    --catalog data/trajectories/crosswalk/final/catalog.jsonl \
    --signs crosswalk --horizon 600 \
    --out-dir data/trajectories/crosswalk/final/experts

SIGN=crosswalk report/table.sh \
    data/trajectories/crosswalk/final
```

Standalone: `python -m traffic_bench.oracle.report.baseline --csv …`
and `python -m traffic_bench.oracle.select.complete`.