# oracle/

Collect expert trajectories, pick the best run per scene, report policy-vs-oracle.


| Verb                            | Role                                                                   |
| ------------------------------- | ---------------------------------------------------------------------- |
| `[collect/](collect/README.md)` | Run policies on a train manifest → `all_runs.jsonl` + replays          |
| `[select/](select/)`            | Filter + F1 pick (`filter.py`); coverage CLI; optional complete-scenes |
| `[report/](report/)`            | Policy-vs-oracle markdown/CSV; post-hoc `oracle_rule` baseline         |


Set `SIGN=` once (same as collect). Paths default to `data/trajectories/<sign>/final`.

```bash
SIGN=crosswalk HORIZON=600 \
  ./traffic_bench/oracle/select/coverage.sh

SIGN=crosswalk HORIZON=600 \
  ./traffic_bench/oracle/report/table.sh

# or export and reuse:
export SIGN=crosswalk HORIZON=600
./traffic_bench/oracle/select/coverage.sh
./traffic_bench/oracle/report/table.sh
```

Optional overrides: `ROOT=…` / pass `OUT_BASE` as the first arg to `table.sh`.

```bash
SIGN=yield ./traffic_bench/oracle/collect/collect.sh
SIGN=yield,stop,direction/right SMOKE=1 \
  ./traffic_bench/oracle/collect/collect.sh
```

Standalone: `python -m traffic_bench.oracle.report.baseline --csv …`
and `python -m traffic_bench.oracle.select.complete`.
