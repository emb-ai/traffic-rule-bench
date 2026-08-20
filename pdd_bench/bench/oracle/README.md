# oracle/

Shared expert-selection logic and reporting for train oracle picks.

| File | Role |
|---|---|
| `select_experts.py` | Library: compliance filter, F1, per-scene pick (+ CLI) |
| `make_oracle_metrics_table.py` | Policy-vs-oracle markdown/CSV after coverage |
| `select_experts_complete_scenes.py` | Optional: oracle only on scenes covered by all experts |

Selection CLI for your layout stays in  
`collect_trajectories/select_experts_coverage.py` (imports this package).

Orchestration: `collect_trajectories/make_oracle_table.sh` → `make_oracle_metrics_table.py`.
