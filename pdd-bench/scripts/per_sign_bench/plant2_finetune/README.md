# plant2_finetune/

Отдельная ветка: finetune PlanT-2 на oracle/expert траекториях из
`priority_bench/collect_trajectories` (replay.pkl + sidecar).

Не часть eval / manifest / oracle-selection. Сюда кладём только
инструменты датасета и обучения PlanT-2.

| File | Role |
|---|---|
| `expert_replay_inenv.py` | Replay recorded pkl in-env; optional `--save-plant2-dir` feature dump |

Typical input: picks from `priority_bench/oracle` / `collect_trajectories`
(`experts_scene_uid_top1.jsonl`) + scenes under `priority_bench/data/<sign>/`.

```bash
cd scripts/per_sign_bench/plant2_finetune
python expert_replay_inenv.py \
    --experts /path/to/experts_scene_uid_top1.jsonl \
    --scenes-root ../priority_bench/data/<sign>/scenes \
    --save-plant2-dir ./plant2_out \
    --count 5 --save-gifs
```

Note: this script still assumes colleague-era env builders in places; wire it
to `priority_bench` when you start the FT pipeline in earnest.
