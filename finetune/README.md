# plant2_finetune/

Отдельная ветка: finetune PlanT-2 на oracle/expert траекториях из
`priority_bench/collect_trajectories` (replay.pkl + sidecar).

Не часть eval / manifest / oracle-selection. Сюда кладём только
инструменты датасета и обучения PlanT-2.

| File | Role |
|---|---|
| `expert_replay_inenv.py` | Replay recorded pkl in-env; optional `--save-plant2-dir` feature dump |
| `plant2_frames.py` | BEV / boxes / measurements writer for PlanTDataset |
| `env_flags.py` | `RELOCATE_EGO_TO_SIGN_LANE` (was `bench.env_builders`) |

Typical input: picks from `priority_bench/oracle` / `collect_trajectories`
(`experts_scene_uid_top1.jsonl`) + scenes under `priority_bench/data/<sign>/`.

```bash
cd plant2_finetune
python expert_replay_inenv.py \
    --experts /path/to/experts_scene_uid_top1.jsonl \
    --scenes-root ../sign_bench/data/<sign>/scenes \
    --save-plant2-dir ./plant2_out \
    --count 5 --save-gifs
```

Note: `_build_env` still expects a colleague-era `expert_replay` module; wire
env construction to `priority_bench` when starting the FT pipeline in earnest.
Env builders should honor `env_flags.RELOCATE_EGO_TO_SIGN_LANE`.
