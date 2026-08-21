# finetune/

Fine-tune PlanT2 on oracle/expert trajectories from
`traffic_bench/oracle/collect_trajectories` (`replay.pkl` + sidecar).

Not part of eval / manifest / oracle-selection.

| File | Role |
|---|---|
| `expert_replay_inenv.py` | Replay recorded pkl in-env; optional `--save-plant2-dir` feature dump |
| `plant2_frames.py` | BEV / boxes / measurements writer for PlanTDataset |
| `env_flags.py` | `RELOCATE_EGO_TO_SIGN_LANE` |

Typical input: picks from `traffic_bench/oracle` (`experts_scene_uid_top1.jsonl`)
+ scenes under `data/<sign>/scenes`.

```bash
python finetune/expert_replay_inenv.py \
    --experts /path/to/experts_scene_uid_top1.jsonl \
    --scenes-root data/<sign>/scenes \
    --save-plant2-dir ./plant2_out \
    --count 5 --save-gifs
```
