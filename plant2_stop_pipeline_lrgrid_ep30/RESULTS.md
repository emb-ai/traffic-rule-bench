# PlanT STOP LR grid — 30 epochs

Train dump/split: `plant2_stop_pipeline_signfix/plant2_l1_stop_split/` (294/50).
Test: `stop_data/output/ts_test/real_manifest.jsonl` + `stop_data/scenes/`.
Resume base: `stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`.

Driver updates this file after each `(lr, ckpt_kind)` eval completes.

Parallel mode: trains run concurrently (one GPU each); evals start as each
train finishes (eval GPU pool; lean `--jobs` when multiple evals overlap).

## Baseline (prior, not re-run)

| tag | lr | epochs | ckpt | n | success | sign_compliance | efficiency | driving_score |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `baseline_ep20_lr3e4_last` | 3e-4 | 20 | last | 42 | 0.714 | 0.548 | 78.109 | 0.000 |

## LR grid results

| tag | lr | epochs | ckpt_kind | best_epoch | n | success | sign_compliance | efficiency | driving_score | ckpt |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `lr1e4_last` | 1e-4 | 30 | last | 3 | 42 | 0.833 | 0.643 | 70.185 | 0.000 | `last_ft_stop_signfix_lr1e4_ep30_1.ckpt` |
| `lr1e4_best` | 1e-4 | 30 | best | 3 | 42 | 0.857 | 0.714 | 76.775 | 0.000 | `best_003_stop_signfix_lr1e4_ep30_1.ckpt` |
| `lr3e4_last` | 3e-4 | 30 | last | 4 | 42 | 0.643 | 0.476 | 83.133 | 0.000 | `last_ft_stop_signfix_lr3e4_ep30_1.ckpt` |
| `lr3e4_best` | 3e-4 | 30 | best | 4 | 42 | 0.952 | 0.738 | 78.099 | 0.000 | `best_004_stop_signfix_lr3e4_ep30_1.ckpt` |
| `lr1e3_last` | 1e-3 | 30 | last | 5 | 42 | 0.738 | 0.381 | 101.993 | 0.000 | `last_ft_stop_signfix_lr1e3_ep30_1.ckpt` |
| `lr1e3_best` | 1e-3 | 30 | best | 5 | 42 | 0.738 | 0.619 | 91.274 | 0.000 | `best_005_stop_signfix_lr1e3_ep30_1.ckpt` |

## Artifact layout

- Train ckpts: `plant2/PlanT/checkpoints_ft/stop_signfix_<lr_tag>_ep30/`
  - `last_ft_stop_signfix_<lr_tag>_ep30_1.ckpt`
  - `best_NNN_stop_signfix_<lr_tag>_ep30_1.ckpt`
- Eval out-dirs: `plant2_stop_pipeline_lrgrid_ep30/eval/<lr_tag>_{last,best}/`
- Train meta: `plant2_stop_pipeline_lrgrid_ep30/train_meta/<lr_tag>.json`
- Logs: `plant2_stop_pipeline_lrgrid_ep30/logs/`
