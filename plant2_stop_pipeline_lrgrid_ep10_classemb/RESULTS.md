# PlanT STOP LR grid — class_emb (ep10 + lr3e4 ep20/ep30/ep40)

Architecture: shared `class_emb` + `attr_emb` (no `tok_emb` / `sign_emb`).
Warm-start: pretrain `CKPT0` has `tok_emb`; `lit_finetune` loads `strict=False`.

Train dump/split: `plant2_stop_pipeline_signfix/plant2_l1_stop_split/` (294/50).
Test: `stop_data/output/ts_test/real_manifest.jsonl` + `stop_data/scenes/`.
Resume base: `stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`.

Drivers update this file after each eval completes.

## Baseline (prior, not re-run)

| tag | lr | epochs | ckpt | n | success | sign_compliance | efficiency | driving_score |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `baseline_ep20_lr3e4_last` | 3e-4 | 20 | last | 42 | 0.714 | 0.548 | 78.109 | 0.000 |

## LR grid results (ep10)

| tag | lr | epochs | ckpt_kind | best_epoch | n | success | sign_compliance | efficiency | driving_score | ckpt |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `lr1e4_last` | 1e-4 | 10 | last | 0 | 42 | 0.429 | 0.357 | 90.771 | 0.000 | `last_ft_stop_classemb_lr1e4_ep10_1.ckpt` |
| `lr1e4_best` | 1e-4 | 10 | best | 0 | 42 | 0.357 | 0.048 | 133.434 | 0.000 | `best_000_stop_classemb_lr1e4_ep10_1.ckpt` |
| `lr3e4_last` | 3e-4 | 10 | last | 2 | 42 | 0.500 | 0.548 | 86.560 | 0.000 | `last_ft_stop_classemb_lr3e4_ep10_1.ckpt` |
| `lr3e4_best` | 3e-4 | 10 | best | 2 | 42 | 0.595 | 0.452 | 93.425 | 0.000 | `best_002_stop_classemb_lr3e4_ep10_1.ckpt` |
| `lr1e3_last` | 1e-3 | 10 | last | 3 | 42 | 0.405 | 0.190 | 114.111 | 0.000 | `last_ft_stop_classemb_lr1e3_ep10_1.ckpt` |
| `lr1e3_best` | 1e-3 | 10 | best | 3 | 42 | 0.595 | 0.500 | 107.401 | 0.000 | `best_003_stop_classemb_lr1e3_ep10_1.ckpt` |

## Longer train @ lr=3e-4 (last ckpt)

| tag | lr | epochs | ckpt_kind | best_epoch | n | success | sign_compliance | efficiency | driving_score | ckpt |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `lr3e4_ep20_last` | 3e-4 | 20 | last | 5 | 42 | 0.405 | 0.167 | 114.543 | 0.000 | `last_ft_stop_classemb_lr3e4_ep20_1.ckpt` |
| `lr3e4_ep30_last` | 3e-4 | 30 | last | 11 | 42 | 0.619 | 0.476 | 86.845 | 0.000 | `last_ft_stop_classemb_lr3e4_ep30_1.ckpt` |
| `lr3e4_ep40_last` | 3e-4 | 40 | last | 2 | 42 | 0.500 | 0.381 | 101.794 | 0.000 | `last_ft_stop_classemb_lr3e4_ep40_1.ckpt` |

## Artifact layout

- Ep10 train ckpts: `plant2/PlanT/checkpoints_ft/stop_classemb_<lr_tag>_ep10/`
- Ep10 eval: `plant2_stop_pipeline_lrgrid_ep10_classemb/eval/<lr_tag>_{last,best}/`
- Ep20/30 train ckpts: `plant2/PlanT/checkpoints_ft/stop_classemb_lr3e4_ep{20,30}/`
- Ep20/30 eval: `plant2_stop_pipeline_lrgrid_ep20_30_classemb/eval/lr3e4_ep{20,30}_last/`
- Ep20/30 train meta: `plant2_stop_pipeline_lrgrid_ep20_30_classemb/train_meta/`
- Ep40 train ckpts: `plant2/PlanT/checkpoints_ft/stop_classemb_lr3e4_ep40/`
- Ep40 eval: `plant2_stop_pipeline_lrgrid_ep40_classemb/eval/lr3e4_ep40_last/`
- Ep40 train meta: `plant2_stop_pipeline_lrgrid_ep40_classemb/train_meta/`
- Logs: respective `logs/` under each workdir
