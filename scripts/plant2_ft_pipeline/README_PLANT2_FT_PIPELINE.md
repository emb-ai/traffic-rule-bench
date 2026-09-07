# PlanT2 FT pipeline — see README.md

This file used to duplicate `README.md` with an older, partially stale
description of the pipeline (it referenced several one-off scripts —
`retrofit_target_speed_expert.py`, `patch_diskcache_2p5_target_speed.py`,
`queue_plant2ft_evals*.sh`, `watch_eval_2p5_tsfix.sh`, … — that were never
committed to this repo or have since been removed).

**Use [`README.md`](README.md) instead** — it covers the same
dump → split → prefill → FT → eval flow, kept in sync with the actual
scripts in `data/`, `train/`, `eval/`, `tools/`.
