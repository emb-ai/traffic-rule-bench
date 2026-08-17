# plant2_ft_pipeline

PlanT2 fine-tune pipeline: dumps → split → diskcache → FT → eval.

Данные, чекпоинты и метрики лежат в `$SHEPELEV` (вне репо).  
Код пайплайна — в `traffic-rule-bench/scripts/plant2_ft_pipeline/`.

---

## Быстрый старт

```bash
export TRB_ROOT=/path/to/traffic-rule-bench
export SHEPELEV=/path/to/workspace          # родитель TRB
export PIPELINE_DIR=$TRB_ROOT/scripts/plant2_ft_pipeline

source $PIPELINE_DIR/shell/env.sh         # PY, CKPT0, SHIM, PLAN_T, …
export CKPT0=$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt

cd $PIPELINE_DIR
```

Все команды ниже предполагают, что `env.sh` уже sourced и `$PY` указывает на conda с PlanT2.

---

## Структура каталогов

```text
plant2_ft_pipeline/
├── lib/          # общие Python-модули (не запускать напрямую)
├── data/         # dumps, split, diskcache
├── train/        # fine-tune
├── eval/         # Sign SR + FV-fast eval
├── tools/        # debug, viz, overfit 1 traj
├── shims/        # Hydra entry (flash_attn off)
├── shell/        # bash env + оркестраторы
├── wrappers/     # локальные обёртки (в .gitignore)
├── README.md
└── README_MODEL_INPUTS.md
```

---

## `lib/` — shared library


| Модуль         | Назначение                                                              |
| -------------- | ----------------------------------------------------------------------- |
| `env.py`       | `$SHEPELEV`, `$TRB_ROOT`, `plan_t()`, `shim_path()`, `resolve_python()` |
| `finetune.py`  | `FinetuneConfig`, `run_finetune()`, Hydra cmd builder                   |
| `eval_core.py` | Sign SR, FV-fast, tag/ckpt resolution                                   |
| `utils.py`     | parallel workers, sample counts, FV expert prep                         |
| `paths.py`     | re-export path helpers                                                  |
| `bootstrap.py` | `sys.path` для `python subdir/script.py`                                |


Импорт из скриптов подпапок:

```python
from lib.env import shepelev, plan_t
from lib.finetune import FinetuneConfig, run_finetune
```

---

## `data/` — данные и cache


| Скрипт                                     | Назначение                                     |
| ------------------------------------------ | ---------------------------------------------- |
| `dump_plant2_l1.py`                        | L1 dumps (experts / fv / lane / rebuild-signs) |
| `make_train_val_split_fv_experts_signs.py` | hardlink split `*_signs` → full split          |
| `make_split_signs_2.5_subset.py`           | symlink subset только sign 2.5                 |
| `extract_patch_2p5_cache.py`               | extract+patch ключей 2.5 из большого cache     |
| `prefill_diskcache.py`                     | prefill / parallel / 2p5 subcommands           |
| `make_balanced_split.py`                   | сплит с потолком на перепредставленные знаки   |
| `make_sign_pair_splits.py`                 | парные сплиты new/old для одного знака         |
| `make_old_half_from_new.py`                | синтез старой половины пары из новых дампов    |
| `assemble_fix_dump.py`                     | сборка дампа из нескольких источников          |


### Примеры

```bash
# Полный rebuild dumps с PDD-знаками в boxes
$PY data/dump_plant2_l1.py rebuild-signs
$PY data/dump_plant2_l1.py rebuild-signs --dry-run
$PY data/dump_plant2_l1.py rebuild-signs --jobs exp:stop fv:3.24 --max-workers 32

# Train/val split
$PY data/make_train_val_split_fv_experts_signs.py
$PY data/make_split_signs_2.5_subset.py

# Prefill full spatial cache (~1.7T с augment)
$PY data/prefill_diskcache.py parallel \
  --ds $SHEPELEV/plant2_l1_fv_experts_split_signs/train \
  --ds-val $SHEPELEV/plant2_l1_fv_experts_split_signs/val \
  --ds-local /tmp/plant2_ds_cache_spatial_aug \
  --cache-size-gb 1800

# Быстрый 2.5 cache из full cache
$PY data/prefill_diskcache.py 2p5 \
  --src /tmp/plant2_ds_cache_spatial_aug \
  --dst /tmp/plant2_ds_cache_2p5_tsfix \
  --split $SHEPELEV/plant2_l1_fv_experts_split_signs_2.5 \
  --cache-size-gb 400 --reset-dst --materialize-missing
```

---

## `train/` — fine-tune


| Скрипт                   | Назначение                                                |
| ------------------------ | --------------------------------------------------------- |
| `launch_ft.py`           | sweeps: `spatial-lr`, `2p5-tsfix`, `2p5-stopw`, `2p5-hyp` |
| `launch_ft.sh`           | thin wrapper → `launch_ft.py`                             |
| `run_plant2_finetune.py` | один FT job (argparse; те же имена читаются из окружения) |

Базовый чекпоинт несёт только `tok_emb.0-6`: эмбеддинги знаков ПДД, `sign_emb`,
`speed_token` и голова скорости создаются заново на каждом дообучении. Ручки для
них — флаги либо одноимённые переменные окружения:

| Ручка                    | Что делает                                                       |
| ------------------------ | ---------------------------------------------------------------- |
| `--gpus` / `GPUS`        | число GPU; при >1 сам заполняет `CUDA_VISIBLE_DEVICES`           |
| `--ddp-strategy`         | по умолчанию `ddp_find_unused_parameters_true` (иначе DDP падает на неиспользованных `tok_emb`) |
| `--init-sign-from-stop`  | инициализировать знаки ПДД из обученного слоя `stop_sign`         |
| `--new-param-lr-mult`    | множитель LR для параметров, которых нет в чекпоинте              |
| `--trunk-lr-mult`        | множитель LR для предобученного транкера; `0` замораживает его    |


Чекпоинты: `$TRB_ROOT/plant2/PlanT/checkpoints_ft/<CHECKPOINT_ADDON>/`  
Логи: `$TRB_ROOT/plant2/PlanT/log/ft_<ADDON>_1/`

### Примеры

```bash
# Sweep 7× LR на GPU 0–6 (full spatial)
$PY train/launch_ft.py spatial-lr

# 2.5 tsfix: 2 job в background (LR 1e-4 / 1e-5)
$PY train/launch_ft.py 2p5-tsfix
$PY train/launch_ft.py 2p5-tsfix --wait

# Stop-weight / hypothesis sweeps
$PY train/launch_ft.py 2p5-stopw --wait
$PY train/launch_ft.py 2p5-hyp --wait

# Один job вручную
$PY train/run_plant2_finetune.py \
  --split $SHEPELEV/plant2_l1_fv_experts_split_signs_2.5 \
  --learning-rate 1e-5 \
  --checkpoint-addon fvexp30_spatial_2p5_tsfix_lr1e5 \
  --cuda-device 0 \
  --ds-local /tmp/plant2_ds_cache_2p5_tsfix \
  --cache-size-gb 400 \
  --batch-size 1344 \
  --max-epochs 30 \
  --augment --no-filter-routes \
  --resume-ckpt $CKPT0
```

---

## `eval/` — evaluation


| Скрипт           | Назначение                            |
| ---------------- | ------------------------------------- |
| `eval_sign25.py` | Sign SR (по умолчанию `--only 2.5`)   |
| `eval_full.py`   | subcommands: `fv`, `queue`, `spatial` |
| `eval_full.sh`   | wrapper → `eval_full.py`              |


Метрики: `$SHEPELEV/plant2_ft_metrics/<root>/<tag>/`

### Примеры

```bash
# Sign SR одного ckpt
$PY eval/eval_sign25.py \
  --ckpt $TRB_ROOT/plant2/PlanT/checkpoints_ft/fvexp30_spatial_2p5_tsfix_lr1e5/best_023_….ckpt \
  --tag my_run_sign25 \
  --gpu 0 --only 2.5 --jobs 8 --scenes-per-job 20

# По addon + slot
$PY eval/eval_sign25.py \
  --addon fvexp30_spatial_2p5_tsfix_lr1e5 --slot best --gpu 1

# Eval одной train-траектории + GIF + predictions
$PY eval/eval_sign25.py \
  --ckpt /path/to.ckpt --tag traj_eval --gpu 0 \
  --trajectory sign_100062_j0_lane0_seed1974118946_v0_default \
  --save-gifs --save-predictions

# FV-fast один ckpt
$PY eval/eval_full.py fv \
  --ckpt /path/to.ckpt \
  --out $SHEPELEV/plant2_ft_metrics/my_tag/fv_fast

# Parallel queue (signs + fv_fast + fv_detour)
$PY eval/eval_full.py queue \
  --metrics-root $SHEPELEV/plant2_ft_metrics

# Spatial 7-GPU eval waves
$PY eval/eval_full.py spatial \
  --metrics-root $SHEPELEV/plant2_ft_metrics/spatial_signs_eval
```

---

## `tools/` — debug и эксперименты


| Скрипт                       | Назначение                                   |
| ---------------------------- | -------------------------------------------- |
| `inspect_boxes.py`           | pretty-print `boxes/NNNN.json.gz`            |
| `print_random_xobjs.py`      | random x_objs из dump                        |
| `viz_train_global_gif.py`    | GIF из train route                           |
| `overfit_1traj_sweep.py`     | train+eval на 1 traj, гиперпараметры из YAML |
| `configs/overfit_1traj.yaml` | конфиг overfit (редактировать здесь)         |
| `check_token_alignment.py`   | срез логитов форкастинга против позиций токенов |
| `compare_train_eval_objects.py` | список объектов в обучении против эвала на одном кадре |


### Примеры

```bash
ROUTE="$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5/train/data/sign_100062_j0_lane0_seed1974118946_v0_default"

# Inspect boxes
$PY tools/inspect_boxes.py --route "$ROUTE" --frame 21 --class 2.5 --verbose
$PY tools/inspect_boxes.py --route "$ROUTE" --summary --class 2.5

# Overfit 1 trajectory (гиперпараметры в YAML, не в .py)
$PY tools/overfit_1traj_sweep.py --config tools/configs/overfit_1traj.yaml --gpu 0
$PY tools/overfit_1traj_sweep.py --dry-run
$PY tools/overfit_1traj_sweep.py --force-train
$PY tools/overfit_1traj_sweep.py --eval-only

# Viz GIF
$PY tools/viz_train_global_gif.py --route "$ROUTE" --fps 10
```

---

## `shims/` — Hydra bootstrap


| Файл                    | Назначение                                                   |
| ----------------------- | ------------------------------------------------------------ |
| `run_lit_finetune.py`   | entry для `lit_finetune.py` (отключает сломанный flash_attn) |
| `disable_flash_attn.py` | patch transformers import checks                             |


Используется автоматически через `$SHIM` из `shell/env.sh`.  
Ручной запуск (из `$PLAN_T`):

```bash
cd $PLAN_T
$PY $SHIM resume=True resume_path=$CKPT0 gpus=1 use_caching=True \
  model.training.learning_rate=1e-5 model.training.max_epochs=30 \
  expname=ft_my_addon
```

---

## `shell/` — оркестраторы

| Файл                        | Назначение                                                     |
| --------------------------- | -------------------------------------------------------------- |
| `env.sh`                    | `PY`, `CKPT0`, `SHIM`, `PLAN_T`, `TRB_ROOT`                     |
| `run_fix_pipeline.sh`       | очередь dump → assemble → split → cache → train → eval          |
| `run_sign_pair_experiment.sh` | парные дообучения на одном знаке: те же сцены, разные кадры   |
| `run_all_checks.sh`         | вся кампания одной очередью, одна таблица в конце               |
| `run_2p5_tsfix_pipeline.sh` | 2.5 tsfix: extract cache → FT → eval                            |
| `launch_overfit_sweep_tmux.sh` | overfit sweep в tmux                                         |

Пути внутри очередей считаются от `$SM`, а вызываемые скрипты — от корня
пайплайна (`$PIPE`), на уровень выше `shell/`.

---

## Типовые пайплайны

### 2.5 tsfix (cache уже есть)

```bash
source $PIPELINE_DIR/shell/env.sh

$PY train/launch_ft.py 2p5-tsfix --wait

$PY eval/eval_sign25.py \
  --addon fvexp30_spatial_2p5_tsfix_lr1e5 --slot best --gpu 0
```

### Full spatial E2E

```bash
source $PIPELINE_DIR/shell/env.sh

$PY data/dump_plant2_l1.py rebuild-signs
$PY data/make_train_val_split_fv_experts_signs.py
$PY data/make_split_signs_2.5_subset.py

$PY data/prefill_diskcache.py parallel \
  --ds $SHEPELEV/plant2_l1_fv_experts_split_signs/train \
  --ds-val $SHEPELEV/plant2_l1_fv_experts_split_signs/val \
  --ds-local /tmp/plant2_ds_cache_spatial_aug \
  --cache-size-gb 1800

$PY train/launch_ft.py spatial-lr
$PY eval/eval_full.py spatial
```

### Overfit одной траектории 2.5

```bash
# 1. Отредактировать tools/configs/overfit_1traj.yaml
# 2. Запустить
$PY tools/overfit_1traj_sweep.py --config tools/configs/overfit_1traj.yaml --gpu 0
```

---

## Где что лежит (вне репо)


| Что        | Путь                                                  |
| ---------- | ----------------------------------------------------- |
| Dumps      | `$SHEPELEV/plant2_l1_*_signs/`                        |
| Splits     | `$SHEPELEV/plant2_l1_fv_experts_split_signs/`         |
| Base ckpt  | `$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt` |
| FT ckpt    | `$TRB_ROOT/plant2/PlanT/checkpoints_ft/<ADDON>/`      |
| Metrics    | `$SHEPELEV/plant2_ft_metrics/`                        |
| Full cache | `/tmp/plant2_ds_cache_spatial_aug`                    |
| 2.5 cache  | `/tmp/plant2_ds_cache_2p5_tsfix`                      |


