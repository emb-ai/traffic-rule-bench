# PlanT2 Spatial / PDD-sign Fine-Tune: полный пайплайн

Практическое руководство по данным, пересборке dump’ов, diskcache, FT и eval.
Скрипты пайплайна — в `$PIPELINE_DIR` (`traffic-rule-bench/scripts/plant2_ft_pipeline/`).
Данные, чекпоинты и метрики — в `$SHEPELEV/` (вне репо).

**Python по умолчанию:**

```bash
export TRB_ROOT=/path/to/traffic-rule-bench          # корень этого репо
export SHEPELEV=/path/to/workspace                   # родитель TRB: данные, ckpt, conda
export PIPELINE_DIR=$TRB_ROOT/scripts/plant2_ft_pipeline
source $PIPELINE_DIR/_env.sh                         # TRB_ROOT, SHEPELEV, PY, SHIM, …
export CKPT0=$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt
```

---

## Quickstart: 2.5-only FT на исправленном cache

Если dump’ы и split уже есть, а `/tmp/plant2_ds_cache_2p5_tsfix` уже собран
(сейчас ~124 G) — достаточно FT + eval:

```bash
# 1) FT (2 LR × 2 GPU), addon = fvexp30_spatial_2p5_tsfix_lr{1e4,1e5}
bash $PIPELINE_DIR/launch_ft_2p5_tsfix_only.sh

# либо полный пайплайн (retrofit → extract cache → FT → eval --only 2.5):
# bash $PIPELINE_DIR/run_2p5_tsfix_pipeline.sh

# 2) Eval Sign SR только на 2.5 (после появления best_*.ckpt / epoch=029_*.ckpt)
METRICS_ROOT=$SHEPELEV/plant2_ft_metrics/spatial_2p5_tsfix_eval_sign25 \
  bash $PIPELINE_DIR/watch_eval_2p5_tsfix.sh
# или вручную по шаблону launch_2p5_sign25_eval.sh (см. §5), сменёнными путями ckpt/METRICS_ROOT
#
# Для полного (не 2.5-only) FT-eval пайплайн дополнительно гоняет FV-fast:
#   run_eval_fast_plant2ft.sh  (fv_fast + fv_fast_detour)
# или параллельную очередь queue_plant2ft_evals_par.sh / launch_spatial_ft_eval_7gpu.sh
# (на 2.5-only эти шаги пропускаются — в catalog_fv_test20 нет sign 2.5).
```

Ключевые переменные для 2.5:

| Env | Значение |
|---|---|
| `SPLIT` | `$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5` |
| `DS` / `DS_VAL` | `$SPLIT/train`, `$SPLIT/val` |
| `DS_LOCAL` | `/tmp/plant2_ds_cache_2p5_tsfix` |
| `CACHE_SIZE_GB` | `400` (хватит с запасом на ~124 G) |
| Resume | `$CKPT0` = `$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt` |

---

## 1. Где лежат данные

### 1.1 Expert dumps (источники L1)

| Путь | Содержание |
|---|---|
| `$SHEPELEV/plant2_l1_from_experts_signs/` | Priority/detour experts с PDD-знаками в `boxes` (yield 2.4, stop 2.5, secondary 2.3.*, main 2.1, roundabout 4.3, detour 4.2.*) |
| `$SHEPELEV/plant2_l1_traj_fv_nodeA_signs/` | FV speed-limit (nodeA): 3.24, 4.6, 5.21, 5.31 + каталожные `v_target_*` |
| `$SHEPELEV/plant2_l1_lane_signs/` | Lane direction 5.15.1 |

Старые деревья без `_signs` (`plant2_l1_from_experts`, `plant2_l1_traj_fv_nodeA`, …) — предыдущие dumps без spatial sign tokens; для текущего FT не использовать.

Layout одного route:

```text
<data>/<scene_uid>_<variant>/
  boxes/NNNN.json.gz
  measurements/NNNN.json.gz
  bev_no_car_semantics/NNNN.png
  bev_no_car_semantics_augmented/NNNN.png
  results.json.gz
```

### 1.2 Train/val splits

| Путь | Описание |
|---|---|
| `$SHEPELEV/plant2_l1_fv_experts_split_signs/` | Полный hardlink-split (SEED=42, fixed50 val на знак). Источники: fv + exp + lane `_signs`. |
| `$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5/` | Symlink-подмножество только знака **2.5** (644 train / 50 val). |

Оба содержат `split_meta.json`, `train/data/`, `val/data/`.

Пример counts полного split (`per_sign`, mode=`fixed50`): 2.5 → N=694; 3.24 → 1567; 5.21 → 1272; 5.31 → 1398; …

### 1.3 Checkpoints

| Роль | Путь |
|---|---|
| Base (pretrain / resume) | `$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt` |
| FT outputs | `$TRB_ROOT/plant2/PlanT/checkpoints_ft/<CHECKPOINT_ADDON>/` |
| Lightning / wandb logs | `$TRB_ROOT/plant2/PlanT/log/ft_<ADDON>_<SEED>/` |

Примеры addon’ов: `fvexp30_spatial_lr*`, `fvexp30_spatial_2p5_tsfix_lr*`, `fvexp30_2p5_stopw*_lr*`, `fvexp30_2p5_h1*_lr*`.

### 1.4 Diskcache

| Cache | Путь | Размер (ориентир) |
|---|---|---|
| Full spatial + aug | `/tmp/plant2_ds_cache_spatial_aug` | ~1.7 T |
| 2.5-only tsfix | `/tmp/plant2_ds_cache_2p5_tsfix` | ~124 G |

Ключи = абсолютные пути `…/boxes/NNNN.json.gz` (+ sibling `…_aug` при `augment=True`).

### 1.5 Metrics

Корень: `$SHEPELEV/plant2_ft_metrics/`

| Подкаталог | Назначение |
|---|---|
| `spatial_2p5_eval_sign25/` | Старый 2.5 eval (до tsfix) |
| `spatial_2p5_tsfix_eval_sign25/` | Eval после tsfix FT |
| `spatial_2p5_hyp_eval_sign25/`, `spatial_2p5_stopw_eval_sign25/` | Hyp / stop-weight sweeps |
| `spatial_signs_eval/`, `spatial_tsfix_eval/` | Full-sign eval |

На run: `$SHEPELEV/plant2_ft_metrics/<root>/<tag>/{ckpt.txt,signs→symlink,logs/}` +
опционально `fv_fast/`, `fv_fast_detour/` (выход `run_eval_fast_plant2ft.sh`).

### 1.6 Ключевой код

| Компонент | Путь |
|---|---|
| PlanT training | `$TRB_ROOT/plant2/PlanT/` (`lit_finetune.py`, `lit_module.py`, `dataset.py`) |
| FT shim (disable flash_attn) | `$PIPELINE_DIR/plant2_py_shims/run_lit_finetune.py` |
| Dump frames | `$TRB_ROOT/pdd-bench/scripts/per_sign_bench/bench/plant2_frames.py` |
| Expert replay | `$TRB_ROOT/pdd-bench/scripts/per_sign_bench/expert_replay_inenv.py` |
| MetaDrive adapter / PID | `$TRB_ROOT/pdd-bench/agents/plant2_in_metadrive/plant2_adapter.py` |
| Policy load | `$TRB_ROOT/pdd-bench/scripts/per_sign_bench/bench/policy_factory.py` |
| Eval harness (Sign SR) | `$TRB_ROOT/pdd-bench/scripts/per_sign_bench/plant2_rule_test/eval_checkpoint_on_test.py` |
| FV-fast eval (FT) | `$PIPELINE_DIR/run_eval_fast_plant2ft.sh` |
| FV-fast parallel queue | `$PIPELINE_DIR/queue_plant2ft_evals_par.sh` |

---

## 2. Скрипты пересбора данных

### 2.1 Параллельный rebuild со знаками

Главный скрипт:

```bash
# полный parallel rebuild → *_signs деревья
bash $PIPELINE_DIR/dump_plant2_l1_rebuild_signs_parallel.sh

DRY_RUN=1 bash $PIPELINE_DIR/dump_plant2_l1_rebuild_signs_parallel.sh
MAX_WORKERS=32 JOBS="exp:stop fv:3.24" bash $PIPELINE_DIR/dump_plant2_l1_rebuild_signs_parallel.sh
```

Выходы (не затирает старые non-`_signs`):

- `$SHEPELEV/plant2_l1_from_experts_signs/`
- `$SHEPELEV/plant2_l1_traj_fv_nodeA_signs/`
- `$SHEPELEV/plant2_l1_lane_signs/`

Логи: `$PIPELINE_DIR/logs_dump_signs/`.

Отдельные (более старые) обёртки по семействам:

- `$PIPELINE_DIR/dump_plant2_l1_from_experts.sh` → default OUT `plant2_l1_from_experts`
- `$PIPELINE_DIR/dump_plant2_l1_traj_fv_nodeA.sh` → default OUT `plant2_l1_traj_fv_nodeA`
- `$PIPELINE_DIR/dump_plant2_l1_lane_parallel.sh` — lane

Для signs-дерева задавайте `OUT_DIR=..._signs` или используйте parallel rebuild.

### 2.2 `expert_replay_inenv.py` + `plant2_frames.py`

Replay вызывает `Plant2FrameCollector` из `bench/plant2_frames.py` при `--save-plant2-dir`.

**Два источника target_speed:**

| Источник | Признак | `target_speed` | `brake` |
|---|---|---|---|
| **exp** (priority/stop/yield/…) | нет `v_target_kmh` / `v_target_raw_kmh` | = expert `ego_speed`, clamp ≤ 20 | `speed < 0.5` |
| **fv** (speed-limit catalog) | есть `v_target_*` | лимит в зоне / raw вне зоны, clamp ≤ 20 | всегда `False` |

После фикса семантики в measurements всегда пишутся:

- `speed`, `ego_speed` — фактическая скорость эксперта (м/с);
- `target_speed` — цель для ego-speed head (см. таблицу);
- `brake` — флаг для `PlanTDataset` (при `brake=True` dataset форсит `target_speed=0`).

### 2.3 Caveat: skip-if-exists

В `expert_replay_inenv.py` batch-режиме:

```text
если <plant2_dir>/data/<uid>_<variant>/results.json.gz уже есть → SKIP
```

Повторный запуск **не** перезапишет старые (битые `target_speed=20`) routes.
Нужен retrofit measurements или удаление route-dir перед re-dump.

### 2.4 Retrofit / extract / patch

| Скрипт | Назначение |
|---|---|
| `$PIPELINE_DIR/retrofit_target_speed_expert.py` | In-place: non-speed-limit routes → `target_speed=min(speed,20)`, `brake=(speed<0.5)`, `ego_speed=speed`. **Не трогает** 3.24 / 4.6 / 5.21 / 5.31. Опционально `--purge-cache`. |
| `$PIPELINE_DIR/extract_patch_2p5_cache.py` | Копирует ключи 2.5 из большого cache → `/tmp/plant2_ds_cache_2p5_tsfix` с патчем `target_speed` (brake→0). Без полного `iterkeys()` по 1.7 T. |
| `$PIPELINE_DIR/patch_diskcache_2p5_target_speed.py` | Точечный patch cache |
| `$PIPELINE_DIR/patch_cache_nonspeed_inplace.py` | In-place non-speed-limit keys (осторожно на full cache) |

Пример retrofit только 2.5:

```bash
$PY -u $PIPELINE_DIR/retrofit_target_speed_expert.py --signs 2.5 --workers 32
```

### 2.5 Train/val split

```bash
# полный signs split (hardlinks)
$PY $PIPELINE_DIR/make_train_val_split_fv_experts_signs.py

# 2.5-only symlink subset
$PY $PIPELINE_DIR/make_split_signs_2.5_subset.py
```

Правила: SEED=42, val=`fixed50` на знак; источники `fv` / `exp` / `lane` `_signs`.

---

## 3. Прогрев кеша параллельно

### Скрипты

- `$PIPELINE_DIR/prefill_plant2_diskcache.py` — один shard
- `$PIPELINE_DIR/prefill_plant2_diskcache_parallel.sh` — шардирование train + один val job

### Env

| Переменная | Default / смысл |
|---|---|
| `DS` | `$SPLIT/train` |
| `DS_VAL` | `$SPLIT/val` |
| `DS_LOCAL` | `/tmp/plant2_ds_cache_spatial_aug` |
| `CACHE_SIZE_GB` | `1800` (full); для 2.5 — `400` |
| `PREFILL_AUGMENT` | `1` → base + `*_aug`; `0` → только base |
| `MAX_WORKERS` | default ≤32 (NFS-friendly) |
| `PREFILL_STOP_FRAC` | `0.97` |
| `PREFILL_SPLIT` | `train` / `val` / `both` |
| `PREFILL_START` / `PREFILL_END` | индексный диапазон |

**Важно:** `PREFILL_AUGMENT=1` создаёт base и `_aug` ключи. При FT с `augment=False` (H5) используются только base-ключи — **пересобирать cache не нужно**.

### Пример: full spatial

```bash
export DS=$SHEPELEV/plant2_l1_fv_experts_split_signs/train
export DS_VAL=$SHEPELEV/plant2_l1_fv_experts_split_signs/val
export DS_LOCAL=/tmp/plant2_ds_cache_spatial_aug
export CACHE_SIZE_GB=1800
export PREFILL_AUGMENT=1
DRY_RUN=1 bash $PIPELINE_DIR/prefill_plant2_diskcache_parallel.sh
MAX_WORKERS=32 bash $PIPELINE_DIR/prefill_plant2_diskcache_parallel.sh
```

Логи: `/tmp/plant2_prefill_logs/`.

Для 2.5 после retrofit предпочтительнее `$PIPELINE_DIR/extract_patch_2p5_cache.py` (быстрее и безопаснее, чем полный prefill).

---

## 4. Обучение FT

### 4.1 Лаунчеры

| Скрипт | Что делает |
|---|---|
| `$PIPELINE_DIR/run_plant2_finetune.sh` | Один job; читает `SPLIT`, `LEARNING_RATE`, `CHECKPOINT_ADDON`, … |
| `$PIPELINE_DIR/plant2_py_shims/run_lit_finetune.py` | Точка входа Hydra (`lit_finetune.py` + disable flash_attn) |
| `$PIPELINE_DIR/launch_plant2_ft_spatial_lr_sweep.sh` | 7 GPU, full split, LR ∈ {1e-6…1e-4}, addon `fvexp30_spatial_lr*` |
| `$PIPELINE_DIR/launch_plant2_ft_2p5_lr_sweep.sh` | 2 GPU, 2.5 split; **старые** addon `fvexp30_spatial_2p5_lr*` и `DS_LOCAL=spatial_aug` |
| `$PIPELINE_DIR/launch_ft_2p5_tsfix_only.sh` | 2.5 + `/tmp/plant2_ds_cache_2p5_tsfix`, addon `…_tsfix_lr*` |
| `$PIPELINE_DIR/launch_ft_2p5_stopw_sweep.sh` | `stop_speed_loss_weight` ∈ {5,10,20} × LR |

### 4.2 Env vars

| Var | Типичное значение |
|---|---|
| `DS`, `DS_VAL`, `DS_LOCAL` | split train/val + cache |
| `LEARNING_RATE` | `1e-4`, `1e-5`, … |
| `CHECKPOINT_ADDON` | имя подкаталога в `checkpoints_ft/` |
| `BATCH_SIZE` | `1344` (spatial / 2.5) |
| `MAX_EPOCHS` | `30` |
| `NUM_WORKERS` | `4` |
| `CKPT_EVERY_N_EPOCHS` | `5` |
| `LR_SCHEDULER` | `cosine_warmup` |
| `WARMUP_RATIO` | `0.1` |
| `CACHE_SIZE_GB` | `1800` full / `400` 2.5 |
| `SEED` | `1` |
| `WANDB_MODE` | `offline` |

### 4.3 Hydra overrides (примеры)

```bash
# базовый spatial / tsfix
model.training.augment=True
model.training.augment_parked=False
'+model.training.filter_routes=False'

# H1: убрать path/forecast, усилить speed
model.waypoints.path_weight=0
model.pre_training.forecastLoss_weight=0
model.waypoints.speed_weight=5

# H2: class weights на stop-bin
model.training.speed_class_weights=[15,1,1,1,1,1,1,1]

# stop-weight (samples с target_speed < 0.5)
model.training.stop_speed_loss_weight=10

# H5: без аугментации (base cache keys)
model.training.augment=False
```

### 4.4 tmux-паттерн (spatial)

`launch_plant2_ft_spatial_lr_sweep.sh` поднимает сессии вида
`arbelyaev-ft-spatial-lr{1e6,…}` с логами `/tmp/plant2_ft_spatial_lr*.log`.

Для tsfix 2.5 удобнее background subshell’ы из `launch_ft_2p5_tsfix_only.sh`
(логи `/tmp/plant2_ft_2p5_tsfix_lr{1e4,1e5}.log` + `$PIPELINE_DIR/logs_pipeline_2p5_tsfix/`).

Пример ручного tmux (один GPU):

```bash
SESSION=arbelyaev-ft-2p5-tsfix-lr1e5
tmux new-session -d -s "$SESSION" bash -lc "
cd $PLAN_T
export CUDA_VISIBLE_DEVICES=1 LEARNING_RATE=1e-5
export CHECKPOINT_ADDON=fvexp30_spatial_2p5_tsfix_lr1e5
export SPLIT=$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5
export DS=\$SPLIT/train DS_VAL=\$SPLIT/val
export DS_LOCAL=/tmp/plant2_ds_cache_2p5_tsfix CACHE_SIZE_GB=400
export MAX_EPOCHS=30 BATCH_SIZE=1344 NUM_WORKERS=4
export LR_SCHEDULER=cosine_warmup WARMUP_RATIO=0.1 SEED=1
export WANDB_MODE=offline PYTHONNOUSERSITE=1
$PY -u $SHIM resume=True resume_path=$CKPT0 gpus=1 use_caching=True \
  lr_scheduler=cosine_warmup warmup_ratio=0.1 \
  model.training.learning_rate=1e-5 model.training.max_epochs=30 \
  model.training.batch_size=1344 model.training.num_workers=4 \
  model.training.augment=True model.training.augment_parked=False \
  '+model.training.filter_routes=False' \
  expname=ft_fvexp30_spatial_2p5_tsfix_lr1e5 \
  2>&1 | tee /tmp/plant2_ft_2p5_tsfix_lr1e5.log
exec bash
"
```

### 4.5 Куда пишутся ckpt / logs

```text
$PLAN_T/checkpoints_ft/<ADDON>/best_NNN_<ADDON>_1.ckpt
$PLAN_T/checkpoints_ft/<ADDON>/epoch=EEE_<ADDON>_1.ckpt
$PLAN_T/log/ft_<ADDON>_<SEED>/
/tmp/plant2_ft_*.log
$PIPELINE_DIR/logs_pipeline_*/ft_*.log
```

---

## 5. Eval pipeline

Полный FT-eval на один ckpt обычно состоит из:

1. **Sign SR** — `plant2_rule_test/eval_checkpoint_on_test.py` (+ `summarize_reports.py`);
2. **FV-fast** — `$PIPELINE_DIR/run_eval_fast_plant2ft.sh`
   → `$SHEPELEV/plant2_ft_metrics/<tag>/fv_fast/` (catalog `catalog_fv_test20`);
3. **FV-fast detour** — тот же скрипт → `…/<tag>/fv_fast_detour/` (detour catalog).

Параллельный аналог (несколько ckpt сразу):  
`$PIPELINE_DIR/queue_plant2ft_evals_par.sh`  
(внутри: signs + `run_eval_fast_plant2ft.sh` ×2). Для 7-GPU spatial waves —  
`$PIPELINE_DIR/launch_spatial_ft_eval_7gpu.sh`.  
Последовательная очередь: `$PIPELINE_DIR/queue_plant2ft_evals.sh`.

> FT-обёртка FV-fast: `$PIPELINE_DIR/run_eval_fast_plant2ft.sh` (не путать с legacy `run_eval_fast.sh`).

### 5.1 Sign SR (`--only 2.5`)

Харнесс:

```bash
cd $TRB_ROOT/pdd-bench/scripts/per_sign_bench/plant2_rule_test
$PY -u eval_checkpoint_on_test.py \
  --policies plant2 \
  --model-paths "plant2:/path/to.ckpt" \
  --jobs 8 --scenes-per-job 20 \
  --only 2.5 --keep-going \
  --run-name <TAG>
$PY summarize_reports.py \
  --run-name <TAG> \
  --baseline plant2_default \
  --out-dir output/<TAG>/_summary
```

Готовность: `$TRB_ROOT/pdd-bench/scripts/per_sign_bench/plant2_rule_test/output/<TAG>/_summary/summary.md`.

Шаблон оркестратора (старые addon-имена — править пути ckpt):

```bash
METRICS_ROOT=$SHEPELEV/plant2_ft_metrics/spatial_2p5_eval_sign25 \
  bash $PIPELINE_DIR/launch_2p5_sign25_eval.sh
```

Для tsfix см. блок eval в `run_2p5_tsfix_pipeline.sh` /
`watch_eval_2p5_tsfix.sh` → metrics в `spatial_2p5_tsfix_eval_sign25/`.

**Замечание:** `catalog_fv_test20` / detour **не содержат** sign 2.5 → шаги
`fv_fast` / `fv_fast_detour` (`run_eval_fast_plant2ft.sh`) для 2.5-only
**пропускаются**. На полном split их запускают очереди ниже (§5.2).

### 5.2 FV-fast: `run_eval_fast_plant2ft.sh` и параллель

Один ckpt (ручной вызов):

```bash
CKPT=/path/to.ckpt \
OUT=$SHEPELEV/plant2_ft_metrics/<tag>/fv_fast \
GPUS="0 1 2 3 4 5 6" NSHARDS=28 CONCURRENCY=28 \
  bash $PIPELINE_DIR/run_eval_fast_plant2ft.sh
```

Ключевые env (из шапки скрипта):

| Env | Default / смысл |
|---|---|
| `CKPT` | **обязателен** — путь к plant2-ft `.ckpt` |
| `OUT` | **обязателен** — каталог метрик этого ckpt |
| `MANIFEST` | `…/catalog_fv_test20.jsonl` (только test20; guard падает иначе) |
| `SCENES` | `…/scenes_balanced` |
| `GPUS` | `"0 1 2 3 4 5 6"` |
| `NSHARDS` / `CONCURRENCY` | `28` / `28` |
| `NN_POLICIES` | `plant2` |
| `EXCLUDE_CODES` | `"3.25 5.22 5.32"` |
| `PLANT2_ACTION_MODE` | `pid` |
| `PY` | `$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python` |

Параллельная очередь по многим ckpt:

```bash
# signs + fv_fast + fv_fast_detour; GPU round-robin
GPUS="0 1 2 3 4 5 6" \
SIGNS_PARALLEL=8 SIGNS_JOBS=20 SCENES_PER_JOB=32 \
FV_PARALLEL=4 FV_NSHARDS=28 FV_CONCURRENCY=28 \
  bash $PIPELINE_DIR/queue_plant2ft_evals_par.sh
```

Spatial 7-GPU waves (best → ep029 → …), тоже вызывает `run_eval_fast_plant2ft.sh`:

```bash
METRICS_ROOT=$SHEPELEV/plant2_ft_metrics/spatial_signs_eval \
  bash $PIPELINE_DIR/launch_spatial_ft_eval_7gpu.sh
```

Готовность FV-шага: `$OUT/reports/report_cumulative.md`.

### 5.3 Layout метрик

```text
$SHEPELEV/plant2_ft_metrics/<root>/<tag>/
  ckpt.txt
  eval_filter.txt          # ONLY_SIGNS=… (для 2.5-only)
  signs -> symlink на plant2_rule_test/output/<tag>
  fv_fast/                 # run_eval_fast_plant2ft.sh (full eval)
  fv_fast_detour/          # тот же скрипт, detour catalog
  logs/eval_checkpoint.log
  logs/run_eval_fast_*.log
```

Сырые отчёты Sign SR: `$TRB_ROOT/pdd-bench/scripts/per_sign_bench/plant2_rule_test/output/<tag>/` + `_summary/summary.md`.

### 5.4 GIFs (опционально)

Через `eval_pipeline.py --save-gifs` (прокидывается из eval-харнесса при
явном флаге). Пример артефактов:  
`$SHEPELEV/plant2_ft_metrics/spatial_2p5_tsfix_eval_sign25/_gifs_lr1e5/gifs/`.

### 5.5 Action mode / lookahead

- Default: `--plant2-action-mode pid` (`eval_pipeline.py` /
  `run_eval_fast_plant2ft.sh` → `PLANT2_ACTION_MODE`).
- В `plant2_adapter.py` lateral PID lookahead ×**2.0** уже по умолчанию
  (компенсация MetaDrive ~10 Hz vs CARLA 20 Hz). Override: `PLANT2_LOOKAHEAD_MULT`.
- Soft speed decode на инференсе: `softmax` по 8 speed bins → desired speed
  (не hard argmax).

---

## Важные gotchas

1. **Skip-existing dumps.** Наличие `results.json.gz` → scene пропускается.
   После фикса `target_speed` нужен `retrofit_…` или удаление route перед re-dump.

2. **Cache печёт `target_speed`.** Retrofit measurements без purge/rebuild/extract
   cache оставляет старые labels в diskcache. Для 2.5: `extract_patch_2p5_cache.py`
   или `--purge-cache` у retrofit.

3. **Speed-limit знаки (3.24 / 4.6 / 5.21 / 5.31)** не retrofit’ить measurements
   и не патчить их cache keys «как stop».

4. **Soft two-hot CE** в `lit_module.py`: обучение — soft labels между соседними
   bins; инференс — softmax expectation. Не путать с hard class CE.

5. **`brake` в lit_module захардкожен в `False`:**

   ```python
   brake = torch.zeros_like(targetspeed_batch, dtype=torch.bool, ...)
   ```

   Стоп-супервизия идёт через `target_speed≈0` (dataset уже обнуляет при
   `measurements.brake`), плюс опционально `stop_speed_loss_weight` /
   `speed_class_weights`. Нельзя полагаться на batch-поле `brake` в loss.

6. **`augment=False` не требует rebuild cache** — просто читаются base-ключи
   (без `_aug`).

7. **Имена addon.** Не перезаписывать старые `fvexp30_spatial_2p5_lr*` /
   `spatial_2p5_eval_sign25`; для фикса target_speed — суффикс `tsfix` и
   отдельный `METRICS_ROOT`.

8. **`filter_routes=False`** на готовом split обязателен: иначе лишний I/O по
   `results`/`slurm` на NFS.

---

## Сводка команд end-to-end (full spatial)

```bash
# A. Dump (долго)
bash $PIPELINE_DIR/dump_plant2_l1_rebuild_signs_parallel.sh

# B. Split
$PY $PIPELINE_DIR/make_train_val_split_fv_experts_signs.py
$PY $PIPELINE_DIR/make_split_signs_2.5_subset.py   # опционально для 2.5-only

# C. Prefill full cache (очень долго / много /tmp)
export DS=$SHEPELEV/plant2_l1_fv_experts_split_signs/train
export DS_VAL=$SHEPELEV/plant2_l1_fv_experts_split_signs/val
export DS_LOCAL=/tmp/plant2_ds_cache_spatial_aug CACHE_SIZE_GB=1800 PREFILL_AUGMENT=1
bash $PIPELINE_DIR/prefill_plant2_diskcache_parallel.sh

# D. FT LR sweep
bash $PIPELINE_DIR/launch_plant2_ft_spatial_lr_sweep.sh

# E. Eval: Sign SR + FV-fast (run_eval_fast_plant2ft) / parallel queue
bash $PIPELINE_DIR/launch_spatial_ft_eval_7gpu.sh
# или: bash $PIPELINE_DIR/queue_plant2ft_evals_par.sh
# один ckpt FV-fast:
#   CKPT=… OUT=$SHEPELEV/plant2_ft_metrics/<tag>/fv_fast bash $PIPELINE_DIR/run_eval_fast_plant2ft.sh
```

## Сводка 2.5 tsfix (после уже существующего full cache)

```bash
# retrofit measurements → extract small cache → FT → eval Sign SR (--only 2.5)
bash $PIPELINE_DIR/run_2p5_tsfix_pipeline.sh
# или по стадиям: retrofit → extract_patch_2p5_cache.py → launch_ft_2p5_tsfix_only.sh → watch_eval_2p5_tsfix.sh
# FV-fast (run_eval_fast_plant2ft / queue_plant2ft_evals_par) на 2.5-only не нужен —
# catalog_fv_test20 без sign 2.5; включается на full spatial eval (§5.2).
```

Доп. эксперименты на том же cache:

```bash
bash $PIPELINE_DIR/launch_ft_2p5_hyp_sweep.sh      # логи: $PIPELINE_DIR/logs_pipeline_2p5_hyp/
bash $PIPELINE_DIR/launch_ft_2p5_stopw_sweep.sh    # логи: $PIPELINE_DIR/logs_pipeline_2p5_stopw/
```

