# plant2_ft_pipeline

Дообучение PlanT2 на дампах кадров: сплит → обучение → отчёт по кривым.

Замкнутый эвал живёт не здесь, а в `traffic_bench/eval` и запускается своей
командой (`python -m traffic_bench.eval run …`).

## Что где

| файл | назначение |
| --- | --- |
| `data/make_train_val_split_fv_experts_signs.py` | дампы → сплит train/val (жёсткие ссылки) |
| `train/run_plant2_finetune.py` | запуск дообучения |
| `lib/env.py` | пути и интерпретатор |
| `lib/finetune.py` | сборка команды и запуск через `shims/run_lit_finetune.py` |
| `tools/report_ft_metrics.py` | кривые обучения из CSVLogger |

## Переменные окружения

Ни одна не имеет значения по умолчанию, указывающего вовне репозитория.

| переменная | что задаёт | по умолчанию |
| --- | --- | --- |
| `SPLIT_SRCS` | источники сплита, `тег=/абс/путь` через `;` | **обязательна** |
| `SPLIT_OUT` | куда собрать сплит | **обязательна** |
| `ORACLE_ROOT` | прогон сбора: `<корень>/<семейство>/experts/` | не задана — знак определяется по кадрам |
| `VERIFY_GZ` | `0` отключает проверку читаемости кадров | `1` |
| `TRB_ROOT` | корень репозитория | сам чекаут |
| `CKPT0` | стартовый чекпойнт | `<репо>/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt` |
| `METRICS_ROOT` | куда писать метрики | `<репо>/data/plant2_ft_metrics` |
| `PYTHON` | интерпретатор для обучения | текущий |

## Сборка сплита

```bash
cd scripts/plant2_ft_pipeline
SPLIT_SRCS="speed_limit=$DUMP/speed_limit;min_speed=$DUMP/min_speed" \
SPLIT_OUT=$WORK/plant2_splits/speed \
ORACLE_ROOT=$WORK/traj_full/<прогон> \
PYTHONPATH=.:$TRB_ROOT python data/make_train_val_split_fv_experts_signs.py
```

Сборщик читает каждый кадр насквозь и отбрасывает маршруты, которые не
читаются: прерванная запись оставляет `.json.gz` нужного имени и размера с
не-gzip содержимым, и без этой проверки прогон падает с `BadGzipFile` внутри
рабочего процесса загрузчика посреди эпохи.

## Дообучение

```bash
SPLIT=$WORK/plant2_splits/speed DS=$SPLIT/train DS_VAL=$SPLIT/val \
CHECKPOINT_ADDON=speed CKPT_EVERY_N_EPOCHS=2 \
LEARNING_RATE=1e-4 MAX_EPOCHS=20 BATCH_SIZE=128 \
PATH_TARGET=future PATH_HORIZON_FRAMES=40 TS_LOOKAHEAD=1 \
PYTHONPATH=.:$TRB_ROOT python train/run_plant2_finetune.py
```

`TS_LOOKAHEAD=1` меняет метку целевой скорости с номинала знака на скорость,
которую вёл эксперт. Разница существенная: с номиналом модель держит табличку
как уставку и превышает её примерно на половине шагов в зоне (соблюдение 0.000),
с меткой эксперта соблюдение 0.95–0.99 у знаков-потолков.

Отбор чекпойнта по валидационной ошибке ненадёжен: в измеренном прогоне она
монотонно падала все 20 эпох, тогда как доездимость в замкнутом прогоне была
выше на седьмой (0.743 против 0.661 на девятнадцатой). Поэтому чекпойнты стоит
сохранять часто и выбирать по эвалу, а не по `best_*`.

## Кривые

```bash
PYTHONPATH=.:$TRB_ROOT python tools/report_ft_metrics.py $PLANT/log/ft_<addon>_1 --every 4
```
