# per_sign_bench — бенчмарк сцен по ПДД-знакам

Чистая версия кода бенчмарка (экспериментальные файлы лежат рядом в `../benchmark/`).

## Что делает

Для 46 ПДД-знаков собирает набор сцен из трёх источников и материализует их в
MetaDrive / SUMO:

| Источник | Модули | Для каких знаков |
|---|---|---|
| **PGMap** (синтетика) | `factorized_space/` | универсальные, intersection-priority, detour (`main_road`, `yield`, `speed*`, `detour_*`, `bus_lane`, …) |
| **CityMap** (синтетика + detour) | `citymap_space/` | prohibition-with-detour (`no_entry`, `no_traffic`, `no_right_turn`, `no_left_turn`, `no_uturn`) |
| **SUMO** (реальные сцены) | `sumo_space/` | все знаки, имеющие `.net.xml` в `pdd-bench/scenes/<code>/` |

## Структура

```
per_sign_bench/
├── per_sign_benchmark.py          # CLI orchestrator (mini/full presets)
├── run_benchmark.sh               # wrapper для локального/remote запуска
│
├── factorized_space/              # PGMap-ядро
│   ├── space_definition.py        # оси, PDD↔key mapping, paired bundles, bike-signs
│   ├── index_codec.py             # flat_index ↔ base scene spec
│   ├── materialized_sampler.py    # sample_for_sign
│   ├── agent_profile_bank.py      # nuPlan profile / spawn_velocity / vehicle_type sampling
│   ├── benchmark_runner.py        # generate_scene, generate_paired_scene, bicycle spawn, NPC hook
│   ├── ego_defaults.py            # DEFAULT_EGO_PARAMS (изоляция ego от NPC-профиля)
│   ├── sign_placement.py          # SIGN_PLACEMENT_RULES (детерминированное расположение знака)
│   └── paired_space.py            # codec для парных сцен (начало/конец зоны)
│
├── citymap_space/                 # CityMap для prohibition-with-detour
│   ├── citymap_analyzer.py
│   ├── citymap_env.py
│   ├── citymap_scene_enumerator.py
│   ├── citymap_runner.py
│   └── citymap_pipeline.py
│
└── sumo_space/                    # реальные SUMO сцены
    ├── sumo_scene_enumerator.py   # + count_lanes_on_road (парсит .net.xml)
    ├── sumo_catalog.py            # все полосы × n_variations × n_velocity
    ├── sumo_runner.py
    └── sumo_pipeline.py
```

## Как запустить

### Локально (mini, ~1 час)

```bash
cd /Users/victoria_s/sdc_new_signs/sdc/pdd-bench/scripts/per_sign_bench
caffeinate -i ./run_benchmark.sh mini > mini.log 2>&1 &
echo $! > mini.pid
tail -f mini.log
```

### Удалённо в tmux (full, ~16 часов)

```bash
ssh user@remote
cd /path/to/pdd-bench/scripts/per_sign_bench
tmux new -s bench-full
./run_benchmark.sh full 2>&1 | tee full.log
# Ctrl-b d — detach
```

### Только один знак / набор знаков

```bash
./run_benchmark.sh mini "2.1,3.19,4.2.1"
```

### Только план (без материализации)

```bash
python per_sign_benchmark.py --preset mini --dry-run
```

### Пропустить бэкенд

```bash
python per_sign_benchmark.py --preset mini --materialize \
    --skip-materialize citymap,sumo
```

## Пресеты (см. `PRESETS` в `per_sign_benchmark.py`)

| Пресет | target_per_sign | n_variations | n_velocity | citymap_maps | Время |
|---|---:|---:|---:|---:|---|
| `mini` | 10 | 1 | 2 | 4 | ~1 час (laptop) |
| `full` | 250 | 10 | 5 | 40 | ~16 часов (remote) |

## Выход

```
benchmark_output/{mini,full}/
├── summary.json                       # план
├── materialization_summary.json       # финальная сводка per-backend
│
└── 2_1/                                # папка на каждый ПДД
    ├── source.json                    # метаданные знака
    ├── synthetic_manifest.jsonl       # PGMap base specs
    ├── pgmap_materialized.jsonl       # PGMap результаты (spec × v_idx)
    ├── paired_materialized.jsonl      # парные сцены (если есть)
    ├── real_manifest.jsonl            # SUMO catalog
    ├── sumo/sumo_manifest.jsonl       # SUMO результаты
    └── citymap_materialized.jsonl     # CityMap результаты (для prohibition)
```

## Параметры сцены

**Из factorized space** (индексируется `flat_index`):
- `block_id × route_intent` (S/C/r/R/T/X/O)
- `lane_num ∈ {2, 3, 4, 5}`
- `spawn_lane_semantic` (left/center/right, дедуп)
- `sign_type` (+ `accident_prob` для detour)

**Из nuPlan (сэмплится per-scene, детерминистически):**
- `spawn_velocity_ms` — `routes.csv.initial_speed`, N_VEL сэмплов на base-сцену
- Профиль NPC: `NORMAL_SPEED, ACC_FACTOR, …` из KDE — применяется к `IDMPolicy` class attrs
- `traffic_density` — из `densities.csv`, cap 1.0
- `vehicle_type` (per-NPC, не per-scene) — из `size_dist`, через hook в `PGTrafficManager`

**Ego изолирован** — использует `DEFAULT_EGO_PARAMS` (не nuPlan-профиль).

**Для bike-знаков** (`bike_lane*`, `exit_to_bike_lane*`) — спавнится 3 велосипедиста на полосе знака.

## Зависимости (внешние, не в per_sign_bench)

- `sdc/pdd-bench/traffic_signs/` — классы знаков
- `sdc/pdd-bench/envs/` — `TrafficSignEnv`, `TrafficSignSumoEnv`
- `sdc/metadrive/` — MetaDrive fork (submodule)
- `sdc/pdd-bench/scenes/` — реальные SUMO .net.xml + meta.json
- `nuplan_statistics/` — CSV-файлы статистик (routes, speeds, acc_pos, acc_neg, following, densities, lane_changes)

## Evaluation (optional)

Для оценки diversity / realism / behavioral метрик можно использовать скрипты
из соседней `../benchmark/`:
- `evaluate_benchmark.py` — diversity + realism report
- `run_behavioral_validation.py` — прогон expert policy через сцены

## Chunked test metrics (18 baselines)

Скрипты для batch-оценки baselines на `full_test_250_x10.zip` с запуском чанками
и накоплением cumulative-метрик:

- `run_18_baselines_test_chunk.sh`
- `prepare_full_test_chunk.py`
- `aggregate_chunk_metrics.py`
- `generate_chunk_markdown_report.py`

Что делает пайплайн:

- распаковывает test zip в дефолтный run root;
- собирает chunk манифестов (например, по 200 сцен);
- запускает baselines по backend'ам `sumo,pgmap,paired,citymap`;
- сохраняет per-episode метрики, включая `route_length_m` и `distance_travelled_m`;
- строит chunk report и cumulative report по мере новых запусков.

Запуск:

```bash
CHUNK_START=0 CHUNK_SIZE=200 \
bash pdd-bench/scripts/per_sign_bench/run_18_baselines_test_chunk.sh

CHUNK_START=200 CHUNK_SIZE=200 \
bash pdd-bench/scripts/per_sign_bench/run_18_baselines_test_chunk.sh
```

Дефолтный output root:

`/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/sdc/benchmark_16_baselines_test`

Отчеты:

- chunk: `.../reports/chunk_chunk_<start>_<size>.json`
- cumulative: `.../reports/cumulative.json`

Markdown-таблицы (в стиле mini benchmark report):

```bash
python pdd-bench/scripts/per_sign_bench/generate_chunk_markdown_report.py \
  --run-root /home/gbuhtuev/sdc/pdd-bench/scripts/per_sign_bench/benchmark_16_baselines_test \
  --chunk-name chunk_0_200
```

По умолчанию файл пишется в:

- `.../reports/report_chunk_<start>_<size>.md`

Правила success и compliance:

- `horizon` фиксируется в `600`;
- episode считается success, если `steps >= horizon`;
- для `NO_ENTRY_SIGNS = {3.1, 3.2, 3.18.1, 3.18.2, 3.19}`
  кейс `dest=false && sign_compliance=true` считается success;
- отдельно считаются SR:
  - sign compliance SR,
  - traffic-light SR,
  - crosswalk SR.
