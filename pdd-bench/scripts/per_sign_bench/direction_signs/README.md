# Direction Signs (4.1.1–4.1.6)

Один пакет для всей группы предписывающих знаков направления. Члены семейства
отличаются только разрешёнными направлениями (`allowed_dirs`); каркас сцен /
манifest / бенчмарка общий.

Спецификация семьи: `lib/direction_sign_spec.py`.

| Код   | Знак                         | `allowed_dirs` |
|-------|------------------------------|----------------|
| 4.1.1 | Движение прямо               | `s`            |
| 4.1.2 | Движение направо             | `r`            |
| 4.1.3 | Движение налево              | `l` (+разворот)|
| 4.1.4 | Движение прямо или направо   | `s`, `r`       |
| 4.1.5 | Движение прямо или налево    | `s`, `l`       |
| 4.1.6 | Движение направо или налево  | `l`, `r`       |

Сейчас по умолчанию активен **4.1.1**. Фильтрация маршрутов / генерация сцен
по направлению ещё не реализована — только подготовка общей структуры.

## Setup

```bash
conda activate zinkovich-plant2
cd pdd-bench/scripts/per_sign_bench/direction_signs
```

## Folder structure

```
direction_signs/
├── build_scene.py
├── generate_manifest.py      # Hydra; sign.pdd_code выбирает члена семьи
├── eval_pipeline.py
├── run_benchmark.py          # Ставит LaneAllowedDirectionSign4_1_* на ego
├── lib/
│   ├── direction_sign_spec.py   # реестр 4.1.1–4.1.6
│   ├── junction_*.py            # общая junction-топология (как main/stop)
│   └── …
├── tools/filter_scenes/      # import/crop из каталога pdd-bench/scenes/<code>/
├── config/config.yaml
├── scenes/
└── benchmark_output/
```

## Workflow (пока каркас)

1. Импорт из каталога (по умолчанию `scenes/4.1.1`):

```bash
python tools/filter_scenes/import_catalog_scenes.py --limit 10
```

2. Crop пересечений:

```bash
python tools/filter_scenes/crop_junction_scene.py --limit 5
```

3. Manifest (активный знак из конфига):

```bash
python generate_manifest.py
# другой член семьи:
python generate_manifest.py sign.pdd_code=4.1.2 paths.output_base=benchmark_output/4_1_2
```

4. Eval:

```bash
python eval_pipeline.py \
    --policies idm \
    --manifest benchmark_output/4_1_1/<timestamp> \
    --scenes-root scenes
```

## Next

- Генерация / отбор сцен и ego-destination только по `allowed_dirs` знака
- Разделение каталогов/сидов по шести кодам при необходимости
