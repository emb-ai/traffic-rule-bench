# PlanT2 FT: входы модели

Документ описывает **то, что реально подаётся в сеть** при fine-tune по пайплайну
`plant2_ft_pipeline`. Источник правды: `plant2/PlanT/dataset.py` → `generate_batch()`
→ `model.py:forward()`.

---

## Общая схема одного training sample

```
dump на диск                    PlanTDataset.__getitem__           generate_batch()              HFLM.forward()
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
boxes/NNNN.json.gz      →      sample["input"]  (список obj)  →   batch["x_objs"]      →      tok_emb[class](feats)
measurements/NNNN.json  →      sample["route*"], targets      →   batch["route"], …    →      route_emb, sign_emb, …
bev_no_car_semantics/   →      sample["BEV"]  (3×H×W float)  →   batch["BEV"]         →      ResNet18 → bev_tok
split_meta.json         →      sample["sign_id"] (int)        →   batch["sign_id"]     →      sign_emb
```

Координатная система: **ego-centric**, origin в ego на кадре захвата.

| поле | convention |
|------|------------|
| `x_objs` x, y | **CARLA**: +x вперёд, **+y вправо** (см. `plant2_frames._ego_xy`) |
| `route` y | **y влево** (в dump инвертируется в `get_route()`) |
| BEV | ego в центре crop 128²; span **32 m** (±16 m), не 64 m |

`x_objs` и BEV снимаются **на одном кадре** (`boxes/NNNN` + `bev/NNNN` при одном `seq`), поэтому координаты объектов — уже относительно ego **этого** кадра, без дополнительной трансформации.

BEV в модель: PNG 256² → `rot90` → crop `[64:-64]` → 128² RGB. Overlay маппит CARLA (x,y) через ту же цепочку, что `render_bev_plant2.ego_to_pix` + `PlanTDataset`.

**Важно:** `x_objs` фильтрует до ~50 m, а BEV crop — только ±16 m; дальние объекты есть в `x_objs`, но **не попадают** на BEV-пиксели (в GIF: `oob=N`).

---

## 1. Объекты: `x_objs` (в коде также `batch["x_objs"]`)

### Откуда берутся

Файл `boxes/NNNN.json.gz` — массив dict'ов. Первый элемент — **ego car** (в `input` не попадает).
Остальные фильтруются и преобразуются в `dataset.py` (строки ~329–431).

### Формат одного объекта в `sample["input"]`

7 float (id отбрасывается перед записью в sample):

| idx | поле | единицы | описание |
|-----|------|---------|----------|
| 0 | **type_id** | int-as-float | индекс класса → `tok_emb[type_id]` |
| 1 | **x** | метры | вперёд от ego |
| 2 | **y** | метры | влево от ego |
| 3 | **yaw** | градусы | [-180, 180], 0 = вперёд |
| 4 | **speed** | **км/ч** | для динамики; 0 для static/sign/TL |
| 5 | **extent_y×2** | метры | ширина bbox |
| 6 | **extent_x×2** | метры | длина bbox |

### Таблица type_id (`PlanTVariables.class_nums`)

| type_id | class | попадает если |
|---------|-------|---------------|
| 0 | padding | служебный padding в батче |
| 1 | car | в радиусе (эллипс, см. ниже) |
| 2 | walker | то же |
| 3 | static | cone / traffic warning |
| 4 | stop_sign | legacy; `affects_ego=True`, ≤30 m |
| 5 | traffic_light | Red/Yellow + `affects_ego=True`, ≤30 m |
| 6 | emergency | police / fire / ambulance |
| 7… | PDD codes | `"2.1"`, `"2.5"`, `"3.24"`, … + `affects_ego=True`, ≤30 m |
| 1 | static_car | если `input_static_cars=True` (default FT: **True**) |

PDD-коды (`SIGN_CODES`): `2.1, 2.3.1, 2.3.2, 2.3.3, 2.4, 2.5, 3.1, 3.2, 3.24, 4.2.1, 4.2.2, 4.2.3, 4.3, 4.6, 5.7.1, 5.7.2, 5.15.1, 5.15.2, 5.19, 5.21, 5.31` — каждый со своим type_id начиная с 7.

### Пространственный фильтр

- **range** = 50 m (default), **range_factor_front** = 2 → впереди до ~100 m по эллипсу:
  - `x>0`: `(x/2)² + y² ≤ 50²`
  - `x≤0`: `x² + y² ≤ 50²`
- **Знаки / TL / PDD**: круг r ≤ 30 m, `|z| ≤ 30`
- Объекты вне зоны → class `"too far"` → **не попадают** в input

### Батчинг (`generate_batch`)

```python
batch["x_objs"]  # float32, shape (N_total+1, 7) — row 0 = padding [0,0,0,0,0,0,0]
batch["idxs"]    # int32,   shape (B, maxseq) — индексы строк в x_objs для каждого sample
```

В `forward()` берутся `class_ids = x_objs[..., 0]`, `obj_feats = x_objs[..., 1:]` (6 атрибутов),
каждый класс embed'ится своим `tok_emb[i]: Linear(6 → hidden)`.

### Forecast target (не вход, но из тех же obj)

`batch["y_objs"]` — quantized xy/yaw/speed на **t+1** (future_timestep=1), формат int32 после `quantize_box()`.

---

## 2. BEV: `batch["BEV"]`

### Откуда

PNG: `bev_no_car_semantics/NNNN.png` (ego без машины, semantic classes).

При `augment=True` (50% samples) — вместо base PNG подставляется
`bev_no_car_semantics_augmented/NNNN.png` **после** геом. aug на объектах/route.

### Преобразование (dataset.py:313–317)

```python
bev = pil_to_tensor(Image.open(png))      # (3, 256, 256) — grayscale in channel 0
bev = torch.rot90(bev, dims=(1, 2))       # поворот 90° CCW
idx = bev[0, 64:-64, 64:-64]              # crop центр 128×128
rgb = bev_colors[idx]                     # index → RGB float
sample["BEV"] = rgb.permute(2, 0, 1)      # (3, 128, 128), float32, ~ImageNet-scale colors
```

### Цвета семантики (`PlanTVariables.bev_colors`)

| class idx | семантика | RGB (≈) |
|-----------|-----------|---------|
| 0 | background / sidewalk | серый (ImageNet mean) |
| 1 | road | синий |
| 2 | sidewalk | серый |
| 3 | lane lines | красный |
| 4 | broken lines | зелёный |

### В модели

```python
bev_tok = resnet18(batch["BEV"])[:, None]   # → (B, 1, 512)
# concat в начало token sequence (после sign/route/speed tokens)
```

---

## 3. Прочие входы батча

| ключ | shape | dtype | источник | embedding |
|------|-------|-------|----------|-----------|
| `route_original` | (B, 20, 2) | float32 | `measurements.route_original[:20]` | `route_emb` → 1 token |
| `route` | (B, 20, 2) | float32 | интерполяция route каждые 1 m | loss path |
| `waypoints` | (B, 8, 2) | float32 | ego_matrix t→t+1…t+8 | loss wps |
| `target_speed` | (B,) | float32 | `measurements.target_speed` (0 если brake) | soft 2-hot CE, 8 bins |
| `ego_speed` | (B,) | float32 | `measurements.ego_speed` или `speed` | опционально `input_ego_speed` |
| `speed_limit` | (B,) | int64 | PDD limit → bin {50,80,100,120} | `speed_emb` → 1 token |
| `sign_id` | (B,) | int64 | route → PDD code (split_meta) | `sign_emb` → 1 token |
| `y_objs` | (N, 4) | int64 | forecast targets quantized | loss (forecast) |

### Speed bins (target / pred)

`[0, 4, 8, 10, 13.89, 16, 17.78, 20]` m/s — soft two-hot между соседними при обучении;
на inference — softmax expectation.

### Token order в transformer (forward)

```
[wp_tokens…] [sign_tok?] [ego_speed_tok?] [bev_tok?] [speed_limit_tok] [route_tok] [obj_tokens…] [speed_tok]
```

---

## 4. Что лежит в dump, но не входит в модель напрямую

| поле dump | использование |
|-----------|---------------|
| `ego_matrix`, `theta`, `pos_global` | построение ego frame, waypoints |
| `augmentation_translation/rotation` | geom aug (если augment=True) |
| `results.json.gz` | фильтр route при `filter_routes=True` |
| `boxes[*].id` | matching forecast; в x_objs не передаётся |
| `boxes[*].affects_ego` | фильтр signs/TL |

---

## 5. Diskcache

При `use_caching=True` в cache кладётся уже обработанный `sample` dict (без `sign_id` — добавляется в runtime).
Ключ = абсolute path `…/boxes/0005.json.gz`; aug = `…_aug`.

---

## 6. Визуализация

Скрипт `viz_train_global_gif.py` — GIF: BEV входа модели + все объекты из `boxes/`.

**Цепочка координат (static / знаки):**
1. `boxes` ego (x=fwd, y=right) → world: `ego_matrix @ [x, -y, 0, 1]`
2. Кластеризация по world xy (id в dump **нестабилен** между кадрами — один конус может быть id=9, потом id=10)
3. Фиксированный world-якорь кластера → текущий ego: `inv(ego_matrix) @ world` → пиксель BEV

BEV **ego-центричный**: знак/конус на экране двигается, пока ego проезжает — это нормально.
«Плывёт» относительно дороги = баг (раньше: якорь per-id; сейчас: per world-cluster).

```bash
source scripts/plant2_ft_pipeline/_env.sh

# 2.5 train (default route)
$PY scripts/plant2_ft_pipeline/viz_train_global_gif.py --fps 10

# sumo detour 4.2.1 — ego-centric BEV
$PY scripts/plant2_ft_pipeline/viz_train_global_gif.py \
  --route $SHEPELEV/plant2_l1_from_experts_signs/data/sumo_4.2.1_95616_lane1_seed2270015646_v0_default \
  --fps 10

# world-fixed map: BEV тайлы склеены в global canvas, static не двигается
$PY scripts/plant2_ft_pipeline/viz_train_global_gif.py \
  --route $SHEPELEV/plant2_l1_from_experts_signs/data/sumo_4.2.1_95616_lane1_seed2270015646_v0_default \
  --canvas world --fps 10
```

Выход: `train_global_<route>.gif` (ego) или `train_world_<route>.gif` (world).

---

## 7. Типичный FT config (spatial / 2.5 tsfix)

Из `PlanT.yaml` + overrides пайплайна:

```yaml
training:
  range: 50
  range_factor_front: 2
  seq_len: 1
  input_bev: True
  input_static_cars: True
  input_ego_speed: False
  augment: True          # FT: True; H5 ablation: False
  augment_parked: False  # FT pipeline: False (True только в base pretrain config)
  filter_routes: False   # обязательно на готовом split
```

---

## 8. Связанные файлы

| файл | роль |
|------|------|
| `plant2/PlanT/dataset.py` | сбор sample, quantize, batch |
| `plant2/PlanT/model.py` | forward, embeddings |
| `plant2/PlanT/plant_variables.py` | class_nums, bev_colors, speed_cats |
| `plant2/PlanT/util/sign_id.py` | SIGN_CODES, sign_id resolution |
| `plant2/PlanT/util/viz_batch.py` | debug viz (похож на наш GIF) |
| `pdd-bench/.../plant2_frames.py` | запись dump'ов |
