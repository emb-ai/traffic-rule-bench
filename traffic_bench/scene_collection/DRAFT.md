Сначала всем местам Москвы вешают ярлык: train или test. Одно место — один ярлык навсегда (`junction_id` или `osm_way_id`).

1. Нашли все подходящие перекрёстки Москвы: 6457 (T 5181, X 1052, O 224) и коридоров 7620.
2. Им всем повесили ярлык train/test (~80/20) **до** раздачи знакам.
3. Вырезали все возможные T / X / O / Dual_path / Segment (H = P, без капа 500).

Знак = запрос к семейству, не отдельная папка кропа.

- T, X, O — junction
- Dual_path — другой net (union двух путей)
- Segment — коридор; straight/curved и число полос — теги

4.2 семплит Segment (`lane_count ≥ 2`, сторона объезда). Полосу препятствия ставит eval.
5.19 семплит Segment, зебру врезает materialize в `data/scenes/crosswalk/`.
5.15 в этой итерации не собираем.

The scene collection stage consists of two steps:

1. the general pool of Moscow maps without signs (`map_pool`)
2. the distribution of these maps by signs (`sign_pool`) in `data/scenes/<sign>`
