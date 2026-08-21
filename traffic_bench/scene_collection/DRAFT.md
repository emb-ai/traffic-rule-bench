Сначала всем перекрёсткам Москвы вешают ярлык: train или test. Один перекрёсток — один ярлык навсегда.

1. Нашли все подходящие перекрёстки Москвы: 6457 (T 5181, X 1052, O 224). коридоров 7620
2. Им всем повесили ярлык train/test (~80/20).
3. Вырезали все возможные T/X/O/Dual_path/Segment.

- T
- X
- O
- Dual_path
- Segment


The scene collection stage consists of two steps: 
1. the general pool of Moscow maps without signs (map_pool)

map_pool/ — a common pool of maps
Idea: slice Moscow once into geometry types (T/X intersection, two routes, straight section…) and then several signs share the same maps. Train/test is sliced by junction_id / osm_way_id so that pieces of the same road don’t end up in both halves.

2. the distribution of these maps by signs (sign_pool) in data/scenes/<sign>




## flowchart LR
  osm[raw OSM] --> net[moscow.net.xml]
  net --> index[index junctions/segments]
  index --> crops[crops T/X/O dual_path segment ...]
  crops --> split[train_ids / test_ids]
  split --> alloc[sign_allocations.json]
  alloc --> mat[materialize_scenes]
  mat --> dataScenes["data/scenes/yield, main_road, ..."]
