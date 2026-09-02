# Allocation verification

Policy: `tiered_place_reuse` · signs: 25

| Check | Result |
| --- | --- |
| Train↔test place overlap = 0 | **PASS** (global=0, within-sign=0) |
| Per-sign scene counts = 80/20 | **PASS** |
| Cross-semantic reuse = 0 | **PASS** |
| Shortfalls | 0 |

## Train↔test

- train place union: 1648
- test place union: 422
- global train∩test: 0
- within-sign leaks: none

## Per-sign counts + topology

| PDD | Behavioral family | Crop | Train scenes | Test scenes | Train places | Test places | Train topo | Test topo |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `3.2` | `access_road_direction` | `junction` | 80 | 20 | 80 | 20 | `{'T': 41, 'X': 39}` | `{'T': 10, 'X': 10}` |
| `4.2.1` | `obstacle_avoidance` | `segment` | 80 | 20 | 78 | 20 | `{'straight': 39, 'curved': 41}` | `{'curved': 9, 'straight': 11}` |
| `4.2.2` | `obstacle_avoidance` | `segment` | 80 | 20 | 80 | 20 | `{'straight': 38, 'curved': 42}` | `{'curved': 11, 'straight': 9}` |
| `4.2.3` | `obstacle_avoidance` | `segment` | 80 | 20 | 79 | 20 | `{'straight': 42, 'curved': 38}` | `{'curved': 9, 'straight': 11}` |
| `2.1` | `junction_priority` | `junction` | 80 | 20 | 80 | 20 | `{'T': 37, 'X': 43}` | `{'T': 10, 'X': 10}` |
| `2.3` | `junction_priority` | `junction` | 80 | 20 | 80 | 20 | `{'T': 40, 'X': 40}` | `{'X': 11, 'T': 9}` |
| `2.4` | `junction_priority` | `junction` | 80 | 20 | 80 | 20 | `{'T': 39, 'X': 41}` | `{'X': 10, 'T': 10}` |
| `2.5` | `junction_priority` | `junction` | 80 | 20 | 80 | 20 | `{'T': 39, 'X': 41}` | `{'T': 10, 'X': 10}` |
| `4.3` | `roundabout` | `junction` | 80 | 20 | 80 | 20 | `{'O': 80}` | `{'O': 20}` |
| `5.19` | `pedestrian_crossing` | `segment` | 80 | 20 | 80 | 20 | `{'curved': 38, 'straight': 42}` | `{'straight': 9, 'curved': 11}` |
| `3.1` | `access_road_direction` | `dual_path` | 80 | 20 | 69 | 18 | `{'T': 38, 'X': 42}` | `{'T': 8, 'X': 12}` |
| `3.18.1` | `turn_restriction` | `dual_path` | 80 | 20 | 72 | 20 | `{'T': 37, 'X': 43}` | `{'T': 11, 'X': 9}` |
| `3.18.2` | `turn_restriction` | `dual_path` | 80 | 20 | 76 | 20 | `{'T': 41, 'X': 39}` | `{'T': 12, 'X': 8}` |
| `4.1.1` | `direction_control` | `dual_path` | 80 | 20 | 70 | 16 | `{'X': 80}` | `{'X': 20}` |
| `4.1.2` | `direction_control` | `dual_path` | 80 | 20 | 72 | 19 | `{'T': 39, 'X': 41}` | `{'T': 9, 'X': 11}` |
| `4.1.3` | `direction_control` | `dual_path` | 80 | 20 | 77 | 20 | `{'T': 34, 'X': 46}` | `{'T': 11, 'X': 9}` |
| `4.1.4` | `direction_control` | `dual_path` | 80 | 20 | 56 | 15 | `{'X': 80}` | `{'X': 20}` |
| `4.1.5` | `direction_control` | `dual_path` | 80 | 20 | 61 | 15 | `{'X': 80}` | `{'X': 20}` |
| `4.1.6` | `direction_control` | `dual_path` | 80 | 20 | 75 | 20 | `{'T': 37, 'X': 43}` | `{'T': 11, 'X': 9}` |
| `5.7.1` | `access_road_direction` | `dual_path` | 80 | 20 | 80 | 20 | `{'T': 80}` | `{'T': 20}` |
| `5.7.2` | `access_road_direction` | `dual_path` | 80 | 20 | 80 | 20 | `{'T': 80}` | `{'T': 20}` |
| `3.24` | `speed_control` | `segment` | 80 | 20 | 80 | 20 | `{'straight': 80}` | `{'straight': 20}` |
| `4.6` | `speed_control` | `segment` | 80 | 20 | 80 | 20 | `{'straight': 80}` | `{'straight': 20}` |
| `5.21` | `speed_control` | `segment` | 80 | 20 | 80 | 20 | `{'straight': 80}` | `{'straight': 20}` |
| `5.31` | `speed_control` | `segment` | 80 | 20 | 80 | 20 | `{'straight': 80}` | `{'straight': 20}` |

## Place reuse (train)

| Bucket | # places | % of all | note |
| --- | ---: | ---: | --- |
| unique | 1683 | 94.1% | — |
| within behavioral family | 106 | 5.9% | 100.0% of shared |
| within semantic, different family | 0 | 0.0% | 0.0% of shared |
| across semantic groups | 0 | 0.0% | 0.0% of shared |

## Place reuse (test)

| Bucket | # places | % of all | note |
| --- | ---: | ---: | --- |
| unique | 427 | 94.9% | — |
| within behavioral family | 20 | 4.4% | 87.0% of shared |
| within semantic, different family | 3 | 0.7% | 13.0% of shared |
| across semantic groups | 0 | 0.0% | 0.0% of shared |

## Reproduce

```bash
python -m traffic_bench.scene_collection assign
python -m traffic_bench.scene_collection analysis assign_verify
# or as part of:
python -m traffic_bench.scene_collection analysis overlap
```
