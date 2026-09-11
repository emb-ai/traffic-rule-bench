# Map overlap analysis (train / test)

Cross-sign reuse moderate (9.3% of train places); global train∩test = 16 (0.8% of train union).

Audits **geographic map reuse** under the tiered assign policy
(unique → same behavioral family → same semantic group; no cross-semantic).

Also see [`allocation_verify.md`](allocation_verify.md) for counts / topology checks.

## How a “place” is defined

| Crop family | Place id |
| --- | --- |
| junction / dual_path / roundabout | `junction:<junction_id>` |
| segment (speed, detour, crosswalk, …) | `way:<osm_way_id>` |

Sources: `data/scenes/<sign>/moscow_pool.json`, enriched from `meta.json`.

## Verdict

- Train cross-sign place reuse is **moderate**: 180/1941 places (9.3%) appear under ≥2 signs.
- Train↔test leakage is **present**: global train∩test = 16 places (0.8% of the train union); within-sign leaked place-instances = 0.
- Map inventory size: train union **1941** places, test union **497** places across all signs.

### Interpretation

- **Within behavioral family** reuse (e.g. `direction_control` 4.1.1–4.1.6) is **by design**:
  same place, different ego rule. Avg shared-% in `direction_control` (train): **59.9%**.
- **Across semantic groups** should be **0** under the new assign policy.
- **Train↔test** place leak must be **0** (same-sign sum=0, cross-sign cell sum=19).

## Headline numbers

| Metric | Value |
| --- | ---: |
| Signs | 29 |
| Pool records | 2901 |
| Train place union | 1941 |
| Train places shared by ≥2 signs | 180 (9.3%) |
| Test place union | 497 |
| Test places shared by ≥2 signs | 39 (7.8%) |
| Global train∩test places | 16 |
| Within-sign train∩test places | 0 |
| Mean off-diagonal train pairwise | 1.07 |

## Reuse buckets (policy taxonomy)

### Train

| Bucket | # places | % |
| --- | ---: | ---: |
| unique | 1761 | 90.7% |
| within_behavioral | 109 | 5.6% |
| within_semantic_diff_family | 50 | 2.6% |
| across_semantic | 21 | 1.1% |

### Test

| Bucket | # places | % |
| --- | ---: | ---: |
| unique | 458 | 92.2% |
| within_behavioral | 27 | 5.4% |
| within_semantic_diff_family | 11 | 2.2% |
| across_semantic | 1 | 0.2% |

## Train place reuse histogram

| # signs sharing place | # places |
| --- | ---: |
| 1 | 1761 |
| 2 | 125 |
| 3 | 30 |
| 4 | 13 |
| 5 | 8 |
| 6 | 4 |

### Test

| # signs sharing place | # places |
| --- | ---: |
| 1 | 458 |
| 2 | 26 |
| 3 | 7 |
| 4 | 2 |
| 5 | 2 |
| 6 | 2 |

## Per-sign pool sizes

| Sign | Behavioral family | Train places | Test places | Train scenes | Test scenes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bike_lane` | restricted_lane | 80 | 20 | 80 | 20 |
| `bike_lane_road` | restricted_lane | 80 | 20 | 80 | 20 |
| `blocked_road` | access_road_direction | 80 | 20 | 80 | 20 |
| `bus_lane` | restricted_lane | 80 | 20 | 80 | 20 |
| `bus_lane_road` | restricted_lane | 80 | 20 | 80 | 20 |
| `crosswalk` | pedestrian_crossing | 80 | 20 | 80 | 20 |
| `detour_either` | obstacle_avoidance | 80 | 20 | 80 | 20 |
| `detour_left` | obstacle_avoidance | 80 | 20 | 80 | 20 |
| `detour_right` | obstacle_avoidance | 80 | 20 | 80 | 20 |
| `direction_left` | direction_control | 76 | 20 | 80 | 20 |
| `direction_left_right` | direction_control | 75 | 20 | 80 | 20 |
| `direction_right` | direction_control | 73 | 20 | 80 | 20 |
| `direction_straight` | direction_control | 68 | 16 | 80 | 21 |
| `direction_straight_left` | direction_control | 59 | 15 | 80 | 20 |
| `direction_straight_right` | direction_control | 50 | 13 | 80 | 20 |
| `main_road` | junction_priority | 80 | 20 | 80 | 20 |
| `min_speed` | speed_control | 80 | 20 | 80 | 20 |
| `no_entry` | access_road_direction | 69 | 18 | 80 | 20 |
| `no_turn_left` | turn_restriction | 73 | 20 | 80 | 20 |
| `no_turn_right` | turn_restriction | 74 | 19 | 80 | 20 |
| `one_way_left` | access_road_direction | 80 | 20 | 80 | 20 |
| `one_way_right` | access_road_direction | 80 | 20 | 80 | 20 |
| `residential_zone` | speed_control | 80 | 20 | 80 | 20 |
| `roundabout` | roundabout | 80 | 20 | 80 | 20 |
| `secondary_road` | junction_priority | 80 | 20 | 80 | 20 |
| `speed_limit` | speed_control | 80 | 20 | 80 | 20 |
| `stop` | junction_priority | 80 | 20 | 80 | 20 |
| `yield` | junction_priority | 80 | 20 | 80 | 20 |
| `zone_speed_limit` | speed_control | 80 | 20 | 80 | 20 |

## Per-sign unique vs shared (train)

| Sign | Behavioral family | Unique | Shared | Total | Shared % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direction_straight_right` | direction_control | 4 | 46 | 50 | 92.0% |
| `direction_straight_left` | direction_control | 12 | 47 | 59 | 79.7% |
| `direction_straight` | direction_control | 25 | 43 | 68 | 63.2% |
| `direction_right` | direction_control | 42 | 31 | 73 | 42.5% |
| `direction_left` | direction_control | 44 | 32 | 76 | 42.1% |
| `direction_left_right` | direction_control | 45 | 30 | 75 | 40.0% |
| `one_way_right` | access_road_direction | 55 | 25 | 80 | 31.2% |
| `detour_either` | obstacle_avoidance | 59 | 21 | 80 | 26.2% |
| `one_way_left` | access_road_direction | 59 | 21 | 80 | 26.2% |
| `no_turn_left` | turn_restriction | 54 | 19 | 73 | 26.0% |
| `bus_lane_road` | restricted_lane | 60 | 20 | 80 | 25.0% |
| `detour_right` | obstacle_avoidance | 61 | 19 | 80 | 23.8% |
| `no_turn_right` | turn_restriction | 57 | 17 | 74 | 23.0% |
| `detour_left` | obstacle_avoidance | 63 | 17 | 80 | 21.2% |
| `bike_lane_road` | restricted_lane | 66 | 14 | 80 | 17.5% |
| `no_entry` | access_road_direction | 57 | 12 | 69 | 17.4% |
| `crosswalk` | pedestrian_crossing | 69 | 11 | 80 | 13.8% |
| `bus_lane` | restricted_lane | 70 | 10 | 80 | 12.5% |
| `bike_lane` | restricted_lane | 72 | 8 | 80 | 10.0% |
| `min_speed` | speed_control | 75 | 5 | 80 | 6.2% |
| `blocked_road` | access_road_direction | 77 | 3 | 80 | 3.8% |
| `zone_speed_limit` | speed_control | 77 | 3 | 80 | 3.8% |
| `speed_limit` | speed_control | 78 | 2 | 80 | 2.5% |
| `main_road` | junction_priority | 80 | 0 | 80 | 0.0% |
| `residential_zone` | speed_control | 80 | 0 | 80 | 0.0% |
| `roundabout` | roundabout | 80 | 0 | 80 | 0.0% |
| `secondary_road` | junction_priority | 80 | 0 | 80 | 0.0% |
| `stop` | junction_priority | 80 | 0 | 80 | 0.0% |
| `yield` | junction_priority | 80 | 0 | 80 | 0.0% |

## Per-sign unique vs shared (test)

| Sign | Unique | Shared | Total | Shared % |
| --- | ---: | ---: | ---: | ---: |
| `direction_straight_right` | 0 | 13 | 13 | 100.0% |
| `direction_straight` | 3 | 13 | 16 | 81.2% |
| `direction_straight_left` | 3 | 12 | 15 | 80.0% |
| `no_entry` | 10 | 8 | 18 | 44.4% |
| `direction_left` | 12 | 8 | 20 | 40.0% |
| `direction_left_right` | 13 | 7 | 20 | 35.0% |
| `no_turn_left` | 13 | 7 | 20 | 35.0% |
| `direction_right` | 14 | 6 | 20 | 30.0% |
| `one_way_left` | 14 | 6 | 20 | 30.0% |
| `one_way_right` | 14 | 6 | 20 | 30.0% |
| `no_turn_right` | 15 | 4 | 19 | 21.1% |
| `detour_right` | 16 | 4 | 20 | 20.0% |
| `detour_either` | 17 | 3 | 20 | 15.0% |
| `detour_left` | 17 | 3 | 20 | 15.0% |
| `blocked_road` | 19 | 1 | 20 | 5.0% |
| `bus_lane` | 19 | 1 | 20 | 5.0% |
| `crosswalk` | 19 | 1 | 20 | 5.0% |
| `bike_lane` | 20 | 0 | 20 | 0.0% |
| `bike_lane_road` | 20 | 0 | 20 | 0.0% |
| `bus_lane_road` | 20 | 0 | 20 | 0.0% |
| `main_road` | 20 | 0 | 20 | 0.0% |
| `min_speed` | 20 | 0 | 20 | 0.0% |
| `residential_zone` | 20 | 0 | 20 | 0.0% |
| `roundabout` | 20 | 0 | 20 | 0.0% |
| `secondary_road` | 20 | 0 | 20 | 0.0% |
| `speed_limit` | 20 | 0 | 20 | 0.0% |
| `stop` | 20 | 0 | 20 | 0.0% |
| `yield` | 20 | 0 | 20 | 0.0% |
| `zone_speed_limit` | 20 | 0 | 20 | 0.0% |

## Behavioral family roll-up (train)

| Family | Unique | Shared across families | Total | Shared % |
| --- | ---: | ---: | ---: | ---: |
| `junction_priority` | 320 | 0 | 320 | 0.0% |
| `restricted_lane` | 268 | 52 | 320 | 16.2% |
| `speed_control` | 310 | 10 | 320 | 3.1% |
| `access_road_direction` | 273 | 10 | 283 | 3.5% |
| `direction_control` | 236 | 15 | 251 | 6.0% |
| `obstacle_avoidance` | 194 | 31 | 225 | 13.8% |
| `turn_restriction` | 120 | 13 | 133 | 9.8% |
| `pedestrian_crossing` | 69 | 11 | 80 | 13.8% |
| `roundabout` | 80 | 0 | 80 | 0.0% |

## Behavioral family place overlap (train)

| Family A | Family B | # shared places |
| --- | ---: | ---: |
| `obstacle_avoidance` | `restricted_lane` | 31 |
| `pedestrian_crossing` | `restricted_lane` | 11 |
| `restricted_lane` | `speed_control` | 10 |
| `direction_control` | `turn_restriction` | 9 |
| `access_road_direction` | `direction_control` | 6 |
| `access_road_direction` | `turn_restriction` | 4 |

## Semantic group place overlap (train)

| Group A | Group B | # shared places |
| --- | ---: | ---: |
| `obstacle` | `priority` | 11 |
| `obstacle` | `speed` | 10 |
| `obstacle` | `reroute` | 3 |

## Top overlapping sign pairs (train)

| # shared places | Sign A | Sign B |
| --- | ---: | ---: |
| 29 | `direction_straight` | `direction_straight_left` |
| 26 | `direction_straight_left` | `direction_straight_right` |
| 24 | `direction_straight` | `direction_straight_right` |
| 23 | `direction_left` | `direction_straight_left` |
| 22 | `direction_left` | `direction_straight_right` |
| 20 | `direction_right` | `direction_straight_right` |
| 19 | `direction_left_right` | `direction_straight` |
| 19 | `direction_right` | `direction_straight_left` |
| 18 | `direction_left_right` | `direction_straight_left` |
| 18 | `one_way_left` | `one_way_right` |
| 17 | `direction_left` | `direction_right` |
| 15 | `direction_left` | `direction_straight` |
| 14 | `direction_left` | `direction_left_right` |
| 14 | `direction_left_right` | `direction_straight_right` |
| 14 | `direction_right` | `direction_straight` |

## Train↔test leakage detail

### Within-sign

_None._

### Global train∩test sample

`way:125353952`, `way:130641823`, `way:131035804`, `way:1370317707`, `way:1536787928`, `way:16181854`, `way:182624938`, `way:25152718`, `way:28741271`, `way:289108100`, `way:40308143`, `way:428757205`, `way:45082597`, `way:699115626`, `way:783040941`, `way:85709235`

## Figures

All PNGs under [`figures/`](figures/).

| File | Meaning |
| --- | --- |
| `figures/train_pairwise_intersection.png` | Off-diagonal shared train places |
| `figures/test_pairwise_intersection.png` | Off-diagonal shared test places |
| `figures/train_top_sign_pairs.png` | Top train sign pairs |
| `figures/test_top_sign_pairs.png` | Top test sign pairs |
| `figures/train_unique_vs_shared.png` | Per sign unique vs shared (train) |
| `figures/test_unique_vs_shared.png` | Per sign unique vs shared (test) |
| `figures/train_reuse_buckets.png` | Policy reuse buckets (train) |
| `figures/test_reuse_buckets.png` | Policy reuse buckets (test) |
| `figures/train_behavioral_pairwise.png` | Shared places between behavioral families |
| `figures/test_behavioral_pairwise.png` | Same for test |
| `figures/train_semantic_pairwise.png` | Shared places between semantic groups |
| `figures/test_semantic_pairwise.png` | Same for test |
| `figures/train_behavioral_unique_vs_shared.png` | Behavioral family unique vs shared |
| `figures/train_place_degree.png` | Degree histogram (train) |
| `figures/test_place_degree.png` | Degree histogram (test) |
| `figures/per_sign_pool_sizes.png` | Train/test place counts |
| `figures/train_scenes_vs_places.png` | Scenes vs collapsed places |
| `figures/train_vs_test_cross_sign.png` | Train(row) ∩ Test(col) |
| `figures/within_sign_train_test_leak.png` | Same-sign split leakage |
| `figures/top_shared_train_places.png` | Most-reused train places |
| `figures/global_train_test_places.png` | Global split coverage |

## Reproduce

```bash
python -m traffic_bench.scene_collection analysis overlap
python -m traffic_bench.scene_collection analysis assign_verify
```
