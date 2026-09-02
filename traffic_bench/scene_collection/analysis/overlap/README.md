# Map overlap analysis (train / test)

Cross-sign reuse moderate (9.3% of train places); global train∩test = 0 (0.0% of train union).

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

- Train cross-sign place reuse is **moderate**: 154/1648 places (9.3%) appear under ≥2 signs.
- Train↔test leakage is **clean**: no place appears in both splits (neither within a sign nor globally).
- Map inventory size: train union **1648** places, test union **422** places across all signs.

### Interpretation

- **Within behavioral family** reuse (e.g. `direction_control` 4.1.1–4.1.6) is **by design**:
  same place, different ego rule. Avg shared-% in `direction_control` (train): **61.5%**.
- **Across semantic groups** should be **0** under the new assign policy.
- **Train↔test** place leak must be **0** (same-sign sum=0, cross-sign cell sum=0).

## Headline numbers

| Metric | Value |
| --- | ---: |
| Signs | 25 |
| Pool records | 2500 |
| Train place union | 1648 |
| Train places shared by ≥2 signs | 154 (9.3%) |
| Test place union | 422 |
| Test places shared by ≥2 signs | 37 (8.8%) |
| Global train∩test places | 0 |
| Within-sign train∩test places | 0 |
| Mean off-diagonal train pairwise | 1.38 |

## Reuse buckets (policy taxonomy)

### Train

| Bucket | # places | % |
| --- | ---: | ---: |
| unique | 1494 | 90.7% |
| within_behavioral | 148 | 9.0% |
| within_semantic_diff_family | 6 | 0.4% |
| across_semantic | 0 | 0.0% |

### Test

| Bucket | # places | % |
| --- | ---: | ---: |
| unique | 385 | 91.2% |
| within_behavioral | 31 | 7.3% |
| within_semantic_diff_family | 6 | 1.4% |
| across_semantic | 0 | 0.0% |

## Train place reuse histogram

| # signs sharing place | # places |
| --- | ---: |
| 1 | 1494 |
| 2 | 88 |
| 3 | 42 |
| 4 | 14 |
| 5 | 7 |
| 6 | 3 |

### Test

| # signs sharing place | # places |
| --- | ---: |
| 1 | 385 |
| 2 | 25 |
| 3 | 5 |
| 4 | 3 |
| 5 | 3 |
| 6 | 1 |

## Per-sign pool sizes

| Sign | Behavioral family | Train places | Test places | Train scenes | Test scenes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blocked_road` | access_road_direction | 80 | 20 | 80 | 20 |
| `crosswalk` | pedestrian_crossing | 80 | 20 | 80 | 20 |
| `detour_either` | obstacle_avoidance | 79 | 20 | 80 | 20 |
| `detour_left` | obstacle_avoidance | 80 | 20 | 80 | 20 |
| `detour_right` | obstacle_avoidance | 78 | 20 | 80 | 20 |
| `direction_left` | direction_control | 77 | 20 | 80 | 20 |
| `direction_left_right` | direction_control | 75 | 20 | 80 | 20 |
| `direction_right` | direction_control | 72 | 19 | 80 | 20 |
| `direction_straight` | direction_control | 70 | 16 | 80 | 20 |
| `direction_straight_left` | direction_control | 61 | 15 | 80 | 20 |
| `direction_straight_right` | direction_control | 56 | 15 | 80 | 20 |
| `main_road` | junction_priority | 80 | 20 | 80 | 20 |
| `min_speed` | speed_control | 80 | 20 | 80 | 20 |
| `no_entry` | access_road_direction | 69 | 18 | 80 | 20 |
| `no_turn_left` | turn_restriction | 76 | 20 | 80 | 20 |
| `no_turn_right` | turn_restriction | 72 | 20 | 80 | 20 |
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
| `direction_straight_right` | direction_control | 6 | 50 | 56 | 89.3% |
| `direction_straight_left` | direction_control | 8 | 53 | 61 | 86.9% |
| `direction_straight` | direction_control | 24 | 46 | 70 | 65.7% |
| `direction_left` | direction_control | 41 | 36 | 77 | 46.8% |
| `direction_left_right` | direction_control | 44 | 31 | 75 | 41.3% |
| `direction_right` | direction_control | 44 | 28 | 72 | 38.9% |
| `detour_left` | obstacle_avoidance | 49 | 31 | 80 | 38.8% |
| `detour_either` | obstacle_avoidance | 51 | 28 | 79 | 35.4% |
| `detour_right` | obstacle_avoidance | 52 | 26 | 78 | 33.3% |
| `one_way_left` | access_road_direction | 59 | 21 | 80 | 26.2% |
| `no_turn_right` | turn_restriction | 55 | 17 | 72 | 23.6% |
| `no_turn_left` | turn_restriction | 59 | 17 | 76 | 22.4% |
| `one_way_right` | access_road_direction | 63 | 17 | 80 | 21.2% |
| `no_entry` | access_road_direction | 61 | 8 | 69 | 11.6% |
| `blocked_road` | access_road_direction | 78 | 2 | 80 | 2.5% |
| `crosswalk` | pedestrian_crossing | 80 | 0 | 80 | 0.0% |
| `main_road` | junction_priority | 80 | 0 | 80 | 0.0% |
| `min_speed` | speed_control | 80 | 0 | 80 | 0.0% |
| `residential_zone` | speed_control | 80 | 0 | 80 | 0.0% |
| `roundabout` | roundabout | 80 | 0 | 80 | 0.0% |
| `secondary_road` | junction_priority | 80 | 0 | 80 | 0.0% |
| `speed_limit` | speed_control | 80 | 0 | 80 | 0.0% |
| `stop` | junction_priority | 80 | 0 | 80 | 0.0% |
| `yield` | junction_priority | 80 | 0 | 80 | 0.0% |
| `zone_speed_limit` | speed_control | 80 | 0 | 80 | 0.0% |

## Per-sign unique vs shared (test)

| Sign | Unique | Shared | Total | Shared % |
| --- | ---: | ---: | ---: | ---: |
| `direction_straight_left` | 1 | 14 | 15 | 93.3% |
| `direction_straight_right` | 2 | 13 | 15 | 86.7% |
| `direction_straight` | 4 | 12 | 16 | 75.0% |
| `direction_right` | 12 | 7 | 19 | 36.8% |
| `direction_left` | 13 | 7 | 20 | 35.0% |
| `direction_left_right` | 13 | 7 | 20 | 35.0% |
| `no_entry` | 12 | 6 | 18 | 33.3% |
| `detour_either` | 14 | 6 | 20 | 30.0% |
| `one_way_left` | 15 | 5 | 20 | 25.0% |
| `detour_left` | 16 | 4 | 20 | 20.0% |
| `detour_right` | 16 | 4 | 20 | 20.0% |
| `no_turn_left` | 16 | 4 | 20 | 20.0% |
| `no_turn_right` | 16 | 4 | 20 | 20.0% |
| `one_way_right` | 16 | 4 | 20 | 20.0% |
| `blocked_road` | 19 | 1 | 20 | 5.0% |
| `crosswalk` | 20 | 0 | 20 | 0.0% |
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
| `speed_control` | 320 | 0 | 320 | 0.0% |
| `access_road_direction` | 284 | 2 | 286 | 0.7% |
| `direction_control` | 241 | 6 | 247 | 2.4% |
| `obstacle_avoidance` | 189 | 0 | 189 | 0.0% |
| `turn_restriction` | 128 | 4 | 132 | 3.0% |
| `pedestrian_crossing` | 80 | 0 | 80 | 0.0% |
| `roundabout` | 80 | 0 | 80 | 0.0% |

## Behavioral family place overlap (train)

| Family A | Family B | # shared places |
| --- | ---: | ---: |
| `direction_control` | `turn_restriction` | 4 |
| `access_road_direction` | `direction_control` | 2 |

## Semantic group place overlap (train)

| Group A | Group B | # shared places |
| --- | ---: | ---: |
| `obstacle` | `reroute` | 2 |

## Top overlapping sign pairs (train)

| # shared places | Sign A | Sign B |
| --- | ---: | ---: |
| 32 | `direction_straight_left` | `direction_straight_right` |
| 29 | `direction_straight` | `direction_straight_left` |
| 26 | `direction_left` | `direction_straight_left` |
| 26 | `direction_straight` | `direction_straight_right` |
| 24 | `direction_left` | `direction_straight_right` |
| 22 | `detour_either` | `detour_left` |
| 22 | `direction_right` | `direction_straight_right` |
| 20 | `detour_left` | `detour_right` |
| 20 | `direction_right` | `direction_straight_left` |
| 19 | `direction_left_right` | `direction_straight` |
| 19 | `direction_left_right` | `direction_straight_left` |
| 17 | `detour_either` | `detour_right` |
| 17 | `direction_left` | `direction_right` |
| 17 | `one_way_left` | `one_way_right` |
| 16 | `no_turn_left` | `no_turn_right` |

## Train↔test leakage detail

### Within-sign

_None._

### Global train∩test sample

_empty_

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
