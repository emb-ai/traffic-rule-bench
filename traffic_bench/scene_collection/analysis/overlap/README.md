# Map overlap analysis (train / test)

Cross-sign reuse moderate (5.9% of train places); global train∩test = 0 (0.0% of train union).

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

- Train cross-sign place reuse is **moderate**: 106/1789 places (5.9%) appear under ≥2 signs.
- Train↔test leakage is **clean**: no place appears in both splits (neither within a sign nor globally).
- Map inventory size: train union **1789** places, test union **450** places across all signs.

### Interpretation

- **Within behavioral family** reuse (e.g. `direction_control` 4.1.1–4.1.6) is **by design**:
  same place, different ego rule. Avg shared-% in `direction_control` (train): **56.6%**.
- **Across semantic groups** should be **0** under the new assign policy.
- **Train↔test** place leak must be **0** (same-sign sum=0, cross-sign cell sum=0).

## Headline numbers

| Metric | Value |
| --- | ---: |
| Signs | 25 |
| Pool records | 2500 |
| Train place union | 1789 |
| Train places shared by ≥2 signs | 106 (5.9%) |
| Test place union | 450 |
| Test places shared by ≥2 signs | 23 (5.1%) |
| Global train∩test places | 0 |
| Within-sign train∩test places | 0 |
| Mean off-diagonal train pairwise | 0.87 |

## Reuse buckets (policy taxonomy)

### Train

| Bucket | # places | % |
| --- | ---: | ---: |
| unique | 1683 | 94.1% |
| within_behavioral | 106 | 5.9% |
| within_semantic_diff_family | 0 | 0.0% |
| across_semantic | 0 | 0.0% |

### Test

| Bucket | # places | % |
| --- | ---: | ---: |
| unique | 427 | 94.9% |
| within_behavioral | 20 | 4.4% |
| within_semantic_diff_family | 3 | 0.7% |
| across_semantic | 0 | 0.0% |

## Train place reuse histogram

| # signs sharing place | # places |
| --- | ---: |
| 1 | 1683 |
| 2 | 65 |
| 3 | 22 |
| 4 | 16 |
| 5 | 2 |
| 6 | 1 |

### Test

| # signs sharing place | # places |
| --- | ---: |
| 1 | 427 |
| 2 | 13 |
| 3 | 7 |
| 4 | 1 |
| 5 | 2 |

## Per-sign pool sizes

| Sign | Behavioral family | Train places | Test places | Train scenes | Test scenes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blocked_road` | access_road_direction | 80 | 20 | 80 | 20 |
| `crosswalk` | pedestrian_crossing | 80 | 20 | 80 | 20 |
| `detour_either` | obstacle_avoidance | 80 | 20 | 80 | 20 |
| `detour_left` | obstacle_avoidance | 80 | 20 | 80 | 20 |
| `detour_right` | obstacle_avoidance | 80 | 20 | 80 | 20 |
| `direction_left` | direction_control | 79 | 18 | 80 | 20 |
| `direction_left_right` | direction_control | 76 | 20 | 80 | 20 |
| `direction_right` | direction_control | 80 | 19 | 80 | 20 |
| `direction_straight` | direction_control | 80 | 20 | 80 | 20 |
| `direction_straight_left` | direction_control | 60 | 15 | 80 | 20 |
| `direction_straight_right` | direction_control | 64 | 16 | 80 | 20 |
| `main_road` | junction_priority | 80 | 20 | 80 | 20 |
| `min_speed` | speed_control | 80 | 20 | 80 | 20 |
| `no_entry` | access_road_direction | 80 | 20 | 80 | 20 |
| `no_turn_left` | turn_restriction | 80 | 20 | 80 | 20 |
| `no_turn_right` | turn_restriction | 80 | 20 | 80 | 20 |
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
| `direction_straight_right` | direction_control | 10 | 54 | 64 | 84.4% |
| `direction_straight_left` | direction_control | 11 | 49 | 60 | 81.7% |
| `direction_straight` | direction_control | 29 | 51 | 80 | 63.8% |
| `direction_left_right` | direction_control | 45 | 31 | 76 | 40.8% |
| `direction_right` | direction_control | 52 | 28 | 80 | 35.0% |
| `direction_left` | direction_control | 52 | 27 | 79 | 34.2% |
| `one_way_left` | access_road_direction | 70 | 10 | 80 | 12.5% |
| `detour_either` | obstacle_avoidance | 72 | 8 | 80 | 10.0% |
| `one_way_right` | access_road_direction | 73 | 7 | 80 | 8.8% |
| `detour_left` | obstacle_avoidance | 76 | 4 | 80 | 5.0% |
| `detour_right` | obstacle_avoidance | 76 | 4 | 80 | 5.0% |
| `no_entry` | access_road_direction | 77 | 3 | 80 | 3.8% |
| `blocked_road` | access_road_direction | 80 | 0 | 80 | 0.0% |
| `crosswalk` | pedestrian_crossing | 80 | 0 | 80 | 0.0% |
| `main_road` | junction_priority | 80 | 0 | 80 | 0.0% |
| `min_speed` | speed_control | 80 | 0 | 80 | 0.0% |
| `no_turn_left` | turn_restriction | 80 | 0 | 80 | 0.0% |
| `no_turn_right` | turn_restriction | 80 | 0 | 80 | 0.0% |
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
| `direction_straight_right` | 3 | 13 | 16 | 81.2% |
| `direction_straight_left` | 3 | 12 | 15 | 80.0% |
| `direction_straight` | 7 | 13 | 20 | 65.0% |
| `direction_left_right` | 10 | 10 | 20 | 50.0% |
| `direction_right` | 13 | 6 | 19 | 31.6% |
| `direction_left` | 14 | 4 | 18 | 22.2% |
| `no_entry` | 18 | 2 | 20 | 10.0% |
| `no_turn_right` | 19 | 1 | 20 | 5.0% |
| `blocked_road` | 20 | 0 | 20 | 0.0% |
| `crosswalk` | 20 | 0 | 20 | 0.0% |
| `detour_either` | 20 | 0 | 20 | 0.0% |
| `detour_left` | 20 | 0 | 20 | 0.0% |
| `detour_right` | 20 | 0 | 20 | 0.0% |
| `main_road` | 20 | 0 | 20 | 0.0% |
| `min_speed` | 20 | 0 | 20 | 0.0% |
| `no_turn_left` | 20 | 0 | 20 | 0.0% |
| `one_way_left` | 20 | 0 | 20 | 0.0% |
| `one_way_right` | 20 | 0 | 20 | 0.0% |
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
| `access_road_direction` | 310 | 0 | 310 | 0.0% |
| `direction_control` | 287 | 0 | 287 | 0.0% |
| `obstacle_avoidance` | 232 | 0 | 232 | 0.0% |
| `turn_restriction` | 160 | 0 | 160 | 0.0% |
| `pedestrian_crossing` | 80 | 0 | 80 | 0.0% |
| `roundabout` | 80 | 0 | 80 | 0.0% |

## Behavioral family place overlap (train)

| Family A | Family B | # shared places |
| --- | ---: | ---: |
| — | — | 0 |

## Semantic group place overlap (train)

| Group A | Group B | # shared places |
| --- | ---: | ---: |
| — | — | 0 |

## Top overlapping sign pairs (train)

| # shared places | Sign A | Sign B |
| --- | ---: | ---: |
| 30 | `direction_straight` | `direction_straight_right` |
| 30 | `direction_straight_left` | `direction_straight_right` |
| 27 | `direction_straight` | `direction_straight_left` |
| 23 | `direction_left_right` | `direction_straight` |
| 23 | `direction_right` | `direction_straight_right` |
| 21 | `direction_left` | `direction_straight_left` |
| 20 | `direction_left` | `direction_straight_right` |
| 16 | `direction_right` | `direction_straight_left` |
| 13 | `direction_left_right` | `direction_straight_left` |
| 10 | `direction_left_right` | `direction_straight_right` |
| 9 | `direction_left` | `direction_right` |
| 8 | `direction_left` | `direction_straight` |
| 7 | `one_way_left` | `one_way_right` |
| 6 | `direction_left` | `direction_left_right` |
| 5 | `direction_left_right` | `direction_right` |

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
