# Agents and duration per sign

| Sign | Mode | N scenarios | Mean agents | Std | Median | Min–Max | Horizon (s) | IDM (s) | IDM rule (s) | Plant2 (s) | Plant rule (s) | CARL (s) | CARL rule (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.1 | aux_convoy | 846 | 3.03 | 0.85 | 3.0 | 2–7 | 60 | 28.2 | 40.3 | 6.0 | 18.8 | 28.1 | 33.8 |
| 2.3.1 | aux_convoy | 90 | 3.80 | 1.56 | 3.0 | 2–7 | 60 | 21.7 | 46.5 | 5.3 | 30.5 | 16.5 | 30.0 |
| 2.3.2 | aux_convoy | 408 | 3.91 | 1.60 | 3.0 | 2–7 | 60 | 19.4 | 46.0 | 5.5 | 23.7 | 20.4 | 31.2 |
| 2.3.3 | aux_convoy | 444 | 3.95 | 1.62 | 3.0 | 2–7 | 60 | 16.9 | 47.0 | 5.3 | 23.5 | 22.3 | 34.1 |
| 2.4 | aux_convoy | 1095 | 4.02 | 1.78 | 3.0 | 2–13 | 60 | 14.8 | 40.8 | 5.5 | 22.3 | 22.9 | 33.8 |
| 2.5 | aux_convoy | 861 | 3.84 | 1.57 | 3.0 | 2–7 | 60 | 16.7 | 78.0 | 5.7 | 43.5 | 25.2 | 45.6 |
| 3.1 | density | 720 | 32.00 | 9.42 | 31.0 | 21–44 | 60 | 30.1 | 31.3 | 9.0 | 8.5 | 31.5 | 31.9 |
| 3.2 | density | 744 | 32.00 | 9.42 | 31.0 | 21–44 | 60 | 30.3 | 30.4 | 9.6 | 9.1 | 26.7 | 25.5 |
| 3.24 | speed_ego | 1200 | 1.00 | 0.00 | 1.0 | 1–1 | 150 | 23.2 | 25.7 | 12.3 | 17.8 | 29.9 | 33.1 |
| 4.3 | aux_convoy | 1015 | 5.61 | 3.12 | 5.0 | 2–21 | 60 | 20.6 | 50.9 | 9.6 | 30.9 | 19.6 | 46.9 |
| 4.2.1 | detour_ego | 645 | 1.00 | 0.00 | 1.0 | 1–1 | 120 | 81.9 | 28.8 | 8.7 | 19.1 | 30.2 | 32.5 |
| 4.2.2 | detour_ego | 504 | 1.00 | 0.00 | 1.0 | 1–1 | 120 | 72.8 | 28.4 | 13.2 | 16.4 | 12.7 | 20.4 |
| 4.2.3 | detour_ego | 702 | 1.00 | 0.00 | 1.0 | 1–1 | 120 | 94.4 | 32.8 | 9.4 | 21.6 | 27.4 | 36.0 |
| 4.6 | speed_ego | 1200 | 1.00 | 0.00 | 1.0 | 1–1 | 150 | 23.4 | 15.7 | 13.5 | 11.9 | 37.2 | 30.4 |
| 5.7.1 | density | 543 | 32.00 | 9.42 | 31.0 | 21–44 | 60 | 46.4 | 62.9 | 10.3 | 10.0 | 67.6 | 128.1 |
| 5.7.2 | density | 582 | 32.00 | 9.42 | 31.0 | 21–44 | 60 | 37.7 | 59.4 | 11.9 | 8.6 | 110.9 | 136.3 |
| 5.15.1 | density | 915 | 32.89 | 18.02 | 31.0 | 2–81 | 60 | 38.4 | 24.1 | 19.3 | 15.2 | 37.6 | 48.0 |
| 5.19 | density_ped | 1050 | 34.63 | 9.49 | 34.0 | 22–48 | 60 | 21.8 | 51.8 | 20.0 | 30.2 | 60.3 | 75.9 |
| 5.21 | speed_ego | 1200 | 1.00 | 0.00 | 1.0 | 1–1 | 150 | 34.0 | 32.7 | 11.7 | 22.8 | 38.4 | 46.7 |
| 5.31 | speed_ego | 1200 | 1.00 | 0.00 | 1.0 | 1–1 | 150 | 26.1 | 21.5 | 7.8 | 13.0 | 28.1 | 31.8 |

**Configured horizon (all):** 94 s.
**Mean realized (weighted):** IDM 30.3 s; IDM rule 31.7 s; Plant2 10.9 s; Plant rule 18.4 s; CARL 34.2 s; CARL rule 41.1 s.

| Traffic mode | Mean agents | N scenarios |
|---|---:|---:|
| `aux_convoy` | 4.13 | 4759 |
| `density` | 32.23 | 3504 |
| `speed_ego` | 1.00 | 4800 |
| `detour_ego` | 1.00 | 1851 |
| `density_ped` | 34.63 | 1050 |

**Agent definition.** `aux_convoy`: 1 ego + `aux_convoy_size × aux_lanes_occupied`. `density`: 1 ego + nuPlan vehicles/frame (fallback: `traffic_density × 80`). `density_ped`: density agents + `pedestrian_count`. `speed_ego` / `detour_ego`: ego-centric scenarios (1 agent).

**Duration.** Configured horizon = `horizon_steps × 0.1 s` (speed: 1500/150 s; detour: 1200/120 s; others: 600/60 s). Realized = weighted `avg_steps`/`final_step` × 0.1 s for `idm` (`idm_*`), `idm_rule` (`modified_idm_*` / `comprehensive_rule_expert_*`), `plant2` (`plant2_default`), `plant_rule` (`plant2_rule_default`), `carl` (`carl_default`), `carl_rule` (`carl_rule_default`). Compact planner-only table: `duration_by_planner.md`.
