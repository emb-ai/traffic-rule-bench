# Detour 4.2.x — summary

Goal: obstacle-avoidance compliance on the 122-scene detour test catalog,
learned from perception (no waypoint post-processing).
Full chronological log with all raw numbers: `DETOUR_HYPOTHESES_FULL_LOG.md`.

## Result so far

| stage | compliance | success | past obstacle |
|---|---|---|---|
| baseline (as found) | 0.014 | 0.000 | 0% |
| + target fix | 0.055 | 0.123 | 24% |
| + oversample & dropout tuning | 0.131 | 0.101 | 27% |
| **+ early sign visibility (90 m)** | **0.626** | **0.158** | **31%** |

Compliance rose 45x while driving *improved* — this is not the degenerate
mode described below.

## Two real bugs, both in data generation

**1. The input contained the answer.** The dump wrote one array into both
fields (`plant2_frames.py`: `"route": route_pts, "route_original": route_pts`,
both from `get_route()` = the navigation lane centreline). So the model's
route INPUT was elementwise identical to its path TARGET — 1475/1475 frames,
deviation exactly 0, in detour *and* stop dumps alike.

Copying the input was therefore *exactly* optimal, and that is what the model
learned: shifting the input route by +3.00 m shifted the prediction by
+2.73 m (91%); with the route zeroed the prediction collapsed to 0.21 m. In
the simulator the route is the real SUMO plan — straight through the cone —
so the model copied it and drove in. This also explains why H1-H10 (loss
reweighting, oversampling, LR) all returned bit-identical metrics: they were
tuning a model whose objective was already solved by copying.

Upstream PlanT 2.0 keeps these distinct (`route_original` = untouched plan,
`route` = the route the expert actually followed, plus a `changed_route`
flag). The copying habit is inherited from pretraining too: the pretrain
checkpoint already follows the route 76% and produces 0.8 m without it.

Fix: `scripts/plant2_ft_pipeline/data/fix_route_target.py` rebuilds the target
offline from the recorded `pos_global`/`theta` — no re-simulation. Heavy files
are symlinked, so shared source dumps are untouched.

**2. The sign was invisible until too late.** `collect_boxes` capped sign
boxes at 30 m, but compliance requires the lane change to be *complete* at
zone entry, 30 m before the obstacle, when the sign is still ~26.5 m ahead.
On the 171/366 scenes without cones the sign is the only cue, making them
unsolvable from perception — the privileged expert scores 18/18 there because
it reads the scene config. Ceiling under the cap was ~0.53.

Fix: `PLANT2_SIGN_RANGE_M` (default 30, unchanged) replaces the hardcoded
radius; re-dumped 1461 routes at 90 m (sign now visible at a median 59 m).
Plain fine-tune alone then reaches 0.601, confirming this was the binding
constraint rather than anything in the training objective.

## The metric is gameable — always read it with driving quality

`DetourSign._is_violating` only asks the vehicle to *be in the adjacent lane
while inside the zone*. A car that drifts sideways immediately and then
stalls satisfies it perfectly:

| route_dropout | compliance | distance | past obstacle |
|---|---|---|---|
| 0.0 | 0.055 | 73 m | 24% |
| 0.3 | 0.145-0.189 | 56 m | 12-15% |
| 0.5 | **0.628** | 32 m | **2%** |

So ">0.75 compliance" is reachable trivially and meaninglessly. Every number
here is reported with success rate and past-obstacle rate for that reason.
Same trap caught earlier: zeroing the route at inference gave 0.836
compliance while dying a median 41 m *before* the obstacle.

## Things checked and ruled out

- Perception range for cones — training sees `range * range_factor_front`
  = 100 m; cones are dumped to ~100 m. Measured, then dropped.
- Longer training — val loss plateaus by epoch 6 and is flat to 20.
- `best`-by-`val/loss_all` is a poor proxy: it picked epoch 0 for H22
  (0.191) over the final checkpoint (0.601). Both are evaluated now.
- `val/loss_path` is not comparable across splits: the old split divided by
  *route*, so the same scene appeared in train and val.

## Infrastructure bugs fixed along the way

- BEV fed at 128 px / 64 m (2 px/m) against training's 128 px / 32 m — the
  `[64:-64]` crop only triggers on a 256 px render. Stop-sign re-verified:
  0.452 -> 0.500.
- TOCTOU race in the shared diskcache (`key in cache` then `cache[key]`)
  killed parallel runs; now atomic `Cache.get`.
- `eval_core.py`: `ProcessPoolExecutor` over a closure (never ran), and a
  missing `--python` that resolved to an unrelated system interpreter.
- `eval_pipeline.py` lacked `pdd-bench` on `sys.path`.

## The compliance metric had two escape hatches (both fixed)

Instrumenting `DetourSign._is_violating` over the runs it scored compliant
showed the violation check was reached on only 24 of 802 steps; 778 exited at
the zone test, none at the drivable-area test. The decisive line, on the last
step of a run scored fully compliant:

    long=39.6 zone=[40,79] on_lane=False lane_idx='lane_1169101417_3'

The sign is 4.2.1 (pass on the right), the allowed lane is `..._0`, and the ego
had steered the other way -- `_1` -> `_2` -> `_3` -- then left the road 0.4 m
short of `zone_start`. The episode ended there with zero violation steps, i.e.
a perfect compliance score for driving off the road away from the obstacle.
`in_zone_total_steps` was 139 and hides this: `_ego_in_sign_zone` counts the
approach lookahead as well.

Two distinct holes:

1. **Ending before the zone.** `_is_violating` only evaluates inside
   [zone_start, zone_end], so a run that dies on the approach records no
   violation. This is the dominant one.
2. **Leaving the drivable surface inside the zone.** `is_in_drivable_area()`
   was tested *before* zone membership, so stepping off the sign's road cleared
   the violation for the rest of the episode. MetaDrive also keeps reporting the
   nearest lane in `vehicle.lane`, so a car veering off on the correct side
   could be credited with the lane change.

Fixes in `pdd-bench/traffic_signs/detour_sign.py`: the zone test now runs first
and purely geometrically; the lane-change credit additionally requires
`vehicle.on_lane`; and the sign exposes `reached_zone`, surfaced per episode as
`reached_zone_by_class` by `run_benchmark.py`. Corrected compliance =
*reached the zone* AND *no violation steps* -- computed by
`summarize_detour.py`, which refuses to report the new number for eval
directories written before the fix.

Effect on the ranking (old headline vs the honest rate over all 366 runs):

| model | headline | out_of_road | honest |
|---|---|---|---|
| detour H26 | 0.626 | 0.51 | **0.189** |
| detour H27 | 0.713 | 0.73 | 0.142 |
| detour H32 | 0.814 | 0.75 | 0.107 |
| joint j3 | 0.279 | 0.18 | 0.096 |

H27 -> H32 read as +0.10 compliance; it was actually a regression. Every
number above the H22 row in the table at the top of this file is inflated the
same way and is being re-measured under the fixed benchmark.


## The lateral frame was mirrored between training and inference

MetaDrive's `convert_to_local_coordinates` returns (forward, LEFT). The dump
negates it once for object boxes (`plant2_frames._ego_xy`) and twice for the
route (`get_route`), so on disk:

| field | frame |
|---|---|
| object boxes (`x_objs`, and the forecast target) | y = RIGHT |
| route input, path target, waypoint target | y = LEFT |

Confirmed geometrically rather than by reading: a cone is static in the world,
so un-projecting its dumped ego coordinates through the recorded
`pos_global`/`theta` must land on the same world point at every frame. Spread
of that reconstructed point: boxes as y=right 1.12 m vs as y=left 14.95 m;
route as y=left 0.28 m vs as y=right 0.54 m.

`plant2_adapter` fed the route as y=right and both controllers read the model's
output as y=right (`steer = -steer`). So the whole lateral geometry was
mirrored with respect to training -- except the objects, which were y=right on
both sides. Route following survived that (mirroring the input mirrors the
output, and the controller mirrors it back, because route-copying is close to
linear), but every deviation the model inferred from the OBJECTS came out
steering to the wrong side. That is exactly what the benchmark trace showed:
under 4.2.1, allowed lane `..._0`, the ego went `_1 -> _2 -> _3`.

Matched-scene A/B (honest compliance, fixed benchmark):

| checkpoint | old | matched to the dump |
|---|---|---|
| h26_last | 0.000 | **0.814** |
| j3_last | 0.000 | **0.879** |
| j4_last | 0.000 | **0.832** |
| h26_best (epoch 0 ~ pretrain) | 0.286 | 0.316 |
| pretrain, epoch 029 | 0.000 | - |

Driving improves with it, so this is not a metric trade: crashes ~0.95 -> ~0.42
and distance driven roughly doubles. `h26_best` being indifferent is the tell --
it is epoch 0, still essentially the CARLA pretrain checkpoint, so it never
adopted the dump's convention. `PLANT2_YLEFT` now defaults to on.

Second consequence, in training: `aug_sample` applied ONE signed shift and one
rotation to both groups, which is correct upstream where every field shares a
frame. Here it moved objects and route in opposite physical directions -- for a
1 m / 5 deg jitter, an object and a route point at the same physical place ended
up 3.77 m apart. Every `--augment` run so far trained on that. Fixed:
the y=right group takes +translation and R, the y=left group -translation and R.T.

## Joint model (2.5 + 4.2.x, 1544 train routes, all three data fixes)

Stop improves over the stop specialist; detour is roughly level on the honest
rate while driving far better (out_of_road 0.75 -> 0.18).

| model | stop compliance | stop success | detour honest | detour oor |
|---|---|---|---|---|
| stop specialist (fixedenv lr3e4) | 0.571 | 0.524 | - | - |
| joint j1 lr1e-4 | 0.429 | 0.571 | 0.038 | 0.12 |
| joint j2 lr1e-4 + ovs | 0.619 | 0.714 | 0.030 | 0.09 |
| joint j3 lr3e-4 | 0.571 | **0.786** | 0.096 | 0.18 |
| joint j4 lr3e-4 + ovs | **0.643** | 0.619 | 0.074 | 0.21 |

None of j1-j4 used `route_dropout_p`, the one ingredient behind H26's best
honest rate; j5-j7 add it (and a lateral-noise variant) on the joint split.
