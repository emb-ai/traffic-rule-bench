# Stop sign (2.5) scores 0.00: what was actually wrong

Branch `dev/stop-2-5`. Diagnosis and fix for the PlanT-2 campaign's `SR&Dest = 0.00`
on 2.5 in all five configurations. **Nothing was trained.** Total simulator work:
61 episodes at ≤ 4 workers.

**Headline:** the dominant cause was not the model. Our 2.5 compliance check
returned "violated" for *every* vehicle that crossed the stop line, including
one that had come to a complete halt 0.001 m/s in front of it. A rule expert
that stops correctly in 20/20 episodes and reaches the destination in 20/20
scored **compliance 0.05**. After the fix it scores **1.00**, while a
non-rule-following plain `idm` baseline stays at **0.00** — so the check now
discriminates rather than being either always-fail or vacuous.

---

## 1. Step 1 — the gate

`traffic_bench/signs/junction/yield_sign.py` `StopSign` had no way to say *which*
of its two obligations failed: `_is_violating` OR-ed "did not give way to the
main road" with "did not stop at the line" into one counter labelled `StopSign`.
So a 0.00 was uninterpretable, and had been for the whole campaign.

Commit `f98b8f3` adds a `stop_audit` accessor exporting
`{crossed, min_speed, yield_steps, stopline_steps, stop_line_long, zone, max_long}`
and surfaces it per-episode as `sign_audits` in the eval record and the sidecar
(`traffic_bench/eval/run/episode.py`). That commit changes **no verdict** — it is
measurement only. Then `expert_idm_rule` was run on the 20 stop validation rows.

**Pre-registered prediction:** `comp ≈ 0.00`, `stopline_steps > 0`, `yield_steps = 0`.

| `expert_idm_rule` on stop val, n=20 | before the fix |
|---|---|
| compliance (`violations == 0`) | **0.05** |
| reached_dest & not crashed | 1.00 |
| SR&Dest | 0.05 |
| crashed or out_of_road | 0.00 |
| `yield_steps > 0` | 0 / 20 (sum 0) |
| `stopline_steps > 0` | 19 / 20 (sum 63) |
| `crossed == True` | 0 / 20 |
| `min_speed` before the line | 0.001 – 0.003 m/s, **< 0.5 m/s in 20/20** |
| stopped, yet stop-line-flagged | **19 / 20** |
| `max_long > zone_end` | 20 / 20 |

Prediction confirmed. The expert halts to a dead stop in every one of the 20
episodes, never trips the yield clause, and is flagged anyway in 19 of them.
C1 is real and dominant.

The single unflagged episode is `junc_134684395_td50_sv0_v1` — the same episode
that is still the only one with `crossed == False` *after* the fix. `max_long` is
an unconditional projection onto the sign lane, taken whatever lane the ego is
actually on, and it reads 199.5 against a verdict point of 191.2; but the
stop-line clause itself is gated on `_is_on_sign_road`, and here the ego had
already left the approach lane by the time its projection passed that point, so
the clause was never sampled at all. The pre-fix 0.05 was that accident, not a
pass — which is the sharpest statement of the problem: the metric's one
non-zero came from an episode it failed to evaluate.

### The arithmetic

All 20 stop validation rows (and all 800 train rows, and every other
junction-priority family) carry `sign_distance_before_end: 0.0`, so
`episode.py`'s `row.get("sign_distance_before_end", 20.0)` fallback never fires —
the key exists. Then:

- `sign_placement_long(lane, 0.0)` returns `lane.length`
  (`traffic_bench/eval/engine/map/junction_sign_placement.py:24-28`);
- `traffic_bench/signs/base.py:75-82` clamps the plate to `L − 0.1` and says so.
  From `$SM/tmp/stop25/logs/expert_idm_rule_s0.log`, verbatim:

  ```
  [TrafficSign] StopSign: requested s=132.0m is off lane lane_1160545752_0
  (length 132.0m); clamped to 131.9m -- check the offset convention
  ```

- `StopSign.stop_line_position = placement_long = L − 0.1`;
- inherited `zone = [L − 30, L]` (`EGO_ZONE_BEFORE = 30.0`);
- the verdict is only taken once `veh_long ≥ stop_line + 0.3` = `L + 0.2`
  (`STOP_LINE_PAST_MARGIN`);
- `L + 0.2 > zone_end = L`, so `_track_stop_before_line` hit
  `self._vehicle_states_stop.pop(vid, None)` and returned `False`;
- `_is_stop_line_violating`'s `return not stopped` was therefore **True
  unconditionally**.

The latch was erased 0.2 m before the only line that reads it. The failure
condition is exactly `sign_distance_before_end < STOP_LINE_PAST_MARGIN`, which is
why the colleague's `rebbutle-wip` — same code, but scenes with the plate set
back from the junction — never hit it and reports "all 11 misses are yield
violations; the car does stop" (`BEST_EXPERIMENT.md`).

The geometry itself is self-consistent and must **not** be "fixed" by moving the
plate: the ego spawns at `L − 15` (`spawn_distance_before_end: 15.0`), so a plate
at `L − 20` would spawn the ego past its own stop line.

### Corroboration over the 200 existing episodes

Re-derived from `$FT/runs_prio/stop/val/eval_out/ft_*/episodes_plant2.jsonl`
(10 runs × 20):

- 41 episodes reached the destination crash-free; **all 41 carry 2–6 violations**;
- only 3 episodes have zero violations, and **none of them reached the
  destination** — two were frozen for the full 600-step horizon after 2.5 m and
  5.7 m, the third crashed and went out of road at 15.4 m.

Fixing every crash would still have left SR&Dest at 0.00.

### A smaller thing found on the way

`episode.py` counts a step twice: once through an ungated
`for sign in sign_mgr.signs: if sign._is_violating(vehicle)` loop, and again
through the edge-triggered `sign_mgr.check_all_violations` (which gates on
`is_in_drivable_area` and de-duplicates by re-arming). So `violations` is a sum
of a per-step count and an event count, and its *magnitude* is not "the number
of infractions" — a single logical stop-line failure reads as 2 or 3. Only
`violations == 0` is meaningful, which is what compliance uses, so no metric is
wrong. Left as is; noted so nobody reads the number as a count.

---

## 2. Step 2 — the fix

Commit `6c490f7`, `traffic_bench/signs/junction/yield_sign.py`
`_track_stop_before_line`. The unconditional `pop` becomes directional:

- `veh_long < zone_start` → a genuinely fresh approach, drop the latch;
- `veh_long > zone_end` → report the latch unchanged; do not update, do not clear.

The latch now means what its name says: *this vehicle came to a stop before the
line during this approach*. The audit's `crossed` flag was also allowed to fire
past `zone_end`, where the crossing actually happens (it could never be `True`
before).

### Guard rails

| `stop` val, n=20 | before | after |
|---|---|---|
| `expert_idm_rule` compliance | 0.05 | **1.00** |
| `expert_idm_rule` SR&Dest | 0.05 | **1.00** |
| `expert_idm_rule` reached_dest & !crashed | 1.00 | 1.00 |
| `expert_idm_rule` `stopline_steps > 0` | 19/20 | **0/20** |
| `expert_idm_rule` `crossed == True` | 0/20 | 19/20 |
| plain `idm` compliance (must stay ≈0) | — | **0.00** |
| plain `idm` SR&Dest | — | **0.00** |
| plain `idm` reached_dest & !crashed | — | 0.90 |

**The check did not become vacuous.** Plain `idm` — no rule mixin — scores 0.00
compliance over the same 20 rows *while reaching the destination in 18 of them*,
so the zero is rule violations, not driving failures:

| plain `idm` after the fix, n=20 | |
|---|---|
| `yield_steps > 0` | 18 / 20 (sum 54) |
| `stopline_steps > 0` | 17 / 20 (sum 75) |
| `crossed == True` | 20 / 20 |
| `min_speed` before the line | min 0.002, **median 1.909**, max 3.904 m/s; < 0.5 m/s in only **3 / 20** |
| stopped, yet stop-line-flagged | **0 / 20** |

The three `idm` episodes that *did* come to a halt were correctly cleared on the
stop-line clause and flagged on the yield clause instead — e.g. one with
`min_speed 0.002, yield_steps 3, stopline_steps 0`. So after the fix the two
clauses separate cleanly, and the spread between the two policies is the full
0.00 → 1.00.

This run is also the first positive control the **yield** clause of 2.5 has ever
had: it fires 18/20 for `idm` and 0/20 for `idm_rule`. Before this it was 0/20 in
the only measurement we had, which is equally consistent with "never violated"
and "dead code".

**`main_road` / `secondary_road` / `yield` are unaffected — by construction, not
by measurement.** `_track_stop_before_line`, `_is_stop_line_violating` and
`stop_line_position`-based verdicts exist only inside `StopSign`
(`yield_sign.py:1091-1240`). `YieldSign._is_violating` (`:1009`) uses only
`_is_vehicle_in_zone` and `_check_main_road_traffic` and never touches the latch;
`RightHandYieldSign` subclasses `YieldSign`, not `StopSign`; nothing subclasses
`StopSign`. The changed lines are unreachable from those families. They were
**not** re-run — that would have cost 60 episodes to confirm a code path that
cannot execute.

Reference points from the same script and the same placement, for the ceiling
C3 asked for and never had:

| expert_idm_rule | n | compliance | SR&Dest |
|---|---|---|---|
| `main_road` | 20 | 1.00 | 0.75 |
| `secondary_road` | 17 | 1.00 | 0.94 |
| `stop` (after the fix) | 20 | 1.00 | 1.00 |
| `yield` | — | never run | never run |

### What could NOT be re-scored

The 200 stored episodes predate `sign_audits`, so they carry no record of whether
the ego stopped. **They cannot be re-scored offline**: whether each of the 41
destination-reaching episodes becomes compliant under the fixed checker is
unknowable from the stored fields. Getting the real corrected campaign numbers
needs a re-run of the 10 checkpoint evals (200 episodes), which was outside the
agreed budget and was not done.

---

## 3. Step 3 — can the ego see what it must yield to?

**The suspected bug does not exist, and the plan's file reference was
mis-attributed.** The `if not vehicles and hasattr(engine, "get_objects")`
fallback at `metadrive_obs_to_plant2.py:171-181` is inside
`_maybe_move_yield_sign_onto_npc` (defined at `:127`), a cosmetic helper that
relocates a yield-sign box onto the nearest NPC. It is **not** the object-box
path.

`x_objs` at eval comes from `collect_objects_ego_frame_from_plant2_boxes`
(`:338`), which calls `plant2_frames.collect_boxes` — the identical function the
dumper uses, with identical arguments (`max_distance=50.0`,
`range_factor_front=2.0` on both sides). There is no fallback and no union to
fix.

Also worth correcting: `traffic_density` is **not** 0.0 in every stop row. Of the
20 validation rows only 2 are 0.0; the rest run 0.034 – 0.575, and
`aux_convoy_size` is 1 (×7), 2 (×8) or 3 (×5), spawning up to 9 vehicles.

Measured over the 191 stop expert dump routes (57 547 frames), through the same
`collect_boxes`:

| other vehicles in the boxes | |
|---|---|
| routes with ≥1 frame carrying another car | 170 / 191 |
| frames carrying ≥1 other car | 39 034 / 57 547 (67.8 %) |
| cars per frame | mean 2.26, median 2, max 14 |
| ego→car distance | min 2.8, median 41.3, p90 76.9, max 100.0 m |

And **at eval**, one plant2 episode on `junc_10793658469_rl120_td25_sv0_v1`
(convoy 3, `traffic_density` 0.034) with `PLANT2_DEBUG_BOXES=1`, checkpoint
`only_stop/epoch=009`:

| eval-side `x_objs`, 42 frames | |
|---|---|
| objects per frame | mean 13.7 (min 11, max 14) |
| frames with ≥1 car | **42 / 42**, mean 10.7 cars per frame |
| frames with the 2.5 plate | **42 / 42** |
| nearest car | min 4.6, median 15.1, max 21.9 m |

A representative frame:

```
[plant2_boxes] n=14 mix={'car': 11, '2.1': 2, '2.5': 1}
  nearest=[('2.1', 10.7), ('2.5', 13.1), ('car', 17.6), ('car', 24.4), ('car', 26.0), ('2.1', 29.5)]
```

**Step 3 passes; no adapter change was made.** The ego can see what it must give
way to, on both sides.

That episode is worth reading for a second reason. It ended at step 42, 16.6 m in,
**crashed**, with the audit reading
`{crossed: True, min_speed: 2.078, yield_steps: 3, stopline_steps: 6}` — the model
never dropped below 2.08 m/s, tripped both clauses, and drove into the convoy it
should have yielded to. With a correct checker and full visibility, this
checkpoint still fails this scene on the merits. That is the part the fix does
not touch.

A small permanent diagnostic was added at the one place both sides build their
boxes: `PLANT2_DEBUG_BOXES=1` now prints `n`, the class mix and the six nearest
objects from `collect_objects_ego_frame_from_plant2_boxes`.

---

## 4. Step 4 — audit of the existing stop dumps (read-only)

191 routes / 57 547 frames across `$SM/ft_rl3/dump_prio/stop` and `dump_prio_r2/stop`.

### 4.1 Route-target leak — not present in training, but one env var away

The dumper writes `route` and `route_original` from the same array
(`finetune/plant2_frames.py:525-526`), so **100 % of dumped frames have
`route == route_original`**. What saves us is `PATH_TARGET=future` in
`train_rl.sh:12`, which replaces `sample["route"]` at load time with the ego's
realised future. Measured with `dataset.py`'s own `_future_path` /
`interpolate_route` at the campaign's `PATH_HORIZON_FRAMES=40`:

| future-path target vs `route_original` | |
|---|---|
| bit-identical | **0 / 57 165 (0.00 %)** |
| per-frame mean deviation | mean 5.95 m, median 1.27 m, p90 15.22 m |
| per-frame max deviation | mean 8.60 m, median 2.01 m, p90 22.84 m |
| frames within 0.10 m mean | 0 / 57 165 |
| frames within 1.00 m mean | 2 080 / 57 165 (3.6 %) |
| stopped frames (< 0.5 m/s), n=21 589 | mean deviation 2.40 m, median 1.60 m |

So the colleague's single biggest data bug (H1–H10, ten bit-identical failed
hypotheses) does not apply to what we actually trained.

It is, however, only one unset variable away. With `PATH_TARGET` at its default
`"route"`, `sample["route"]` is `interpolate_route(route[:20])` against a raw
`route_original[:20]`:

| would-be leak under `PATH_TARGET=route` | |
|---|---|
| mean deviation | 3.58 m, median **1.02 m**, p99 65.02 m |
| frames within 0.10 m | 321 / 57 547 (0.6 %) |
| frames within 0.01 m | 21 / 57 547 (0.0 %) |

Not the bit-identical copy the colleague had — `interpolate_route` prepends a
zero point and re-samples by arc length, which shifts the whole route by roughly
one index — but a median 1.0 m separation between the path target and a token the
model already holds is still a very cheap thing for the head to exploit. Caveat:
this measures *geometric separation*, not "is copying optimal"; the latter was
not tested.

### 4.2 Sign visibility — fine

| 2.5 plate in `x_objs` | |
|---|---|
| routes with ≥1 frame carrying the plate | **191 / 191** |
| frames carrying the plate | 57 538 / 57 547 (**100.0 %**) |
| ego→plate distance | min 2.9, median 13.3, p90 70.6, max 119.8 m |

Distance histogram: `[0,10) 13.2 %`, `[10,20) 52.2 %`, `[20,30) 5.6 %`,
`[30,40) 4.8 %`, `[40,50) 4.9 %`, `[50,60) 4.6 %`, `[60,80) 8.6 %`,
`[80,100) 4.0 %`, `[100,120) 2.1 %`, `≥120` none — our 120 m
`PLANT2_SIGN_RADIUS_M` binds exactly where it should. The 52 % mass at 10–20 m is
the expert dwelling at the line. The plate is present; this is not a visibility
problem, and the colleague's 30 m cap is not something we need.

### 4.3 Speed-label resolution — the strongest remaining candidate

Our bins are `[0, 4, 8, 10, 13.89, 16, 17.78, 20]` m/s
(`third_party/plant2/PlanT/plant_variables.py:45`). The campaign trained 2.5 with
`TS_WINDOW_FRAMES=default:1` (confirmed in the train log: *"target_speed = min ego
speed over 1 frames (0.1 s)"*, and 2.5 is absent from the per-code list), so the
label is essentially the instantaneous ego speed, snapped to 0.0 below 0.5 m/s.

| stop-frame speed labels | |
|---|---|
| raw ego speed < 0.5 m/s | 37.5 % of frames |
| label exactly 0.0 | 38.1 % |
| label in [0.5, 1) | 2.2 % |
| label in [1, 2) | 5.2 % |
| label in [2, 4) | 14.3 % |
| label in [4, 8) | 28.3 % |
| label ≥ 8 | 12.1 % |

**21.6 % of all stop frames (12 438) carry a label in the open interval (0, 4)
m/s — the entire first gap of our grid.** The soft two-hot target puts on average
only **0.384** of their mass on bin 0 (0.0 m/s); **0.616 leaks onto the 4 m/s
bin**. In other words, for about one frame in eight of the whole stop corpus the
model is told "mostly 4 m/s" while the expert was crawling towards a halt. The
colleague's grid `[0, 0.025, 0.055, 1, 1.5, 2, 4, 8, 10, 20]` spreads those same
frames across four distinct lower bins.

This quantifies the case for finer bins. It does **not** authorise the change: it
is checkpoint-breaking (speed head 8 → 10) and needs a full retrain.

### 4.4 `TS_WINDOW_FRAMES`

2.5 takes `default:1`. Noted, as instructed — crosswalk (0.67) and yield (0.95)
take the same default, so this is a candidate, not a cause.

---

## 5. Step 5 — augmentation made real (code + unit test only)

Commits `b8d0e30` (superproject + test), plant2 `8121e99`, metadrive `9f6b8b94`,
gitlinks in `4b42185`. **Nothing changes until data is re-dumped** — the new
magnitudes default to 0, which reproduces the previous bytes exactly.

1. `metadrive_obs_to_plant2.py` `render_bev_plant2` gains `lateral_offset_m` /
   `heading_offset_rad` and draws the map from that virtual viewpoint. The
   vehicle is not moved.
2. `finetune/plant2_frames.py` `Plant2FrameCollector.on_step` (the plan says
   `record`; the method is `on_step`) draws a per-frame jitter from
   `PLANT2_AUG_TRANSLATION_M` / `PLANT2_AUG_ROTATION_DEG`, renders the second BEV
   from the jittered pose and writes the real magnitudes. It previously hardcoded
   `0.0 / 0.0` and copied the plain BEV into `bev_no_car_semantics_augmented/`, so
   `--augment` was a silent no-op.
3. `PlanT/dataset.py` `aug_sample` now applies to each group the transform its own
   frame requires. Object boxes are y=RIGHT (`collect_boxes` negates what
   `convert_to_local_coordinates` returns, `plant2_frames.py:234`); route /
   route_original / waypoints are y=LEFT (`plant2_frames.py:403`,
   `build_ego_matrix` at `:39`). One signed shift and one `R.T` for all of them
   moved the two groups in **opposite physical directions**. The y=right group now
   takes `+translation` and `R`, the y=left group `-translation` and `R.T`.

### The yaw sign — we deliberately differ from the reference

The reference implementation (`emb-ai/plant2` at `52159a93`, the submodule of
`rebbutle-wip`) flips the yaw to `input[:, 3] += rad2deg(rot)`. **We keep `-=`.**
The dump stores `wrap_to_pi(obj.heading_theta - ego_heading)`, a y=LEFT CCW angle
that is *not* mirrored alongside the position (their dumper does the same,
`col_plant2_frames.py:299`), so an ego rotation of `+rot` takes every relative
heading to `yaw - rot` in either frame.

`finetune/test_aug_sample_frames.py` decides this without appeal to authority: it
builds a synthetic world, asks the dumper's own formulas what it *would* have
written standing at the jittered pose, and requires `aug_sample` to produce
exactly that. Results:

| variant | verdict |
|---|---|
| pre-fix code | positions wrong; an object and the route point on top of it end up **4.16 m apart** under 1 m / 5° |
| reference (`+=` yaw) | positions correct, **yaw off by 2·rot = 10°** |
| this branch (`-=` yaw) | **PASS** on all five jitter cases |

(The colleague measured 3.77 m for the same class of error on their geometry.)

### Note, not acted on

Any diskcache built before an `aug_sample` change is **stale and silently
reused** — augmented samples live under their own `_aug` key. Whoever re-dumps
must clear it.

---

## 5b. What was and was not taken from `rebbutle-wip`

Ranked by what it was worth to us.

| piece | taken? | why |
|---|---|---|
| the split `aug_sample` (y=right `+t`/`R`, y=left `−t`/`R.T`) | **yes**, re-derived | the one genuinely portable fix; their prose in `DETOUR_HYPOTHESES.md` matches the derivation |
| BEV pose jitter in `render_bev_plant2` | **yes**, ported verbatim | 13 lines, exactly what the second BEV needs |
| `PLANT2_AUG_TRANSLATION_M` / `_ROTATION_DEG` in the dumper (1.0 m / 5.0°) | **yes**, knobs only | defaults left at 0, so nothing changes until a re-dump |
| their `input[:, 3] += rad2deg(rot)` yaw sign | **no** | the unit test shows it is wrong by 2·rot; see above |
| the `stop_audit` accessor | **yes** | it is what made the 0.00 interpretable |
| `fix_route_target.py` | no | our in-dataset `PATH_TARGET` / `PATH_HORIZON_FRAMES` supersedes it, and theirs must be re-run after every dump |
| `sign_range_m` 30 / `PLANT2_SIGN_RANGE_M` 90 | no | our 120 m is deliberate and better measured (2 % of frames at the narrow radius against 33 % at 120 m, `plant2_frames.py:108-112`), and §4.2 shows the plate is in 100 % of frames |
| eval `max_distance=75, range_factor_front=16` | no | ours (50 / 2) already matches training on both sides |
| their `dataset.py` wholesale | no | ~1200 lines behind ours; drops `_log_sign_metrics`, `frame_weight`, the `wps_weight` key fix and the 2.3-variant merge |
| `PLANT2_YLEFT` | not done | same knob as our `PLANT2_AXIS_ALIGN`, same default, same two flip points; an alias would only stop their scripts being silently ignored |
| the 10-bin speed grid + `speed_class_weights` | **no — needs a decision** | §4.3 quantifies the case; it is checkpoint-breaking (speed head 8 → 10) and needs a full retrain |
| the checkpoint itself (446 MB) | no | on a different machine (`antonov`), and the split that produced it was on `/tmp` and is gone |

**One caveat on provenance.** The `aug_sample` and BEV-jitter fixes are *not* in
the `rebbutle-wip` refs fetched into `$SM/traffic-rule-bench-main` (metadrive
fetched 2026-08-13, plant2 2026-08-24) — those are byte-identical to our pre-fix
code. They are only in the submodule commits the GitHub branch points at
(metadrive `aa30ce85`, plant2 `52159a93`), reachable through
`gh api repos/emb-ai/traffic-rule-bench/contents/<sub>?ref=rebbutle-wip`. Anyone
re-checking this against the node's local refs will conclude, wrongly, that the
fix does not exist.

---

## 6. What is still unknown

- **The corrected campaign numbers are not known.** Only the expert was
  re-measured. What `only_stop` / `prio_all` actually score under the fixed
  checker requires re-running those 10 evals (200 episodes). By prior
  measurement a 10-epoch `only_stop` rerun is ~13 min on a free GPU — offered,
  not started.
- **How much of the remaining failure is driving rather than scoring.** The
  10 stored runs show crash-or-out-of-road 0.65–0.90. The fixed checker cannot
  help those; the expert's 0.00 crash rate over the same 20 rows says the scenes
  are drivable, so that gap is the model's. The single `only_stop` epoch-9
  episode run in Step 3 crashed at 16.6 m without ever going below 2.08 m/s,
  which is consistent with that reading — but it is n=1 and must not be treated
  as a rate.
- **`yield` (2.4) has still never had an expert baseline.** `expert_val_prio.sh`
  died after `secondary_road` and this work only filled in `stop`.
- Whether the speed-bin change would actually help — the 21.6 % / 0.616 figures
  are a diagnosis of the label, not evidence about the model.
- Whether the `PATH_TARGET=route` separation (median 1.02 m) would be exploited.
  Geometric separation was measured; "is copying optimal" was not tested.
- The augmentation fix is verified against a synthetic oracle only. No dump, no
  training run, no closed-loop number.

## Two things the owner of `check/priority_signs` should decide

1. **This changes every 2.5 number ever reported.** Nothing has been pushed and
   no PR was opened.
2. **`sign_distance_before_end: 0.0` should probably not stay 0.0.** With the
   plate clamped to `L − 0.1`, the stop-line clause is only ever sampled in a
   ~3-step sliver where `vehicle.lane` still reports the approach lane but the
   longitudinal projection has passed its end. One of the 20 validation
   episodes slips through that sliver entirely — the same one, before and after
   the fix — so the clause is not reliably evaluated. Setting it to 5.0 would put
   the line at `L − 5`, comfortably inside the `[L − 30, L]` zone, still 10 m
   ahead of the `L − 15` spawn, and would match `YieldSign.YIELD_STOP_BEFORE_END`.
   That is a benchmark-semantics change and was **not** made here.

## Reproduce

Mac worktree `/Users/victoria_s/sdc_new_signs/trb-stop` (branch `dev/stop-2-5`),
node worktree `$SM/trb_stop` (`SM=/home/jovyan/shares/SR006.nfs2/smirnova`).
Scripts in `$SM/tmp/stop25/`, logs in `$SM/tmp/stop25/logs/`.

```bash
# unit test (no simulator, no GPU)
python finetune/test_aug_sample_frames.py          # or: pytest -q finetune/test_aug_sample_frames.py

# on the node — one policy over the 20 stop val rows, 4 shards, from $SM/trb_stop
ssh ssh-sr006-jupyter.ai.cloud.ru 'bash -lc "cd $SM/tmp/stop25 && bash run_stop_eval.sh idm_rule <tag>"'
ssh ssh-sr006-jupyter.ai.cloud.ru 'bash -lc "cd $SM/tmp/stop25 && bash run_stop_eval.sh idm     <tag>"'

# summarise one eval_out directory (compliance / SR&Dest / the stop audit)
$PY $SM/tmp/stop25/summarize_stop.py $SM/ft_rl3/runs_prio/stop/val/eval_out/<tag> "<label>"

# read-only dump audits
bash $SM/tmp/stop25/run_audit.sh audit_stop_dumps.py       # sign visibility + speed labels
bash $SM/tmp/stop25/run_audit.sh audit_route_target.py     # PATH_TARGET=future vs route_original
bash $SM/tmp/stop25/run_audit.sh audit_step3_and_leak.py   # PATH_TARGET=route leak + cars in boxes

# Step 3, one plant2 episode with the box diagnostic
GPU=2 bash $SM/tmp/stop25/step3_boxes.sh
```

Nothing in this work wrote to `$SM/traffic-rule-bench-main`, `$SM/trb_eval_rl3`,
`$SM/tmp/ft_v6` or `$Z`. No GPU other than 2 was used, and only for the single
Step 3 episode.

---

# Part II — follow-up investigations

Added after the checker fix landed. Everything below is measurement; no training,
no fix implemented.

## 7. The corrected campaign numbers (all ten checkpoints, n=20 each)

All ten previously-evaluated checkpoints re-scored under the fixed checker, run
from `$SM/trb_stop`, reusing the original `shard8_*` grouping, into new
`*_fixed` directories. The pre-fix records were not overwritten.

| run | n | comp before | comp **after** | SR&Dest before | SR&Dest **after** | crash\|oor | `stopline_steps>0` | `yield_steps>0` |
|---|---|---|---|---|---|---|---|---|
| ft_only_stop_e1 | 20 | 0.10 | **0.50** | 0.00 | **0.00** | 0.85 | 10/20 | 10/20 |
| ft_only_stop_e3 | 20 | 0.00 | **0.40** | 0.00 | **0.00** | 0.80 | 10/20 | 12/20 |
| ft_only_stop_e5 | 20 | 0.00 | **0.35** | 0.00 | **0.00** | 0.75 | 10/20 | 13/20 |
| ft_only_stop_e9 | 20 | 0.00 | **0.30** | 0.00 | **0.00** | 0.80 | 10/20 | 14/20 |
| ft_prio_all_e1 | 20 | 0.00 | **0.20** | 0.00 | **0.00** | 0.80 | 3/20 | 16/20 |
| ft_prio_all_e3 | 20 | 0.00 | **0.40** | 0.00 | **0.00** | 0.65 | 10/20 | 12/20 |
| ft_prio_all_e5 | 20 | 0.00 | **0.05** | 0.00 | **0.05** | 0.65 | 13/20 | 18/20 |
| ft_prio_all_e9 | 20 | 0.00 | **0.05** | 0.00 | **0.00** | 0.75 | 13/20 | 19/20 |
| ft_prio_all_e9_blind | 20 | 0.00 | **0.00** | 0.00 | **0.00** | 0.90 | 14/20 | 20/20 |
| ft_v6pad_e0 | 20 | 0.05 | **0.05** | 0.00 | **0.00** | 0.90 | 9/20 | 19/20 |

`crash|oor` is identical before and after by construction — see the reproduction
check below.

**The fix does not rescue the campaign.** Compliance moves off the floor
(0.00–0.10 → 0.00–0.50), but **SR&Dest stays 0.00 in nine of ten runs**; the one
exception is `ft_prio_all_e5` at 0.05, a single episode. The 0.00 was never only
a scoring artefact — it was a scoring artefact sitting on top of a driving
failure, and only the artefact is gone.

Two things the corrected numbers now show that the broken checker hid:

- **Compliance is not trustworthy as a headline at these crash rates.** The
  best compliance in the table, `ft_only_stop_e1` at 0.50, comes with
  `crash|oor` 0.85 — most of that 0.50 is episodes that crash before reaching
  the line and are "compliant" by never being judged. Compliance and crash rate
  move together in the wrong direction across the table.
- **The yield clause is now the dominant violation, not the stop-line clause.**
  `yield_steps>0` fires in 10–20 of 20 episodes everywhere, and in the later
  checkpoints (`prio_all` e5/e9, blind, v6pad) it fires in 18–20 of 20. The
  blind ablation is at 20/20 with compliance 0.00, which is the expected
  direction for a model denied the sign.

### Reproduction check

**`reached_dest`, `crashed`, `out_of_road` and `steps` are identical episode by
episode on all 200 episodes across all 10 runs.** A scoring-only fix must leave
these untouched, and it did — including across a GPU change (the originals ran
on GPUs 0/1/2/3, the re-runs all on GPU 3). This also confirms the two trees
differ only in the three files I changed: `traffic_bench` in `episode.py` and
`yield_sign.py`, `metadrive` in `metadrive_obs_to_plant2.py` (two hunks, both
no-ops with `PLANT2_DEBUG_BOXES` unset and jitter at 0), and `plant2` in
`dataset.py` (training-only, never called at eval).

Worth recording: `$SM/trb_eval_rl3`, where the originals ran, reaches its
submodules by **symlink into `$SM/traffic-rule-bench-main`** — the tree other
sessions edit in place. `$SM/trb_stop` has real checkouts. That was checked
rather than assumed; the shared `metadrive_obs_to_plant2.py` is dated
2026-08-31 and the originals ran 2026-09-09 16:15–17:31, so it did not move
under them either.

## 8. Decomposing the gap to the 1.00 expert ceiling

`ft_prio_all_e5_fixed`, the only run with a non-zero SR&Dest. Mutually exclusive
partition of all 20 episodes:

| bucket | n | share |
|---|---|---|
| success | 1 | **0.05** |
| driving failure (crash / off-road / no destination) | 13 | **0.65** |
| drove fine, but tripped **both** 2.5 clauses | 6 | **0.30** |
| drove fine, stop-line clause only | 0 | 0.00 |
| drove fine, yield clause only | 0 | 0.00 |
| drove fine, some other sign class | 0 | 0.00 |

Counterfactual ladder:

| | SR&Dest |
|---|---|
| as measured | 0.05 |
| + every driving failure fixed | **0.70** (+0.65) |
| + the stop-line clause alone always satisfied | 0.05 (+0.00) |
| + the yield clause alone always satisfied | 0.05 (+0.00) |
| + **both** clauses satisfied (driving untouched) | 0.35 (+0.30) |

**Answer: the gap is driving, not scoring.** 0.65 of the 0.95 shortfall is
episodes that never arrive. The remaining 0.30 is episodes that drive fine and
violate *both* clauses together — which is why fixing either clause alone is
worth exactly nothing; the same six episodes fail both.

`ft_prio_all_e3_fixed` gives the same shape (success 0.00, driving 0.65, both
clauses 0.35) and adds the sharpest single observation in this report: **10 of
its 13 driving failures had come to a complete halt before the line
(`min_speed` 0.001–0.036 m/s) and carry zero violations.** They obeyed 2.5
perfectly and then crashed. Within the 13 driving failures, all 13 crossed the
line and the median distance travelled is 19.6 m — the ego spawns 15 m before
the lane end, so they die roughly 4 m into the junction.

**So the next lever is neither the checker nor the rule head.** It is whatever
stops the car from following its route through the intersection.

## 9. Where the road is lost (all 200 episodes)

From `sign_audits.max_long` minus the approach-lane length:

| out-of-road episodes, n=104 | |
|---|---|
| died on the approach lane | **8 / 104 (0.08)**, all within 5 m of the mouth |
| lost the road 0–5 m into the junction | **94 / 104 (0.90)** |
| lost the road 5–15 m in | 2 / 104 (0.02) |
| median position past the mouth | **+2.3 m** (p25 +1.6, p75 +3.0, max +6.5) |
| crossed the stop line | 94 / 104 |
| **had halted before it** | **81 / 104** |
| median distance travelled | 19.9 m |

Outcomes over the 200: out_of_road 104, collision-without-oor 53, reached
destination 41, stuck 2. Per map: `junc_10793658469` dest 0.39 / oor 0.10 /
crash 0.61; `junc_134684395` dest 0.02 / **oor 0.94** / crash 0.96.

**The car stops correctly, enters the junction, and cannot follow the route
through it.** Plain `idm` has out-of-road **0.00** on the same scenes and the
rule expert 0.00 with dest 1.00, so the geometry is drivable and the failure is
the model's.

## 10. Is it overfitting to the 8 training maps? No.

Both e9 checkpoints evaluated on the maps they were *trained* on. The map set
came from the training split's own route directories
(`$SM/ft_rl3/splits/only_stop/train/data`), **not** by filtering the train
manifest against `split_meta.json`'s `val_maps` — that manifest spans 78 maps of
which only 8 ever reached the model, and filtering that way would have quietly
turned this into a second held-out measurement. All 20 sampled routes are
themselves training routes: same map *and* same route.

| n=20 each | dest | oor | crash | comp | SR&Dest | median dist |
|---|---|---|---|---|---|---|
| `only_stop_e9` — **trained** maps | 0.25 | 0.65 | 0.70 | 0.75 | 0.20 | 68.8 m |
| `only_stop_e9` — held-out maps | 0.20 | 0.50 | 0.80 | 0.30 | 0.00 | 20.9 m |
| `prio_all_e9` — **trained** maps | 0.40 | 0.25 | 0.45 | 0.45 | 0.15 | 53.3 m |
| `prio_all_e9` — held-out maps | 0.25 | 0.50 | 0.75 | 0.05 | 0.00 | 20.1 m |
| `expert_idm_rule` — held-out | **1.00** | 0.00 | 0.00 | 1.00 | 1.00 | 98.9 m |
| plain `idm` — held-out | 0.90 | **0.00** | 0.10 | 0.00 | 0.00 | 84.0 m |

There **is** a train-map advantage, and it is larger for `prio_all` (dest +0.15,
crash −0.30) than for `only_stop` (dest +0.05, crash −0.10) — so the gradient is
not zero and I will not claim it is. But the pre-registered arms were "dest high
/ crash low on train maps" versus "dest low / crash high on train maps too", and
**dest 0.25–0.40 with crash 0.45–0.70 on data the model was trained on, against
an expert ceiling of 1.00 on the *harder* held-out maps, is unambiguously the
second.** It never learned to drive these scenes. More maps is not the first
lever.

Per trained map it is uniform rather than one bad map: for `only_stop_e9`,
5 of 8 maps have dest 0.00.

**The pre-registered arm-2 *lever* is refuted by this same run.** Arm 2 named
the speed head and the label bins. On trained maps the stop behaviour is
*perfect* — `halted-before-line 20/20`, `stopline_steps>0` in **0/20** for both
checkpoints — and they still crash 45–70 % of the time. The speed head is doing
its job. Arm 2's *diagnosis* stands; arm 2's *prescription* does not.

## 11. The leading candidate: `_future_path`'s straight-ahead fallback

`PlanT/dataset.py::_future_path` builds the path target from the ego's realised
future over `PATH_HORIZON_FRAMES` (40 = 4 s), arc-length resampled. When the ego
travels less than `path_len + 1.0` = 21 m in that window it *extends* the tail
rather than clamping. The direction is the last inter-frame step; below
`MIN_EXTEND_STEP_M` (0.05 m) it falls back to the net displacement; if that is
also below 0.05 m it falls back to **`np.array([1.0, 0.0])` — the ego's own
forward axis**, discarding the route entirely.

A vehicle held at a stop line for the whole window lands on that last fallback.
Reproduced independently over **40 routes per family, 7.9k–13.4k frames each**:

| family | straight-ahead | net-disp | last-step | no extension | route `\|y\|max` on straight frames (med / p90 / max) |
|---|---|---|---|---|---|
| stop | **22.1 %** | 5.9 % | 47.3 % | 24.8 % | **4.3** / 10.0 / 11.7 m |
| main_road | 21.0 % | 13.5 % | 37.0 % | 28.5 % | 3.7 / 19.0 / 19.1 m |
| secondary_road | 17.4 % | 9.9 % | 54.9 % | 17.8 % | 7.8 / 9.7 / 10.6 m |
| yield | 16.1 % | 10.4 % | 50.4 % | 23.2 % | **7.9** / 8.6 / 13.0 m |
| crosswalk | 20.7 % | 22.6 % | 20.2 % | 36.5 % | **0.1** / 3.2 / 4.7 m |
| bus_lane | **0.0 %** | 2.9 % | 23.8 % | 73.2 % | (0.3) |
| bike_lane | 3.4 % | 2.7 % | 36.8 % | 57.1 % | 0.6 / 0.8 / 1.6 m |
| bus_lane_road | **0.0 %** | 1.9 % | 26.4 % | 71.7 % | — |
| bike_lane_road | **0.0 %** | 0.1 % | 15.2 % | 84.7 % | — |

**The defect is real.** A fifth of stop frames carry a path target pointing
straight down the ego's axis while the navigation route departs by a median
4.3 m. It fits the failure signature: correct stopping, off-road inside the
junction, and near-identical on trained and held-out routes because the *label*
is wrong — no quantity of data fixes a wrong target.

### Three corrections to the original reading

1. **The reserved-lane families are not a control.** The fallback essentially
   never fires there (0.0–3.4 %) because those experts never stop: 57–85 % of
   their frames travel the full 21 m and never reach the extension branch. They
   do not demonstrate "fires but harmless". **Crosswalk is the real control** —
   it fires at 20.7 %, indistinguishable from stop's 22.1 %, but its route is
   straight on those frames (median 0.1 m) and crosswalk's best runs have
   out-of-road 0.05. Same defect rate, different consequence. This is a cleaner
   form of the argument than the reserved-lane comparison would have given.
2. **The crosswalk rate was under-measured** (14.2 % on 12 routes vs 20.7 % on
   40), which strengthens rather than weakens the reading.
3. **"It explains why all four junction families die inside the junction" is not
   supported — three of the four do not.**

| specialist e9, n=40 | dest | **oor** | crash | SR&Dest | straight-ahead rate | route `\|y\|` there |
|---|---|---|---|---|---|---|
| `yield` | 0.85 | **0.00** | 0.15 | 0.60 | 16.1 % | 7.9 m |
| `secondary_road` | 0.50 | **0.05** | 0.50 | 0.30 | 17.4 % | 7.8 m |
| `main_road` | 0.30 | **0.20** | 0.50 | 0.05 | 21.0 % | 3.7 m |
| `stop` | 0.20 | **0.50** | 0.80 | 0.00 | 22.1 % | 4.3 m |

The *rate* of the fallback orders the four families exactly as out-of-road does
(22.1 > 21.0 > 17.4 > 16.1 against 0.50 > 0.20 > 0.05 > 0.00). But the *size of
the lie* runs the other way — `yield` has the largest route mismatch and **zero**
out-of-road. Neither rate nor magnitude alone explains the pattern, their product
is not monotone, and four points is not a trend.

**Standing verdict.** A measured defect in the training label, the best candidate
on the table, consistent with every signature. **Not proven**, does **not**
explain stop-versus-yield, and the monotone-in-rate ordering is far too small a
sample to lean on. Proving it needs a retrain against a corrected target.

## 12. Fixing it — the design tension, for the user to decide

Not implemented. The options are not equivalent.

- **Extend along `route_original`.** Encodes the turn, which is what the current
  fallback destroys. But it partially reintroduces the copy-the-input shortcut
  that `PATH_TARGET=future` exists to prevent: `route_original` is an input token
  the model already holds, and on a stopped frame the *entire* target past the
  first point would come from it. §4.1 measured the future target currently
  deviating from `route_original` by a median 1.27 m; on exactly these frames
  that would drop to zero by construction.
- **Middle option — route direction for the extension tail only.** Keep the
  realised motion for the part actually driven, and use the route's local heading
  only to aim the synthetic tail instead of `[1, 0]`. What is copied is then a
  direction, not a trajectory, and only on frames that have no real motion to
  report. Looks strictly better than both the status quo and a wholesale route
  extension — but untested.
- **Mask stopped frames out of `loss_path`.** Cheapest and safest: invent
  nothing. Costs 16–22 % of junction frames of path supervision, and it is not
  obvious whether that supervision is worth more than the damage it does.

**What must be measured to choose**, none of which exists yet:
1. the fraction of the target that becomes a copy of `route_original` under each
   option, on the affected frames — the shortcut cost, directly comparable to the
   1.27 m in §4.1;
2. closed-loop out-of-road on the four junction families after a retrain under
   each option — the only measurement that actually decides it;
3. whether masking alone recovers most of the gain, which would settle whether
   the problem is the *wrong* target or merely the *invented* one.

A retrain is required either way. This is a decision to take deliberately, not a
patch to land.

## 13. Operational note: an external process killed both eval jobs

At ~00:37 every process of mine died at once — both GPUs, both driver scripts,
SIGKILL — while another user's GPU 4–7 jobs continued. It was **not** the OOM
killer: 1903 GB of 2011 GB were free. My jobs were `nohup`-ed, so their parent is
PID 1, which is exactly what `who.sh` lists under *"СИРОТЫ (родитель 1) —
кандидаты на уборку"*. The most likely cause is another session's orphan cleanup.
Four completed runs survived; one run lost at 8/20 and five never started.

The rerun script was made idempotent (per-shard episode counts, only short shards
re-executed, capped at 4 concurrent workers regardless of shard count) and the
work was completed. Anyone running that cleanup should know it will take out any
long `nohup`-ed eval on this node.

---

# 13. The capability was not missing — it was erased

Everything above treats junction traversal as a skill the model never acquired.
That framing is wrong, and one measurement overturns it.

PlanT-2's CARLA pretraining is full of intersections. Running that pretrain
unchanged on the 20 stop validation episodes, from `$SM/trb_eval_rl3` (no latch
fix, so `dest`/`crash` are comparable with every earlier row):

| checkpoint | dest | out_of_road | crash | median distance |
|---|---|---|---|---|
| `checkpoints/plant2_pretrain/epoch=029_final_3.ckpt` | **0.50** | 0.45 | 0.50 | **98.9 m** |
| `v6best_padded` (`ft_v6pad_e0`) | 0.10 | 0.60 | 0.90 | **15.7 m** |
| `ft_only_stop_e9` | 0.20 | 0.50 | 0.80 | 21.2 m |
| `expert_idm_rule` | 1.00 | 0.00 | 0.00 | 100.4 m |

The pretrain drives 98.9 m of a route the expert covers in 100.4 m. The v6
checkpoint — our fine-tuning base — drives 15.7 m.

**The v6 campaign destroyed junction driving.** Its families are all
straight-road (speed limits, detours, reserved lanes, crosswalk); it was trained
at lr 1e-3 with `TRUNK_LR_MULT=1`, i.e. the whole trunk at full rate. It scores
1.00 on speed limits and 0.9+ on reserved lanes, so it did not fail — it
specialised, and the cost was paid somewhere no straight-road benchmark looks.
Using it as the base for junction signs then starts from a model that has to
relearn what the pretrain already knew.

This reorders every recommendation in §12. The path-target defect (§11) and the
augmentation no-op (§5) are real and measured, but they are second-order against
a 98.9 → 15.7 m regression that happened before any priority-sign fine-tuning ran.

## What this implies for the recipe

The colleague's working stop model (`BEST_EXPERIMENT.md`, compliance 0.738 /
success 0.857) initialises from **this same CARLA pretrain**, at lr 3e-4, trained
jointly over stop and detour. Our recipe initialises from `v6best_padded` at lr
1e-3, one family at a time. On the evidence here the first difference is the one
that matters.

The pretrain is not itself a solution: crash 0.50, out_of_road 0.45, and it has
no 2.5 in its vocabulary at all. The goal is to keep its driving while adding
sign compliance — which is exactly what the colleague demonstrates is reachable.

## The experiment that settles it

One variable changed against the existing `prio_all` baseline — the init:

    CKPT0=<repo>/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt \
    bash tmp/ft_rl3/train_rl.sh prio_pre $FT/splits/prio_all \
      TS_WINDOW_FRAMES=default:1,3.24:8,5.31:8,5.21:8 MAX_EPOCHS=10 \
      CKPT_EVERY_N_EPOCHS=2 EVAL_FAMS="main_road secondary_road yield stop" \
      EVAL_EPOCHS="1 3 5 9" EVAL_VARIANTS=base

`CKPT0` is `${CKPT0:-...}` in `lib_rl.sh`, so no script edit is needed. Keep lr at
1e-3 for this run: changing init and lr together would leave the outcome
unattributable, which is exactly how three GPU-hours were lost earlier in this
campaign. If the init alone does not recover the driving, lr 3e-4 is the second
run, not a simultaneous one.

## Caveats

n = 20 per row, so ±0.2 on the rates; 98.9 m against 15.7 m is not marginal, but
the rates are. The pretrain was evaluated with a 45-class sign vocabulary against
a 49-class model config — it loaded and ran, but it cannot be said to have been
given the sign at all, which is the point of the comparison rather than a flaw in
it. Whether the loss happened during the v6 campaign or never transferred from
CARLA to MetaDrive/SUMO is not separated here; the colleague reports the same
pretrain crashing 100% on their detour benchmark, which argues the transfer is
fragile in both directions.
