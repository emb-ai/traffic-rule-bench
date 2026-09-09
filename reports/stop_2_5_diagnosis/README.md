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

Prediction confirmed. The expert halts to a dead stop in every episode, never
trips the yield clause, and is still flagged. C1 is real and dominant.

The single unflagged episode (`junc_134684395_td50_sv0_v1`) is the one where
`crossed` is `False` even after the fix: its ego left the sign lane before the
longitudinal projection passed the verdict point, so the clause was never
sampled. The pre-fix 0.05 was that accident, not a pass.

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
   longitudinal projection has passed its end — that is how one of 20 episodes
   escaped the clause entirely, in both directions. Setting it to 5.0 would put
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
