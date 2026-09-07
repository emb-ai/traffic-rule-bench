# Detour (4.2.1/4.2.2/4.2.3) obstacle-avoidance — hypothesis loop

Goal: sign_compliance > 0.5 on the 122-scene test catalog
(`$SM/traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl`),
via genuine learned swerving — no waypoint post-processing.

## Baseline (no hypothesis, plain fine-tune)

`checkpoint-addon = detour_4212223_lr1e4_ep20`, LR=1e-4, 20 epochs, batch=512,
same pretrain ckpt as stop-sign (`epoch=029_final_3.ckpt`), data =
`detour_split_filtered` (925 train / 161 val, cone-absent routes dropped).

**Result**: sign_compliance 4.2.1=0.000, 4.2.2=0.020, 4.2.3=0.022 (366 runs,
success rate 0.000 across the board).

**Diagnosis** (before formulating H1):
- 98.6% of crashes land within ±5m of the obstacle's exact spawn position
  (mean offset -1.46m, median -0.74m) — not random failures, the model
  drives essentially straight into the cone every time.
- Live `x_objs` trace on a representative scene (`sumo_4.2.1_114520`) shows
  the sign token (class 16.0) and 4 cone tokens (class 3.0, generic
  "static") correctly present and visible for the last 68 frames before
  impact — so this is not a perception/visibility bug (already separately
  fixed today via `PLANT2_DUMP_SIGN_CLASSES`).
- Steering output (`action[0]`) stays within ±0.002 (essentially zero) for
  every frame right up to the crash, at ~14 m/s (50+ km/h) — the model
  isn't attempting any lateral deviation at all. This points at the
  waypoint/path loss not being weighted enough to force learning the rare
  (only a handful of frames per route are near the obstacle out of the
  whole ~80-100m route), but critical, lateral swerve.
- Aside: crash_attribution is unconditionally "npc" for any collision with
  a non-`BaseVehicle` object (`bench/sign_eval.py::_ego_at_fault_for_crash`
  only checks `isinstance(obj, BaseVehicle)`) — cones never match, so this
  field is uninformative for cone collisions specifically. Not fixed (out
  of scope for the training loop; diagnosis used `distance_travelled_m` vs
  `sign_spawn_distance` instead, which is unambiguous).

## H1: boost path-loss weight

**Hypothesis**: `model.waypoints.path_weight` (default 5, vs `wps_weight=1`,
`speed_weight=1`, `pre_training.forecastLoss_weight=1`) isn't dominant
enough for the model to prioritize the rare lateral-deviation frames over
the much more common "drive straight" frames that make up most of every
route. Raising it substantially, and reducing the competing
`forecastLoss_weight` (a future-frame auxiliary task, unrelated to
avoidance) to redirect capacity, should push the model toward learning to
swerve.

**Change**: `model.waypoints.path_weight=25 model.pre_training.forecastLoss_weight=0.2`,
otherwise identical to baseline (same data, LR, epochs, pretrain ckpt).

`checkpoint-addon = detour_h1_pathweight25_lr1e4_ep20`

**Status**: launched 2026-08-26 04:25 MSK, pid 315406, GPU 7; restarted 04:30
MSK (pid 317273) reusing the already-warm 81G `/tmp/plant2_ds_cache_detour`
disk cache instead of a fresh one (safe — cache keys on raw route data,
independent of loss weights/augmentation) after ~0.3 it/s first-run cold
start; confirmed ~1.0-1.1 it/s once warmed up (~6-7 min/epoch). Training
completed cleanly 07:51 MSK.

**Result**: sign_compliance 4.2.1=0.000, 4.2.2=0.020, 4.2.3=0.022 — **within
noise of baseline** (was 0.000/0.020/0.022). `avg_distance`/`avg_efficiency`
per-sign match baseline to 3 decimal places despite genuinely different
checkpoint weights (verified via md5 — not a stale-checkpoint bug). H1
falsified: raising path_weight 5x (uniformly across ALL frames) doesn't
change *relative* pressure between the ~96% of frames that need ~0 lateral
deviation and the ~4% that need a real 2-3m swerve — it scales both
equally, so it can't fix the imbalance. Confirmed ground-truth training
labels DO contain the real swerve (checked `measurements/*.json.gz`
`route` field directly on a training route: lateral offset jumps from
-0.78 to 2.61 across the lookahead in the obstacle-proximity frame) — so
this isn't a data/label problem, it's a loss-imbalance problem.

## H2: per-sample lateral-deviation loss reweighting

**Hypothesis**: the frame-level imbalance itself is the problem, not the
overall path-vs-other-losses weight. Reweight `loss_path` *per sample*
by how much lateral deviation its own ground-truth path requires:
`weight = 1 + alpha * max(|path_y|)` over the sample's path points — flat
"drive straight" frames stay at weight 1, a frame needing a 3m swerve gets
`1 + 3*alpha`. Implemented in `plant2/PlanT/lit_module.py::_path_loss()`
(new method, used in both `training_step`/`validation_step`), gated by new
config key `model.waypoints.lateral_reweight_alpha` (default 0.0 = exact
prior uniform-mean L1 behaviour, verified via unit test: alpha=0 output
bit-identical to `F.l1_loss` before this change).

**Change**: `model.waypoints.lateral_reweight_alpha=10 model.waypoints.path_weight=5 model.pre_training.forecastLoss_weight=1`
(path_weight/forecastLoss_weight reset to baseline defaults — isolating
the new mechanism as the only change from baseline, since H1's global
reweight already showed no effect on its own).

`checkpoint-addon = detour_h2_lateralreweight10_lr1e4_ep20`

**Status**: launched 2026-08-26 08:17 MSK, pid 324990, GPU 7. Completed
cleanly 11:46 MSK (train/loss_path 0.087→0.055 over 20 epochs — the
reweight is visibly doing something during training).

**Result**: sign_compliance 4.2.1=0.000, 4.2.2=0.020, 4.2.3=0.022 — again
within noise of baseline, `avg_distance` matching to 3 decimals despite a
genuinely different checkpoint (3rd run in a row with this pattern).

**Deeper diagnosis**: confirmed `pred_path` (not `pred_wps`) is what
actually drives steering (`plant2_adapter.py::_pid_action_persistent`,
"prefer pred_path, fallback pred_wps") — so I was reweighting the right
tensor. Added a temp debug print of raw `pred_path` lateral (y) values and
compared baseline vs H2 on the identical scene/frame
(`sumo_4.2.1_114520`), right before impact:
- baseline: y drifts monotonically +0.001 → +0.024 m across the 20-point
  lookahead (near-flat, all positive)
- H2: y dips to **-0.042** then crosses to **+0.071** m — a qualitatively
  different, non-monotonic shape. The reweight IS changing what the model
  predicts.
- But both are **centimeters**, not the ~2-3 **meters** the ground-truth
  training label requires at the same route position (see baseline
  diagnosis above). ~30-70x too small to matter physically.

So H2's mechanism is directionally validated but alpha=10 is nowhere near
strong enough to move the prediction into a physically meaningful range.
Two live possibilities: (a) just needs much more alpha — worth a
value-scaling test before concluding it's an architecture/capacity limit;
(b) something scale-limits the path head's output regardless of loss
pressure (e.g. residual/tanh-bounded decoder, or 20 epochs @ LR=1e-4 is
too conservative to move the value that far against the pretrained prior)
— if (a) also plateaus, that's the signal to switch to (b)-type fixes
(higher LR, more epochs, or inspect the path decoder architecture).

## H3: much larger lateral-reweight alpha

**Hypothesis**: H2 validated the mechanism (qualitative shift, correct
direction of effect) but under-powered it. Scale alpha 10x (10→100) to
test whether the effect scales roughly linearly toward the needed
magnitude, or plateaus (which would rule out "just needs more weight" and
point at a capacity/architecture limit instead).

**Change**: `model.waypoints.lateral_reweight_alpha=100`, otherwise
identical to H2 (path_weight/forecastLoss_weight at baseline defaults).

`checkpoint-addon = detour_h3_lateralreweight100_lr1e4_ep20`

**Status**: launched 2026-08-26 (pid 335452, GPU 7). Completed cleanly
16:06 MSK — train/loss_path=0.0558, essentially identical to H2's 0.0551
despite 10x more alpha (loss barely moved further).

**Raw pred_path check** (same scene/frame as H2's comparison,
`sumo_4.2.1_114520`): max_abs_lateral progression baseline=0.024 →
H2(alpha=10)=0.07 → H3(alpha=100)=0.10 m. A 10x alpha increase bought
only ~1.4x more magnitude — clearly **sublinear/plateauing**, not the
roughly-proportional scaling you'd expect if it were purely "not enough
weight yet". Checked the path decoder architecture
(`model.py::GRUWaypointsPredictorInterFuser`) for a hard cap (tanh/sigmoid
clamp) — none found, it's `nn.Linear` → `torch.cumsum`, fully unbounded.
So this isn't a hard architectural ceiling, but it IS acting like the
optimizer can't push the prediction much further via loss-reweighting
alone at LR=1e-4/20 epochs — likely the pretrained "drive straight" prior
is strong and this LR/epoch budget is too conservative to overcome it
even with a heavily reweighted loss. (Noted, not fully confirmed: the
path GRU's initial hidden state is always zeros —
`target_point_size=0` in the `path_generator` init — so scene context
only reaches it via per-step input, never via initialization; a
contributing factor, not chasing an architecture change for now since
alpha clearly buys *something*, just slowly.)

Full 122-scene eval launched for completeness/record; expect similar-to-H2
numbers based on the raw-prediction check, not waiting on it to decide H4.

## H4: higher learning rate

**Hypothesis**: H2/H3 show the reweighting mechanism nudges the model in
the right direction but LR=1e-4 over 20 epochs isn't enough step size to
move the prediction into a physically meaningful range against a strong
pretrained straight-driving prior. Keep the strongest working reweight
(alpha=100) and raise LR 3x — LR and loss-reweighting are complementary
levers (reweighting says *where* to push, LR controls *how far/fast*),
not substitutes for one another.

**Change**: `--learning-rate 3e-4` (was `1e-4`) with
`model.waypoints.lateral_reweight_alpha=100`, otherwise identical.

`checkpoint-addon = detour_h4_lateralreweight100_lr3e4_ep20`

**Status**: launched 2026-08-26 16:11 MSK (pid 401435/401436, GPU 7).
Completed cleanly ~17:38 MSK (train/loss_path 0.086→0.051, noticeably
lower than H2/H3's ~0.055-0.058 at the same alpha=100/10 — LR change is
visibly moving the loss further this time).

**Result**: sign_compliance 4.2.1=0.000, 4.2.2=0.020, 4.2.3=0.022 — 5th
run in a row (baseline, H1-H4) landing at exactly this point.

**Raw pred_path check, both `best_016` and `last_ft`**: max lateral
magnitude ~0.04-0.06m — actually *smaller* than H3's ~0.10m despite 3x
higher LR (ruled out "best checkpoint by val/loss_all discarded the
better epoch" — checked `last_ft` too, same tiny range). LR is not the
missing lever either.

**Revised diagnosis**: four different loss-reweighting/LR combinations
all plateau in the same ~0.04-0.10m range, nowhere near the ~2-3m needed
— and scaling the "pressure" (alpha 10→100, LR 1e-4→3e-4) barely moves it
or actively doesn't help. Best explanation: **Adam's per-parameter
adaptive normalization is absorbing the reweighting.** Adam divides each
parameter's update by a running estimate of that parameter's own gradient
magnitude — scaling one sample's loss 10-100x doesn't scale the actual
step size anywhere near that much, because the running variance estimate
for the same parameters adapts alongside it. This is a well-known failure
mode of "just multiply the loss by a bigger number" under Adam. It would
explain the sub-linear plateau across every alpha/LR combination tried so
far, cleanly.

## H5: oversample near-obstacle frames instead of reweighting their loss

**Hypothesis**: reweighting the *loss value* of a sample doesn't change
how many gradient *steps* incorporate it — under Adam that's a
qualitatively weaker lever than reweighting suggested (see above).
Reweighting the *sampling frequency* instead — literally including
near-obstacle frames many more times across an epoch — changes the
running gradient statistics Adam actually tracks, which loss-scaling
can't replicate. This is a different mechanism from H1-H4, not another
turn of the same knob.

**Plan**: `PlanTDataset.__init__` builds one entry per (route, frame)
position in `self.BEV`/`self.labels`/`self.measurements` — frame-level
already, good. Added `_oversample_repeat()`: duplicates a frame's entry
`oversample_factor` times if its own `boxes/*.json.gz` has a cone
(`type_id=='static.prop.constructioncone'`) within `oversample_cone_distance_m`
(ego frame, straight-line distance). Gated by both new keys defaulting to
0/1 = exact prior behaviour (verified: default config gives identical
`len(dataset)` before/after this change, 30855 samples on the val split).

**First attempt used the wrong signal**: tried gating on the ground-truth
path's own lateral extent (`max(|path_y|)`, same quantity `_path_loss`
uses) instead of cone proximity — found this hits **59% of all frames**
at a 0.5m threshold, and still 16% at 3.0m (max lateral value in the
dataset is ~20m). Ordinary turns/junctions produce lateral values in the
same range as an obstacle swerve in this representation, so it can't
distinguish "needs to dodge a cone" from "route has a bend in it" —
switched to checking the frame's own perceived cone distance instead,
which is unambiguous. Swept distance thresholds on the val split (all
`factor=8`): 5m→7.6% of frames flagged, 10m→15.4%, 15m→22.7%, 20m→30.1%.
Picked **5m** — closest to the ~4-8% "near obstacle" fraction expected
from the route-length/swerve-window ratio, and matches the earlier
crash-clustering evidence (crashes land within ±5m of the obstacle).

**Change**: `model.training.oversample_cone_distance_m=5 model.training.oversample_factor=8`,
`model.waypoints.lateral_reweight_alpha=0` (pure oversampling, isolated
from H2-H4's loss-reweighting mechanism), LR back to baseline 1e-4.

`checkpoint-addon = detour_h5_oversample8_d5_lr1e4_ep20`

**Status**: launched 2026-08-26 22:03 MSK (pid 479130, GPU 7). Dataset
construction now reads every frame's boxes file at init (new I/O this
hypothesis introduces) so startup is slower than H1-H4. Train dataset
built: **282078 samples vs ~192512 baseline (1.46x)** — consistent with
7.6% of frames duplicated 8x. Training running.
**Result**: <!-- fill in -->

---

# The route-copying shortcut (found before H6-H10)

H1-H4 all failed the same way, and the reason turned out to be visible in
the model's inputs rather than its losses.

`model.py:287` embeds `batch["route_original"]` as the **first token** of
the sequence — this is the SUMO-planned route. For 4.2.x detour scenes
that route runs **straight through the obstacle** (the whole premise of
the benchmark: SUMO routing is unaware of the cone; documented in
`README.md` and re-verified here). The regression target `route` is the
*actually driven* path, which swerves around it.

So "copy `route_original` into `pred_path`" is a near-optimal strategy on
the training objective — correct on ~96% of frames, wrong only in the
handful near the obstacle. And that is exactly what the measurements show
the model does: predicted lateral ~0.04m == "follow the route exactly".

This reframes H1-H4's failure: they were all trying to *reweight the loss*
of a shortcut that is genuinely near-optimal **under that loss**. No amount
of reweighting makes copying the route a bad strategy when the route is
right 96% of the time — it just makes the 4% more expensive, and Adam's
per-parameter normalization absorbs much of even that. What can work is
making the shortcut *unreliable*.

**New mechanism** (`model.py::_perturb_route`, training-time only, not
applied under `eval()` so val loss stays comparable and inference is
untouched):
- `model.training.route_dropout_p` — per-sample probability of zeroing the
  entire route input, forcing the path to come from perception (x_objs/BEV).
- `model.training.route_lateral_noise_m` — per-sample Gaussian lateral
  offset (std, m) on the route, keeping it as a hint but an unreliable one
  that perception must correct.

Both default 0.0 = disabled. Verified numerically before spending GPU time:
default is bit-identical to the unmodified path; `eval()` untouched even
when enabled; dropout zeroes 49.1% of samples at p=0.5 and leaves the rest
bit-identical; noise shifts only `y` (never `x`), is constant along a
sample's path, and has measured std 1.502 at σ=1.5.

**Shared-data safety** (all 6 runs share one dataset + one diskcache):
verified the cache is keyed by **frame file path**
(`dataset.py:298`, `labels[0].decode()`), not dataset index — so runs with
different oversampling factors (hence different dataset lengths) reuse the
same entries for the same frames instead of corrupting each other. Nothing
writes into the shared data dirs. GPUs 0 and 3 left alone (other users).

## H6-H10 (launched in parallel 2026-08-26 22:1x MSK)

Launcher: [`launch_detour_h6_h10.sh`](launch_detour_h6_h10.sh). All at
LR=1e-4, 20 epochs, same split/pretrain ckpt as baseline — so each is
attributable against baseline and against H5.

| # | GPU | Change | Question it answers |
|---|-----|--------|---------------------|
| H6 | 1 | `route_dropout_p=0.5` | Does removing the shortcut alone produce real swerving? |
| H7 | 2 | `route_lateral_noise_m=1.5` | Is a *degraded* route better than a *missing* one (keeps goal info, kills exact-copy)? |
| H8 | 4 | `route_dropout_p=0.8` | Dose-response: is more removal monotonically better, or does it break general driving? |
| H9 | 5 | `route_dropout_p=0.5` + oversample 8x/5m | Shortcut removal + H5's frame oversampling. |
| H10 | 6 | `route_dropout_p=0.5` + `lateral_reweight_alpha=30` | Shortcut removal + H2's loss reweighting (reweighting may finally bite once the shortcut is gone). |

H8 is the one with a real failure mode worth watching: at p=0.8 the model
rarely sees the route at all, which may damage ordinary navigation
(knowing which way to go at a junction) even if it helps avoidance —
that trade-off is itself the useful signal.

### Infrastructure bug found by running these in parallel (fixed)

The first parallel launch killed H6 at epoch 0 with
`KeyError: '.../boxes/0166.json.gz'` raised from inside a DataLoader worker.

Root cause: `dataset.py`'s cache lookup was `if key in self.data_cache:`
followed by `self.data_cache[key]` — a **TOCTOU race**. With six trainings
sharing one diskcache that was sitting *at* its size limit (measured 81G
against an 80G limit, so evicting constantly), another process can evict
the key between the check and the read, and the read raises. Single-run
training never hit this because nothing else was evicting concurrently.

Fixed to the atomic `Cache.get(key)`, which returns `None` on a miss so the
code just falls through and reloads from disk. Raised the shared cache
limit 80G → 250G (disk has ~626G free) so it stops thrashing.

While fixing, a self-inflicted bug was caught by the verification test
before it ever reached GPU: the first version of the rewrite assigned the
lookup result straight to `sample`, but `sample` is already bound to the
dict built earlier in `__getitem__` and the load-from-disk path fills that
dict *in place* — so a cache miss (`None`) broke the fallthrough with
`TypeError: 'NoneType' object does not support item assignment`. The
lookups now use separate locals and never rebind `sample`. Verified after
the fix: cache hit path works (20 samples, 3.37s cold → 0.02s warm, ×204),
samples come back as proper dicts, and the no-cache path still works.

All six runs were restarted on the fixed code so none of them carry the
old module in memory.

### Scheduling note (shared box)

Other users' jobs (smirnova, spiridonov) occupy GPUs 0/3/4/6, so only 4
GPUs were free at restart. Rather than co-locating onto their cards, the
six hypotheses are run in two batches via a generic single-run launcher,
[`launch_detour_one.sh`](launch_detour_one.sh).

**Batch 1 (running)** — the four most informative:

| # | GPU | pid | Change |
|---|-----|-----|--------|
| H5 | 1 | 498399 | oversample 8x within 5m of cone |
| H6 | 2 | 498403 | `route_dropout_p=0.5` |
| H9 | 5 | 498407 | dropout 0.5 + oversample 8x/5m |
| H10 | 7 | 498411 | dropout 0.5 + `lateral_reweight_alpha=30` |

Configs verified applied in each log; dataset sizes confirm the
oversampling took effect (282078 samples for H5/H9 vs 192156 for
H6/H10). Note logs are *appended* across restarts, so the pre-restart
crash traceback is still visible in H6's log above the new run's config
block — not a new failure.

**Batch 2 (queued, pending free GPUs)**: H7 (`route_lateral_noise_m=1.5`)
and H8 (`route_dropout_p=0.8`). **Cancelled** — see root cause below.

**Result**: H5/H6/H9/H10 all 0.000 / 0.020 / 0.022 (overall 0.014) — the
**ninth** consecutive run, counting baseline and H1-H4, landing on exactly
these numbers.

---

# ROOT CAUSE: `route_original` leaks the training target

Nine different interventions producing identical metrics is not nine failed
hypotheses, it is one wrong premise. Stopping the hypothesis loop to find it
turned up a data bug that invalidates all of them.

### 1. The model HAS learned to swerve

Ran the trained H6 checkpoint on held-out **validation frames** (not the
simulator), on frames with a cone within ~10m whose label needs a real
maneuver, comparing prediction against ground truth:

| cone dist | label max\|y\| | pred max\|y\| | L1 |
|---|---|---|---|
| 3.3 m | 4.32 | **4.38** | 0.099 |
| 3.3 m | 2.15 | **2.41** | 0.070 |
| 9.2 m | 5.48 | **5.32** | 0.092 |
| 2.3 m | 4.45 | **4.47** | 0.068 |

Metres of lateral deviation, accurately. So this was never a training
problem — which is what every one of H1-H10 assumed.

### 2. Why: the input contains the answer

In the dumps, the model INPUT `route_original` is **elementwise identical**
to the regression TARGET `route`:

```
DETOUR train (4.2.x)   identical=1475/1475   max deviation=0
DETOUR val   (4.2.x)   identical= 914/914    max deviation=0
STOP  train  (2.5)     identical=1793/1793   max deviation=0
```

Deviation exactly `0` — not "similar", identical. The dump wrote the driven
trajectory into `route_original` instead of the obstacle-ignorant SUMO plan.
So "copy `route_original` to `pred_path`" is not near-optimal on this data,
it is *exactly* optimal, and that is the identity mapping the model learned.

In the simulator, `route_original` is the genuine SUMO route — straight
through the cone. The model faithfully copies it and drives in.

This explains every observation at once: excellent val numbers (copying),
total simulator failure (copies a straight route), and complete immunity to
loss reweighting / oversampling / LR (you cannot make a model learn
perception when the answer is handed to it in its input).

Note this leak is **not detour-specific** — the stop-sign dumps have it too.
Stop-sign still scored 0.45-0.57 because stopping is a *speed* decision, and
the speed head had to genuinely learn it; only the *path* head was shortcut.

### 3. Proof: remove the leaked input and the maneuver appears

Same H6 checkpoint, same scene, only difference is zeroing `route_original`
at inference (`PLANT2_ZERO_ROUTE=1`, the inference-side counterpart of
training's `route_dropout_p`; H6 trained at p=0.5 so a zeroed route is
in-distribution for it):

| inference | max lateral over episode |
|---|---|
| route fed (leaked input present) | **0.072 m** — drives into the cone |
| route zeroed | **8.87 m** — plans a real avoidance maneuver |

### Consequences

- H1-H10 are void as tests of "how do we teach avoidance": they were all
  tuning a model whose objective was already solved by copying.
- The real fix is upstream: regenerate the dumps so `route_original` holds
  the planned SUMO route, not the executed trajectory.
- Meanwhile the leak can be bypassed with existing data + existing
  checkpoints by removing the route input on both sides
  (`route_dropout_p` in training, `PLANT2_ZERO_ROUTE` at inference).

### Also fixed while diagnosing: BEV scale mismatch

`plant2_adapter.py` passed `bev_resolution=128` to
`metadrive_obs_to_plant2_batch`, but that function only applies
PlanTDataset's `[64:-64]` centre crop when the render is 256x256. So the
model was fed a 128px BEV covering 64 m (2 px/m), against training's
128px/32m (4 px/m) — half the resolution, twice the field of view. Now
renders at 256 so the crop triggers and matches training. Real bug, but
*not* the cause here (fixing it alone did not change behaviour) — and it
affects stop-sign eval too, so those numbers need re-verifying.

### Zero-route eval: 0.836 compliance, but it is VACUOUS — not a solution

Full 122-scene eval with `PLANT2_ZERO_ROUTE=1`:

| run | 4.2.1 | 4.2.2 | 4.2.3 | overall |
|---|---|---|---|---|
| H6 zero-route | 0.783 | 0.859 | 0.870 | **0.836** |
| H9 zero-route | 0.736 | 0.879 | 0.833 | 0.811 |
| any run with route fed | 0.000 | 0.020 | 0.022 | 0.014 |

That clears the >0.5 target numerically, and it is **not** a real result.
Checking where episodes actually end, against each scene's obstacle
position:

| | median gap to obstacle | died AT obstacle (\|gap\|≤5m) | died long before (gap<-15m) |
|---|---|---|---|
| route fed | −0.7 m | **99%** | 1% |
| route zeroed | **−41.2 m** | 2% | **87%** |

Without the route the model loses navigation entirely: 100% crash rate, 0
arrivals, 21.7m travelled out of a 258m route (8%). It dies dozens of
metres *before* the obstacle, so no sign violation is ever recorded and
"compliance" is high by vacuity. The earlier 8.87m lateral was not an
aimed detour, it was the policy flailing without a goal.

**Lesson for this loop**: `sign_compliance` alone cannot be the success
criterion — a policy that never reaches the sign scores perfectly. It has
to be read with arrival/route-completion, which no target above did.

### Where this actually leaves the task

- The leak is proven and is the blocker: the model was trained to copy an
  input that already contained the answer, so it never had to perceive the
  obstacle.
- Deleting the route at inference is *not* the fix — the model legitimately
  needs the route for navigation; removing it trades one failure for a
  worse one.
- **The real fix is upstream in dump generation**: `route_original` must
  hold the planned SUMO route (obstacle-ignorant), not the executed
  trajectory. Then "copy the route" stops being optimal, the perception
  path has to do the work, and the existing training machinery (plus,
  possibly, some of H1-H10's reweighting) becomes meaningful again.
- This affects the whole benchmark, not just 4.2.x — the stop-sign dumps
  carry the identical leak.

Until the dumps are regenerated, no training-side hypothesis on this data
can be expected to teach obstacle avoidance.

### Measured: how much of the prediction is just the route input?

Not inferred from the arrays being equal — tested by feeding a *perturbed*
route on val frames and watching whether the prediction follows it:

| | route shifted +3.00 m → prediction shifts | route zeroed → pred max\|y\| |
|---|---|---|
| **baseline** (no dropout) | **+2.726 m** (91% of the shift) | **0.208 m** |
| H6 (`route_dropout_p=0.5`) | +2.797 m (93%) | 2.595 m |

The baseline is functionally a copier: the predicted path tracks the route
input ~1:1, and with the route removed it outputs an almost flat path — it
has no independent path-planning ability at all. (Not literally 100%: L1 to
target is 0.29 m and ~9% of the shift doesn't transfer, but the route
dominates.)

**The copying is inherited from pretraining, not created by our fine-tune.**
Same probe run against the pretrain checkpoint itself, using its own
architecture (`tok_emb`, via a throwaway git worktree at `90d0c098^`, the
commit before `class_emb` replaced it — loading it into the current model
would leave the whole object encoder random and prove nothing):

| | route +3.00 m → pred shift | route zeroed → max\|y\| | L1 |
|---|---|---|---|
| pretrain (own arch) | **+2.275 m (76%)** | **0.805 m** | 0.525 |
| baseline (our FT) | +2.726 m (91%) | 0.208 m | 0.291 |

The pretrain checkpoint already follows the route 76% and produces almost
nothing without it. Our fine-tune on the leaky dumps only sharpens an
existing habit. So the leak is very likely present in the pretraining data
too — a whole-pipeline issue, not something the detour/stop dumps
introduced.

Caveat on that measurement: 46 keys were missing on load — `tok_emb.7..27`,
the PDD *sign* embeddings, because pretraining used only 7 object types
while that revision's config expects 28. Cones load correctly (they are
`static`, index 3), and cones are what carry the obstacle information, so
the route-copying conclusion holds; what this probe does *not* test is the
model's response to the 4.2.x sign token itself.

H6 shows what route dropout actually bought: a *conditional* — copy the
route when present (93%), fall back to perception when absent (2.6 m). That
is why H6 didn't help at eval: the route IS present there, so it copies the
straight line into the cone. Earlier note "route dropout didn't help" was
right about the outcome but wrong about the mechanism — dropout does not
remove copying, it only adds a fallback.

Wider consequence: on these dumps the path head never learns to plan from
perception, it learns to reproduce the route. Any trajectory-quality metric
measured on this data is measuring route-following, not scene understanding.
The speed head is unaffected, which is why stop-sign still produced
meaningful numbers — braking had to be learned honestly.

### Stop-sign re-verified after the BEV fix (shared code)

Same checkpoint (`stop_classemb_lr3e4_ep30/best_011`), 42 scenes:

| | sign compliance | success rate | avg dist |
|---|---|---|---|
| before BEV fix | 0.452 | 0.667 | 79.7 m |
| after BEV fix | **0.500** | 0.548 | 72.1 m |

Compliance up, success rate down. The fix is objectively correct (the model
now gets the BEV scale it was trained on, where before it got a 2x zoomed-out
one it had never seen), so 0.500 is the truthful number and 0.452 was
measured on an out-of-distribution input. But the trade is not one-sided and
any stop-sign numbers quoted from before this fix should be re-measured, not
compared directly against post-fix ones.

---

# Fixed-target era: H11-H21, and the structural ceiling

With `route_original` no longer equal to the target, the mechanisms from
H1-H10 finally test something real. Results (all verified to have loaded the
checkpoint, `get_action failed = 0`):

| run | compliance | success | mean dist | past obstacle |
|---|---|---|---|---|
| H11 fixed target, no extras | 0.055 | 0.123 | 73.4 m | 24% |
| H12 oversample <=5 m | 0.096 | 0.131 | 72.8 m | 23% |
| H13 lateral alpha=30 | 0.079 | 0.150 | - | - |
| H14 oversample + alpha | 0.027 | 0.107 | - | - |
| H15 route_dropout 0.3 | 0.145 | 0.025 | 55.9 m | 15% |
| H16 lr 3e-4 | 0.066 | 0.175 | 82.3 m | 39% |
| H17 oversample 40-70 m x8 | 0.074 | 0.101 | 72.0 m | 27% |
| H18 oversample 40-70 m + drop 0.3 | **0.131** | 0.101 | 69.8 m | 27% |
| H19 drop 0.3 + lr 3e-4 | 0.189 | 0.011 | 55.5 m | 12% |
| H21 drop 0.5 + lr 3e-4 | 0.628 | 0.000 | 32.1 m | 2% |

## `route_dropout` inflates compliance by breaking driving

The trend is monotonic and it is an artefact, not progress:

    dropout 0.0 -> 0.055 compliance, 73 m driven, 24% past obstacle
    dropout 0.3 -> 0.145-0.189,      56 m,        12-15%
    dropout 0.5 -> 0.628,            32 m,         2%

The compliance rule only asks the vehicle to *be in the adjacent lane while
inside the zone*. A car that drifts sideways immediately and then stalls or
crashes satisfies it perfectly. So **"compliance > 0.75" is reachable
trivially and meaninglessly** — H21 is already at 0.628 while driving 32 m
and clearing the obstacle in 2% of scenes. Any target for this metric has to
be paired with route completion.

By that honest standard the best run is **H18: 0.131 at undegraded driving**
(27% past obstacle, 69.8 m) — a 2.4x gain over H11 with no loss elsewhere.

## Why the model is late, and the ceiling that follows

- It always changes lane (0 episodes never change), but **late**: median 31%
  of in-zone steps are still violations.
- The training data does contain the right behaviour: the expert starts its
  shift a median **51.5 m** before the cone and completes it before the zone
  in 77% of routes.
- Perception range is not the limit for cones: training sees objects to
  `range * range_factor_front` = 50*2 = **100 m**, and cones are dumped out to
  ~100 m. (An earlier hypothesis blamed a 50 m limit; measured and dropped.)
- **The sign is the limit.** `collect_boxes` capped sign boxes at 30 m, while
  compliance demands the lane change be finished at zone entry, when the sign
  is still ~26.5 m ahead. On the 171/366 scenes *without cones* the sign is
  the only cue, so those are unsolvable from perception. The privileged
  expert scores **18/18** on exactly those scenes because it reads the scene
  config rather than seeing the sign.

So the honest perception-only ceiling under a 30 m sign cap is about
195/366 = **0.53**, and >0.75 is unreachable without changing it.

## Fix in progress: sign visibility

`plant2_frames.py` now takes `PLANT2_SIGN_RANGE_M` (default 30, unchanged)
instead of the hardcoded 900.0 radius. Verified end-to-end: at 90 m the sign
is present in 95/95 frames of a probe episode versus 69/95 at 30 m.

Existing dumps contain no signs beyond 30 m, so this needs a re-dump:
1461 train80 rows, 8 shards, `PLANT2_SIGN_RANGE_M=90`, into a fresh
directory (shared source dumps untouched). The dump still writes
`route == route_original`, so `fix_route_target.py` must be re-applied to
the result before training.

## BREAKTHROUGH: early sign visibility (90 m) — the structural fix

Re-dumped all 1461 train80 routes with `PLANT2_SIGN_RANGE_M=90` (sign now
visible at a median 59 m, max 90 m, instead of a hard 30 m cap), rebuilt the
split **by scene** (1242/219; the old split divided by *route*, so the same
scene appeared in train and val under different seeds — its val numbers were
optimistic), and re-applied `fix_route_target.py`. Verified before training:
leak gone (0/853 identical route/route_original pairs, mean deviation 4.91 m)
and the sign reaches 59 m through the split symlinks. Eval also run with
`PLANT2_SIGN_RANGE_M=90` so train and eval agree.

| run | compliance | success | mean dist | past obstacle |
|---|---|---|---|---|
| **H26 s90 ovs40-70 + drop0.3, 7 ep (best)** | **0.626** | **0.158** | 68.6 m | 31% |
| H25 s90 lr3e4 (last) | 0.607 | 0.104 | 56.2 m | 23% |
| H22 s90 base (last) | 0.601 | 0.139 | 59.1 m | 24% |
| H25 s90 lr3e4 (best) | 0.536 | 0.156 | 64.1 m | 33% |
| H18 old 30 m cap (best) | 0.131 | 0.101 | 69.8 m | 27% |
| H11 old 30 m cap | 0.055 | 0.123 | 73.4 m | 24% |

**This is not the degenerate pattern.** Compliance 0.131 -> 0.626 while
driving *improved*: success 0.101 -> 0.158, past-obstacle 27% -> 31%, mean
distance essentially unchanged. Contrast H21 (old data, dropout 0.5) which
reached 0.628 with 2% past-obstacle and 32 m driven.

**H22 — plain fine-tune, no extra mechanism — alone reaches 0.601**, which
confirms the diagnosis: the binding constraint was the 30 m sign cap, not the
training objective. Every mechanism from H1-H21 was tuning around a
perception limit that made 47% of scenes unsolvable.

Note `best`-by-`val/loss_all` picked epoch 0-1 checkpoints and is again a poor
proxy (H22 best 0.191 vs last 0.601), so both are evaluated from now on.
`val/loss_path` also rose (0.34-0.43 vs 0.237) — expected, since the new
scene-level split removed the old train/val scene overlap; it is not
comparable to earlier numbers.

Remaining gap to the 0.75 target: 0.626 -> 0.75. Next batch (H27-H30, 7
epochs each for equal sample budget) sweeps around the winning H26 config:
lr 3e-4 vs 1e-4, dropout 0/0.15/0.3, and a wider oversample window (25-90 m).
