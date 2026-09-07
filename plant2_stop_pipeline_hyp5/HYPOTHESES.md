# 5 hypotheses for raising 2.5 (stop-sign) compliance

## Background

Two known reference points, both on the signfix split (294 train / 50 val
routes), same pretrain checkpoint (`stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`):

| Architecture | sign_compliance |
|---|---:|
| stop sign via x_objs **and** a dedicated route-level `sign_id` token (`sign_emb`, since removed from the codebase) | **0.738** |
| stop sign via x_objs **only** (current HEAD: shared `class_emb`, no `sign_emb`) | **0.548** |

Investigation of the removed `sign_emb` path (`plant2/PlanT/util/sign_id.py`,
pre-removal `model.py`/`dataset.py` via git history) showed it was an
**episode-level label derived from route metadata** (regex on the route
folder name / an experts-jsonl lookup), injected as one dedicated,
always-present token, completely independent of what the simulator's
per-frame object detector (`x_objs`) actually saw that frame. It is not
"the same information as x_objs but stronger" — it is a different modality:
metadata-injected and immune to the visibility gating x_objs signs go
through. Reintroducing it would not be the model *learning* stop-sign
compliance from perception; it would be closer to answer-injection, which is
exactly what the task asked to avoid (the equivalent of the "just brake N
meters before the sign" post-processing shortcut, at the input side instead
of the output side).

The two most likely, evidence-grounded root causes of the gap, given x_objs
already assigns sign 2.5 its own distinct class id (`class_nums["2.5"]`,
`PlanTVariables.pdd_object_classes`):

1. **Visibility is intermittent.** `dataset.py`'s `_keep_staticish()` only
   keeps a sign-like object in x_objs when the simulator's `affects_ego` flag
   is `True` *and* it's within 30 m — and `affects_ego` is documented (and
   observed) to flicker frame-to-frame even while the sign is squarely in
   range, unlike statics/cars which have no such gate.
2. **Multi-task loss competition.** The ego-speed head shares gradient
   budget with waypoint, path and forecast losses; `speed_class_weights`
   already upweights the near-zero bins 15x by default, so blunt loss
   reweighting alone was already implicitly "tried" as the baseline default
   and evidently insufficient.

Prior unused sweep configs found in the codebase (`train/sweeps/2p5_hyp.yaml`,
`2p5_stopw.yaml` — H1/H2/H1+H2/H5/stopw{5,10,20} in the old naming) were
**never actually run** (no matching checkpoints, logs, or metrics anywhere on
disk) — they were candidates only. The 5 hypotheses below supersede them:
H2 below folds in the previously-unrun "de-emphasize competing losses" idea
under a clearer rationale; the rest are new, targeting the visibility/gating
theory directly since it wasn't tested by anything already in the repo.

## The 5 hypotheses

### H1 — Persistent sign visibility (input-side fix)

Relax `_keep_staticish()`'s `affects_ego` requirement for PDD/stop-sign
objects (keep purely by range, like statics already are) and extend the
detection radius from 30m to 45m, giving more lead-time frames where the
sign is actually present in x_objs.

- **Mechanism**: fixes cause (1) directly — the model can only learn to stop
  at a sign it can see.
- **Implementation**: `plant2/PlanT/dataset.py`, config-gated by
  `model.training.sign_ignore_affects_ego` / `sign_range_m` (default off,
  zero risk to other experiments). Zero new parameters, zero inference-side
  changes needed.
- **Config**: `sign_ignore_affects_ego=True sign_range_m=45`

### H2 — Concentrate capacity on the stop decision (loss-side fix)

Zero out the forecasting and path losses (`forecastLoss_weight=0`,
`waypoints.path_weight=0`) and raise the ego-speed loss weight
(`waypoints.speed_weight=5`), so gradient isn't split across three other
objectives that don't need sign information at all.

- **Mechanism**: fixes cause (2) — even with a visible sign, does the model
  have enough gradient signal earmarked for "should I stop" specifically to
  use it?
- **Implementation**: pure Hydra overrides, zero new code (both loss weights
  were already wired in `lit_module.py`, just never combined and run this
  way).
- **Config**: `model.pre_training.forecastLoss_weight=0 model.waypoints.path_weight=0 model.waypoints.speed_weight=5`

### H3 — Temporal sign memory (input-side, cross-frame)

Add a decayed scalar input token: 1.0 if a relevant sign is in x_objs this
frame; else linearly decayed based on how many of the last K=10 frames ago it
was last seen; 0.0 beyond that window / at episode start.

- **Mechanism**: directly compensates for `affects_ego` flicker (cause 1)
  *without* loosening the filter globally (a different, narrower bet than
  H1) — the model gets to "remember" a recently-seen sign for a few frames
  instead of losing the signal the instant one frame's flag flickers False.
- **Implementation**: `dataset.py` (`_recent_sign_signal`, looks back through
  sibling `boxes/NNNN.json.gz` files) + `model.py` (new `sign_memory_emb`
  Linear token, mirrors the existing `ego_speed_emb` input pattern). At
  **inference** time there's no dump to look back through, so
  `plant2_in_metadrive/plant2_adapter.py` maintains its own rolling
  per-step history buffer (reset every episode) reproducing the same decay
  — validated to match the offline computation exactly, and to detect the
  checkpoint's `sign_memory_emb` weights the same way it already does for
  `input_ego_speed` (so eval reconstructs the trained architecture instead
  of silently dropping these weights under `strict=False`).
- **Config**: `sign_memory_frames=10`

### H4 — Auxiliary sign-presence loss (regularizer, train-only)

Add a small linear probe off the same trailing token the ego-speed head
reads, trained with an auxiliary BCE loss to predict "is a relevant sign
present in x_objs this frame" — ground truth computed straight from the
frame's own boxes (the same information the main path already has, just
supervised explicitly).

- **Mechanism**: a genuinely different failure mode than H1/H3 — even when
  the sign token *is* present, is the shared backbone forced to represent
  "sign-ness" strongly enough for it to reach the speed decision? An
  explicit auxiliary task on the same readout token adds a sharper,
  more direct training signal than the diluted multi-task ego-speed CE
  alone. (This cannot recover information that's genuinely absent from a
  given frame's input — unlike H1/H3, it targets *how well the model uses
  what's already there*, not availability.)
- **Implementation**: `model.py` (`sign_presence_head`, stashed on
  `self._last_aux_sign_logit` rather than added to `forward()`'s public
  4-tuple return — extending that tuple would require updating every caller
  that unpacks `pred_plan`, including the live MetaDrive inference adapter)
  + `lit_module.py` (`_aux_sign_presence_loss`, BCE, weighted by
  `aux_sign_presence_weight`). Train-only: the head is simply unused at
  inference, so **zero** adapter changes needed and zero eval-time risk.
- **Config**: `aux_sign_presence_weight=1.0`

### H5 — Dedicated pooled sign token (architecture fix, perception-derived)

Pool this frame's sign-like object embeddings (mean over any x_objs token
with class id `== 4` "stop_sign" or `>= PDD_OBJECT_CLASS_START`) into one
extra global conditioning token, prepended to the transformer sequence the
same way `route_tok`/`speed_tok` already are. A learned placeholder token
stands in when no sign is present this frame.

- **Mechanism**: the single most evidence-grounded lever available — the
  investigation found the *disentangled, dedicated-token* architecture
  (the old `sign_emb`) was worth +0.19 sign_compliance over folding sign
  identity into the shared per-object `class_emb`, independent of LR/epoch
  choice (confirmed by the classemb refactor regressing performance back to
  baseline). H5 reconstructs that architectural pattern's benefit — a
  channel the model can attend to unconditionally rather than one entry
  among ~30 competing object tokens — but computes it **from x_objs each
  forward pass** instead of injecting a route-metadata label, so it stays
  perception-grounded and still functions correctly at inference on frames
  the model hasn't seen labels for.
- **Implementation**: `model.py` only (`sign_pool_no_sign_emb` placeholder
  parameter + masked mean-pool in `forward()`). Inference-safe: pooling only
  reads `x_objs`/`idxs`, which the MetaDrive adapter already builds normally
  for every step — the adapter only needs the same checkpoint-key detection
  pattern as H3/`input_ego_speed` to reconstruct the flag, no new live
  signal computation required.
- **Config**: `sign_pool_token=True`

## Validation before spending GPU time

All 5 were smoke-tested against a synthetic batch (forward pass shapes,
finite loss, `loss.backward()` reaching every new parameter with a non-zero
gradient) and, for H3/H5, a full checkpoint round-trip test: train a tiny
model under each config, save its `state_dict`, reconstruct a fresh model
using the *same* checkpoint-key-detection logic `plant2_adapter.py` uses at
eval time, `load_state_dict(strict=False)`, and confirm (a) no trained key is
silently dropped and (b) the reloaded model's forward pass is
bit-identical to the original given the same input.

## Training setup (all 5, run in parallel)

Same split (`plant2_stop_pipeline_signfix/plant2_l1_stop_split`, 294/50),
same pretrain checkpoint, same LR (3e-4) and epoch budget (10, matching the
best-performing point of the existing classemb LR/epoch grid — ep10 beat
ep20/30/40, which is why 10 was chosen here rather than a longer budget) as
the 0.548 reference point, so any delta is attributable to the hypothesis
mechanism and not a confound. `checkpoint-addon` names:
`stop_hyp_{h1_persist,h2_reweight,h3_memory,h4_auxloss,h5_pool}` under
`plant2/PlanT/checkpoints_ft/`. Launcher: `launch_hyp5.sh` in this directory.

## Eval

`python eval_pipeline.py --policies plant2 --model-paths plant2:<ckpt> --manifest stop_data/output/ts_test --scenes-root stop_data/scenes --plant2-action-mode pid --jobs 8`
(from `pdd-bench/scripts/per_sign_bench/priority_bench/`, the same harness
used for the 0.738/0.548 reference numbers) against `last_ft` (epoch 9) for
each of the 5 checkpoints. Results recorded in `RESULTS.md` in this
directory once training completes.
