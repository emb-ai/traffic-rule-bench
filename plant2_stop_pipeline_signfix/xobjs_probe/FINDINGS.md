# Is the 2.5 stop sign in `x_objs` at eval time?

**Yes — on 100% of approach frames, in all 19 violating episodes.** The hypothesis
left open by `../eval_test/gifs_violations/ANALYSIS.md` §3.3 is disproved. Nothing
was rebuilt.

Instrumented rollouts: 19 violators + 9 compliant, same checkpoint
(`last_ft_stop_signfix_lr3e4_ep20_1.ckpt`), `--plant2-action-mode pid`, one process
per episode. All 28 reproduced the original `metrics_per_episode.csv` verdict and
`final_step` exactly (0 mismatches), so the instrumentation is behaviour-neutral.

---

## 1. Per-frame evidence

Full table: `xobjs_report.json`; raw per-frame traces: `logs/<scene_uid>.xobjs.jsonl`.


| group     | n   | 2.5 token on every approach frame | frames where sign was in range but absent from `x_objs` |
| --------- | --- | --------------------------------- | ------------------------------------------------------- |
| violators | 19  | **19 / 19**                       | **0**                                                   |
| compliant | 9   | 9 / 9                             | 0                                                       |


- **Detection distance.** The token appears on the very first frame of every episode,
at 13.6–18.0 m, and tracks the ground-truth sign pose to 0.01 m for the whole
approach. The ego spawns 15 m before the lane end (`spawn_distance_before_end=15.0`),
so the sign is *never* further than 18 m — the 30 m cutoff cannot fire in this
benchmark, in either the dump or the eval.
- **Frame counts.** 12–159 consecutive approach frames per episode carry the token
(`approach_frames` = sign ahead and within 12 m). The token is continuous: no gaps,
no late onset, no truncation.
- **The** `in_xobjs < in_mgr` **gap in the table is not a loss.** It is the post-junction
tail: once the ego has driven >30 m past the sign the collector correctly drops it.
Every one of those frames has the sign *behind* the ego. Counting only frames where
the sign was within 30 m, the two columns agree exactly (`dropped = 0` everywhere).
- **Closest approach is a lateral effect, not an absence.** Lane-2 spawns at
`junc_cluster_1933909972_2068371768` / `junc_cluster_307620151_59642360` never get
closer than ~10 m because the sign sits on the far right of a multi-lane approach;
the token is present with a large `y`, which matches `ANALYSIS.md` §3.2 on lane 2
violating 4/5.



## 2. Eval path vs training dump: sign handling agrees

The eval-time observation is built live by
`metadrive/metadrive/policy/metadrive_obs_to_plant2.py::metadrive_obs_to_plant2_batch`,
called from `pdd-bench/agents/plant2_in_metadrive/plant2_adapter.py::get_action`.
It calls the *same* `bench.plant2_frames.collect_boxes` the train dump uses, so:


| aspect                 | train dump (`PlanTDataset.__getitem__`)             | eval (`boxes_to_objects_list`)  | agree? |
| ---------------------- | --------------------------------------------------- | ------------------------------- | ------ |
| collector              | `collect_boxes`                                     | `collect_boxes`                 | yes    |
| class → token          | `PlanTVariables.class_nums["2.5"]` = **12.0**       | same                            | yes    |
| coordinate frame       | CARLA, x=forward, y=right                           | same (`_ego_xy` negates y once) | yes    |
| sign radius gate       | 30 m (`pos_x²+pos_y² > 30²`)                        | 30 m (`x*x+y*y > 900.0`)        | yes    |
| `affects_ego` required | yes                                                 | yes (always `True` for signs)   | yes    |
| row layout             | `[type, x, y, yaw°, 0.0, extent[1]*2, extent[0]*2]` | identical                       | yes    |
| normalisation          | none (raw m / deg / km·h⁻¹)                         | none                            | yes    |
| allowlist              | `PLANT2_DUMP_SIGN_CLASSES` (default `2.5`)          | same module-level constant      | yes    |


Two differences exist, neither of which touches the stop sign:

1. **Token order.** The eval sorts all tokens by distance; the dump emits cars first,
  then static/sign tokens. HFLM has no positional embedding over object tokens
   (`tok_emb` is a per-class `Linear`), so the object set is permutation-invariant and
   this is a no-op.
2. **Car ellipse gate.** The dump calls `collect_boxes` with the defaults
  (`max_distance=50`, `range_factor_front=2`, matching `PlanT.yaml` `range: 50` /
   `range_factor_front: 2`); the adapter passes `75.0` / `16.0`. This admits more
   *vehicle* tokens at eval than training. Signs bypass both parameters (they use the
   fixed 30 m gate), and with `traffic_density=0.0` these episodes only ever have
   4 objects total, so it cannot explain the violations. Worth aligning anyway.

Sign *geometry* also matches between train and eval. Sampling 12 training routes from
`plant2_l1_stop_train/data`: first-frame sign distance mean 16.4 m (min 14.9, max 22.1),
range 3.1–30.0 m, token present on 77.5% of frames (the missing frames are the same
post-junction tail). Eval first-frame distance is 15.4 m. The model was trained seeing
the sign from exactly the distances it sees at eval.

## 3. So what does differ

Since the input is identical, the difference is the model's response to it. The clean discriminator is *where the ego first comes to rest while the sign is still ahead*:


| group     | halted before reaching the sign | mean halt position           |
| --------- | ------------------------------- | ---------------------------- |
| compliant | 6 / 9                           | **12.2 m** ahead of the sign |
| violators | 6 / 19                          | **5.0 m** ahead of the sign  |


Compliant episodes brake essentially off the spawn — 4 of them halt at ~14.7 m, i.e.
within a frame or two of the first observation — which is the same latch that
`ANALYSIS.md` §3.4 measured from the action trace. Violators either never stop at all
(13/19) or stop far too late, already inside the zone. The model's own predicted speed
tells the same story: `desired_speed` collapses below 0.05 m/s on 6–152 frames in the
compliant episodes that stop cleanly, and on 0 frames in 15 of the 19 violators.

## 4. Conclusion

The stop-sign token reaches the model on every violating frame, at the right position,
in the right frame of reference, from the right distance, through the same code path
and the same filters as training. The residual 0.548 compliance is therefore the
out-of-distribution junction gap identified in `ANALYSIS.md` §3.2 and §3.5 — violation
is locked per junction and none of the 15 test junctions is among the 71 training
junctions — not a data-plumbing defect. The levers remain the ones `ANALYSIS.md` §4
lists: more training junctions, and `best_003` vs `last_ft` checkpoint selection.

## 5. Re-enabling the instrumentation

Off by default. Set one env var:

```
PLANT2_XOBJS_LOG_PATH=/path/to/trace.jsonl
```

Hook: `_maybe_log_xobjs` in
`pdd-bench/agents/plant2_in_metadrive/plant2_adapter.py`, called at the end of
`PlanT2MetaDriveAdapter.get_action`. One JSON line per inference step:
frame index, ego speed, object token types in `x_objs`, the 2.5 token's ego-frame
`(x, y, d, yaw)`, the ground-truth sign poses read independently from
`engine.traffic_sign_manager` (so a missing token can be distinguished from a missing
sign), the model's `desired_speed`, and the commanded `[steer, throttle]`.
`PLANT2_XOBJS_LOG_STRICT=1` makes logging failures print instead of staying silent.

Drivers:

- `scripts/plant2_ft_pipeline/tools/run_xobjs_probe.py` — selects episodes, vendors
maps, runs `priority_bench/run_benchmark.py` once per episode with the log path set.
- `scripts/plant2_ft_pipeline/tools/report_xobjs_probe.py` — summarises the traces.

Reproduce:

```
python scripts/plant2_ft_pipeline/tools/run_xobjs_probe.py \
    --work plant2_stop_pipeline_signfix/xobjs_probe --group both
python scripts/plant2_ft_pipeline/tools/report_xobjs_probe.py \
    --work plant2_stop_pipeline_signfix/xobjs_probe
```

