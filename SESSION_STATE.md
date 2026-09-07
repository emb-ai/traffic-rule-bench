# Session state — PlanT2 sign compliance (stop 2.5 + detour 4.2.x)

## Environment (use this, not zinkovich-plant2)
- python: `/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python`
  (clone of zinkovich-plant2 with metadrive editable-installed from THIS repo's
  submodule and transformers pinned 4.49.0/hub 0.29.3/tokenizers 0.21.1)
- always pass `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to training (proxy flaky)
- eval must set `PLANT2_SIGN_RANGE_M=90` to match how the data was dumped

## Three data-generation bugs found and fixed (all were in the dump, not training)
1. `route_original` (model INPUT) was byte-identical to `route` (TARGET) —
   copying the input was exactly optimal. Fixed offline by
   `scripts/plant2_ft_pipeline/data/fix_route_target.py` (rebuilds target from
   recorded `pos_global`/`theta`; symlinks heavy files, never touches shared dumps).
   The dumper still writes them equal, so this tool must be re-run after any dump.
2. Sign boxes capped at 30 m in `collect_boxes`, but compliance needs the lane
   change finished at zone entry (30 m before obstacle). Now `PLANT2_SIGN_RANGE_M`
   (default 30 unchanged).
3. Augmentation was a no-op: dump wrote `augmentation_translation/rotation = 0.0`
   and saved the "augmented" BEV as a copy. `aug_sample` itself was correct.
   Now `PLANT2_AUG_TRANSLATION_M` / `PLANT2_AUG_ROTATION_DEG` (default 0), with
   `render_bev_plant2(lateral_offset_m, heading_offset_rad)` added in metadrive.

## Metric caveats (do not report compliance alone)
- `DetourSign._is_violating` only wants the vehicle in the adjacent lane inside
  the zone; a car that drifts sideways and stalls scores ~1.0. Always report
  distance + past-obstacle + out_of_road.
- On the detour catalog `success`/`arrived_dest` is 0 even for the privileged
  expert (route_completion median 0%). Use past-obstacle / out_of_road instead.
- `best`-by-`val/loss_all` is a bad selector (picked epoch 0 = 0.191 vs last
  0.601). Evaluate both `best` and `last_ft`.
- `val/loss_path` is not comparable across splits (old split leaked scenes
  between train and val; new splits are by scene).

## Detour results (specialist models, 122-scene catalog, 366 runs)
| stage | compliance | dist | past obstacle | out_of_road |
|---|---|---|---|---|
| as found | 0.014 | 65 m | 0% | - |
| + target fix | 0.055 | 73 m | 24% | - |
| + sign 90 m | 0.626 | 68.6 m | 31% | - |
| H32 ovsWIDE lr3e4 (last) | **0.814** | 48.7 m | 19% | 75% |
| H34 ovs lr1e4 14ep (best) | 0.333 | 65.8 m | 34% | **27%** |
Clear trade-off: higher compliance costs driving quality. LR 1e-4 halves
out_of_road vs 3e-4. Augmentation (bug 3) is the intended fix and was NOT yet
active in any of these runs.

## Stop-sign reference
`stop_classemb_lr3e4_ep30/best_011` = 0.500 compliance after the BEV fix
(was 0.452 measured with a wrong-scale BEV).

## RUNNING NOW: joint model (stop + detour), first data with all 3 bugs fixed
- data: `/tmp/joint_split_fx` (1544 train / 261 val routes, 567 scenes,
  446k frames; stop 314+30, detour 1230+231; sign visible 80 m / 60 m;
  leak check 0/1010; augmentation 100% of frames)
- source dumps: `/tmp/joint_dump_stop` (344), `/tmp/joint_dump_detour` (1461)
- runs (checkpoint-addon `joint_*`, logs in `plant2_detour_pipeline/logs/`):
  j1 aug lr1e4 ep14 (gpu0) | j2 +ovs40-70 lr1e4 ep14 (gpu1)
  j3 aug lr3e4 ep14 (gpu2) | j4 +ovs lr3e4 ep7 (gpu3)
- next: eval each on BOTH benchmarks separately, compare against the
  specialists above; if stop degrades, oversample the stop share (it is only
  20% of the joint data).

## Gotchas hit repeatedly
- zsh arrays are 1-indexed; `${a[0]}` is empty.
- pass checkpoint paths to `run_eval.sh` as ABSOLUTE (subprocesses run from a
  different cwd; the policy swallows the load error per frame and the car just
  sits still, which looks like perfect compliance).
- a `pgrep -f <pattern>` watcher matches its own command line and never exits.
- GPUs are shared with other users; check `nvidia-smi` before launching.
