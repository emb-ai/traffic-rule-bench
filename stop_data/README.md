# stop_data — vendored inputs for the STOP (2.5) pipeline

Everything `plant2_stop_pipeline_signfix/run_signfix_pipeline.sh` reads that used to
come from the zinkovich tree. Vendored 2026-08-20 because that tree was being
restructured underneath the pipeline and eval was failing.

Regenerate with `plant2_stop_pipeline_signfix/vendor_stop_data.py` (read-only with
respect to the zinkovich tree, idempotent). Validate with
`plant2_stop_pipeline_signfix/check_stop_data.py`, which the pipeline also runs at
startup.

## Layout

| path | what | size |
|---|---|---|
| `scenes/junc_*/` | 87 SUMO maps (`map.net.xml`, `center.json`, `meta.json`, preview png) — the 16 test junctions plus the 71 train junctions | 6.3 MB |
| `output/ts_test/` | test manifest (`real_manifest.jsonl`, 50 rows) and its `manifest.json` / `real_manifest_summary.json` / `config.yaml` sidecars | 192 KB |
| `trajectories/debug_train_400/` | the 344 top-1 expert replays (`replay.pkl` + `replay.json`) and `experts/experts_scene_uid_top1.jsonl` | 721 MB |
| `checkpoints/plant2_pretrain/` | `epoch=029_final_3.ckpt`, the pretrain checkpoint `stage_train` resumes from | 428 MB |

`manifest.json` is load-bearing: `load_manifest_config()` reads it from the
manifest's own directory, and without it `spawn_distance_before_end` silently
falls back to 12.0, which is a different scenario from the one the manifest
describes.

## Provenance

Sources, all under `.../zinkovich/zinkovich/traffic-rule-bench/pdd-bench`:

- manifest + sidecars, expert index, replays: `sign_bench/data/stop/...` (the
  post-move home of the old `scripts/per_sign_bench/priority_bench/data/stop`)
- maps: `moscow_scenes/scenes/{T,X}/junc_*`
- checkpoint: `checkpoints/plant2_pretrain/`

`sign_bench/data/stop/scenes/junc_*` are symlinks into
`scripts/per_sign_bench/moscow_scenes/scenes/{T,X}/`, which the move deleted. All
87 scenes the pipeline needs pointed at `moscow_scenes` (none at the older
`moscow_junctions` target), and `moscow_scenes` was moved up one level with its
internal layout unchanged, so each dangling link was resolved by copying the real
directory from `pdd-bench/moscow_scenes/scenes/{T,X}/<scene_id>`.

## Rewritten paths

`experts_scene_uid_top1.jsonl` and the 344 `replay.json` sidecars carried
absolute `pkl_path` / `sidecar_path` values pointing into the deleted tree. Those
are rewritten to this directory; the untouched original index is kept beside it as
`experts_scene_uid_top1.jsonl.zink_orig` so the relocation stays auditable. It is
the only file here that still names the zinkovich tree.

Each `replay.pkl` also embeds one absolute `map.net.xml` path from when it was
recorded. It is inert: `expert_replay_for_plant2.py` reads only the recorded NPC
frames out of the pkl and rebuilds the env from `--scenes-root` plus the
manifest row's relative `net_path`.

## Integrity checks that passed

- `output/ts_test/manifest.json` and `real_manifest_summary.json` are
  byte-identical to `plant2_stop_pipeline_signfix/gif_input/`, i.e. the copies the
  successful `eval_test/full` run loaded.
- `output/ts_test/real_manifest.jsonl` is byte-identical to the zinkovich source,
  and equals `eval_test/full/input_manifest.jsonl` on all 50 rows and all common
  keys (that file is the same manifest plus `aux_convoy_size_max` /
  `aux_lanes_occupied_max`, which `eval_pipeline.py` injects from `manifest.json`).
- Zero symlinks and zero dangling links anywhere under `stop_data/`.
- Every `scene_id` and every relative `net_path` in both manifests resolves to a
  real, readable, non-symlink file.
- All 688 replay `pkl_path` / `sidecar_path` entries resolve inside `stop_data/`.
