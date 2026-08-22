# collect/segments/ — corridor crops

Incoming edges cropped so the scene **ends before the junction** (10 m margin). `segment_type` (straight / curved) is a tag in `meta.json`, not a folder.

- `metrics.py` — length, straightness, `vehicle_lane_indices`, `pass_right_ok` / `pass_left_ok`
- `crop.py` — write `maps/crops/segment/<scene_id>/` (`--workers N` for parallel netconvert)

Used by speed signs, 4.2.x (obstacle lane is chosen in eval), and 5.19 (zebra added later by `prepare`).
