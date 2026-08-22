# engine/ — shared eval runtime (no sign rules)

Package inits stay thin so importing `engine.map.lane_keys` does not load
MetaDrive.

| Drawer | Role |
| --- | --- |
| [`map/`](map/) | what the net looks like (SUMO keys, T/X arms, roundabout topology) |
| [`traffic/`](traffic/) | who drives (nuPlan / IDM / density) |
| [`spawn/`](spawn/) | shared spawn types; dispatch to `signs/<group>/spawn.py` |
| [`expand/`](expand/) | shared row-axis types, not per-sign builders |
| [`sim/`](sim/) | MetaDrive glue, checkpoints, HUD/GIF patches |

Sign-specific runtime lives with the sign: `signs/dual_path/nav.py`,
`signs/roundabout/aux.py`. Harvest crop lives in
`scene_collection/collect/lib/junction_crop.py`.
