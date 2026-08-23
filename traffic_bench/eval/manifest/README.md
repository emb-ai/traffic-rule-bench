# manifest/ — scenes → `real_manifest.jsonl`

1. resolve `sign=` via `sign_registry`
2. dispatch `signs.<group>.expand.generate`
3. write `real_manifest.jsonl` + `repro/`
4. optional GIFs via `run` in process


| File       | Role                                |
| ---------- | ----------------------------------- |
| `run.py`   | Hydra main + GIF                    |
| `types.py` | resolved knobs passed to `generate` |
| `io.py`    | discover scenes, split, write jsonl |
| `lanes.py` | SUMO spawn-lane parse               |


Junction and roundabout share `signs/junction/expand.generate`.