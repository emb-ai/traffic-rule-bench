# manifest/ — scenes + Hydra → `real_manifest.jsonl`

Shared shell only: discover scene dirs under `data/scenes/<sign>/`, apply
split / `max_total`, write `real_manifest.jsonl` + `repro/`, run Hydra,
dispatch to `signs/<family>/expand.py`.

Does **not** contain family expansion (yield vs crosswalk rows). That lives
under `signs/`.

| File | Role |
| --- | --- |
| `io.py` | `discover_scenes`, split filter, `max_total`, write jsonl + `repro/` |
| `lanes.py` | incoming-lane parse for junction / blocked / dual-path |

Hydra entry is still [`../generate_manifest.py`](../generate_manifest.py)
(`generate_*_manifest` shells + GIF). Command:
`python -m traffic_bench.eval manifest sign=yield`.
