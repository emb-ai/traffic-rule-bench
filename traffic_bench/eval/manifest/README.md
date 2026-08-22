# manifest/ — scenes + Hydra → `real_manifest.jsonl`

Shared shell only: discover scene dirs under `data/scenes/<sign>/`, write
`repro/`, run Hydra, dispatch to `signs/<family>/expand.py`.

Does **not** contain family expansion (yield vs crosswalk rows). That lives
under `signs/`.

Today: [`../generate_manifest.py`](../generate_manifest.py).
Command: `python -m traffic_bench.eval manifest sign=yield`.
