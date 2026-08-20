# bench/

Library helpers — **no CLI**. Invoked indirectly via `run_benchmark.py` 

- `_paths.py` – sys.path bootstrap
- `env_builders.py` – build env from a manifest row + sign placement
- `policy_factory.py` – resolve/load ego policy + IDM variant
- `sign_eval.py` – violations, zones, crash attribution
- `episode_metrics.py` – per-episode metric helpers
- `manifest_io.py` – manifest load / resume keys
- `util.py` – small shared helpers
