# sign_scenes/prepare/ — map surgery after materialize

Runs yaml `prepare:` hooks. Today only **crosswalk** (PDD 5.19).
`materialize` already calls this after placing a sign that has `prepare:` —
use the CLI to re-run.

```bash
python -m traffic_bench.scene_collection prepare --sign crosswalk
# or: python -m traffic_bench.scene_collection prepare --all
```

`run.py` dispatches by PDD. Hooks live in subfolders; see [`crosswalk/`](crosswalk/README.md).
