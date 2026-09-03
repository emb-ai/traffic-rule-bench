# analysis/

```text
analysis/
  inventory/          # harvest crop counts
  overlap/            # train/test place reuse + allocation verify
  assign_verify.py    # post-assign leak / counts / topology / reuse
  run.py
```

```bash
python -m traffic_bench.scene_collection analysis inventory
python -m traffic_bench.scene_collection analysis overlap          # figures + README + allocation_verify.*
python -m traffic_bench.scene_collection analysis assign_verify    # verify only → overlap/
```

| Path | Role |
| --- | --- |
| `overlap/README.md` | place-overlap narrative + numbers |
| `overlap/allocation_verify.md` | assign policy verification |
| `overlap/figures/` | PNGs (21) |
| `overlap/summary.json` | machine-readable overlap metrics |
