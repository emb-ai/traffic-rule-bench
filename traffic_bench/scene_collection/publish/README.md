# publish/ — Hugging Face dataset `emb-ai/traffic-sign-bench`

```bash
# Pack + upload (needs huggingface-cli login or HF_TOKEN)
python -m traffic_bench.scene_collection publish
```

Download (all signs → `data/scenes/<sign>/`):

```bash
huggingface-cli download emb-ai/traffic-sign-bench \
    --repo-type dataset \
    --local-dir data
```

you can 
```bash
# Staging only (dist/hf-traffic-sign-bench/)
python -m traffic_bench.scene_collection pack --all
```