# `publish/`

Publishes the official TrafficSignBench scenes.

```bash
# Pack + upload (needs huggingface-cli login or HF_TOKEN)
python -m traffic_bench.scene_collection publish
```

*or you can:* 

```bash
# Staging only (dist/hf-traffic-sign-bench/)
python -m traffic_bench.scene_collection pack --all
```

