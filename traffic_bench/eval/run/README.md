# `run/` — closed-loop episodes

Runs evaluation scenarios from a manifest in a closed-loop environment.

Hydra entry point: [`main.py`](main.py) (`configs/run.yaml`).

## Usage

```bash
# One policy
python -m traffic_bench.eval run policy=idm sign=yield

# Multiple policies
python -m traffic_bench.eval run \
    policies=[idm,plant2] \
    sign=yield

# Use a specific manifest
python -m traffic_bench.eval run \
    policies=[idm,plant2] \
    manifest=data/runs/yield/debug