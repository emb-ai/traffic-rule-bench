# `run/` — closed-loop episodes

Runs evaluation scenarios from a manifest in a closed-loop environment.

Hydra entry point: [`main.py`](main.py) (`configs/run.yaml`).

## Usage

```bash
# One policy
python -m traffic_bench.eval run policy=idm sign=yield

# Multiple policies (CPU jobs=4, NN jobs_nn=1, leave physical GPU 0 free)
python -m traffic_bench.eval run \
    policies=[idm,idm_rule,ppo_rule,carl,carl_rule] \
    sign=all \
    jobs=4 \
    jobs_nn=1 \
    cuda_devices=[1,2,3,4,5,6,7]

# Use a specific manifest
python -m traffic_bench.eval run \
    policies=[idm,plant2] \
    manifest=data/runs/yield/debug