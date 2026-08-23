# run/ — closed-loop episodes

Hydra entry: [`main.py`](main.py) (`configs/run.yaml`).

```bash
python -m traffic_bench.eval run policy=idm sign=yield
python -m traffic_bench.eval run policies=all sign=yield
python -m traffic_bench.eval run policies=[idm,plant2] manifest=data/runs/yield/debug
```

`policy=` runs one policy. `policies=[…]` or `policies=all` is the same runner
in a loop ([`policies.py`](policies.py)), then `metrics`. Without `manifest=`,
`sign=` reads `data/runs/<sign>/test/`. IDM-family policies in the list expand
to ego variants `default,s1–s4`.

| File | Role |
| --- | --- |
| `main.py` | Hydra: load rows, call `run_episodes` |
| `policies.py` | loop policies, then csv / aggregate / report |
| `episode.py` | step loop, write episode JSON |
| `env.py` | build env, apply spawn / dest |
| `policy.py` | load IDM / PPO / CaRL / PlanT2 |
| `gif.py` | top-down film size |
| `score.py` | TTC, smoothness, `aggregate_results` |
| `place.py` | plate dispatch → `signs/<group>/place.py` |

Aux / horizon / spawn come from the saved manifest `config.yaml` + row.
