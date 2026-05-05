# TrafficRuleBench: A Benchmark for Evaluating Traffic Rule Compliance in Autonomous Driving

<p align="center">
  <a href="https://arxiv.org/abs/TODO">Paper</a> •
  <a href="https://huggingface.co/datasets/emb-ai/traffic-rule-bench">Dataset (HuggingFace)</a> •
  <a href="https://github.com/emb-ai/traffic-rule-bench">Code</a>
</p>

## Abstract

Autonomous driving planners are typically evaluated using aggregate metrics such as collision rate, route completion, and comfort, which do not explicitly measure compliance with traffic rules. As a result, planners can achieve high benchmark scores while still exhibiting unsafe or illegal behaviors, limiting their applicability to real-world deployment. To address this gap, we introduce **TrafficRuleBench**, a large-scale, rule-centric benchmark for systematic and interpretable evaluation of traffic-rule compliance in autonomous driving. Our framework combines real-map-based simulation for realistic road layouts with rule-targeted procedural scenario generation for scalable and balanced coverage of underrepresented rules. We implement traffic rules corresponding to **45 traffic signs**, each equipped with an automatic rule checker for detecting violations during closed-loop execution. This design yields **15,200 diverse road scenes** and **18 distinct testing scenario types**, enabling controlled evaluation of rule-specific planner behavior. We construct **5,400 testing scenes** and demonstrate that current autonomous driving planners can exhibit poor traffic-rule compliance despite strong performance on standard evaluation metrics. To address this limitation, we transform existing planners into rule-compliant trajectory experts via explicit traffic-sign constraints, enabling scalable generation of high-quality oracle trajectories for fine-tuning. Code and data are publicly available at [github.com/emb-ai/traffic-rule-bench](https://github.com/emb-ai/traffic-rule-bench) and [huggingface.co/datasets/emb-ai/traffic-rule-bench](https://huggingface.co/datasets/emb-ai/traffic-rule-bench).

---

## Repository Structure

```
sdc/
└── pdd-bench/
    ├── envs/                          # MetaDrive / SUMO environments with sign checkers
    │   ├── traffic_sign_env.py        # TrafficSignEnv (synthetic road layouts)
    │   ├── sumo_env.py                # TrafficSignSumoEnv (real road layouts)
    ├── traffic_signs/                 # 45 sign classes + automatic rule checkers
    ├── agents/
    │   ├── policies/                  # baseline policies
    │   │   ├── comprehensive_rule_expert.py   # IDM + sign-compliance overlay
    │   │   ├── rule_compliant_expert.py       # PPO + sign-compliance overlay
    │   │   ├── carl_sign_compliant.py         # CaRL + sign-compliance overlay
    │   │   └── plant2_sign_compliant.py       # PlanT2 + sign-compliance overlay
    │   ├── carl_in_metadrive/         # CaRL adapter
    │   └── plant2_in_metadrive/       # PlanT2 adapter
    ├── scenes/                        # real SUMO road layouts (.net.xml)
    └── scripts/per_sign_bench/        # benchmark pipeline (scenes → eval → metrics)
        ├── per_sign_benchmark.py      # scene generation orchestrator
        ├── run_benchmark_mini.py      # eval runner (all baselines)
        ├── factorized_space/          #  synthetic scene generation
        ├── citymap_space/             # synthetic scene generation
        ├── sumo_space/                # real SUMO scene enumeration
        ├── build_episode_metrics_csv.py
        ├── aggregate_episode_metrics.py
        ├── build_oracle_baseline.py
        ├── generate_cumulative_markdown_report.py
        └── run_full_metrics_pipeline.sh
```

---

## Setup

### Requirements

- Python 3.10+
- MetaDrive (submodule: `sdc/metadrive/`)
- SUMO (for real-map scenes)

### Installation

```bash
git clone --recurse-submodules https://github.com/emb-ai/traffic-rule-bench
cd traffic-rule-bench
git submodule update --init --recursive

conda create --name metadrive_signs python=3.10
conda activate metadrive_signs

pip install -e sdc/metadrive
pip install eclipse-sumo sumolib pyproj stable_baselines3
pip install pandas "geopandas<1.0" gym timm
pip install -e sdc/pdd-bench
```

### PlanT2 environment (for `plant2` / `plant2_rule` baselines)

```bash
cd sdc/plant2
conda env update -f environment.yml --prune
conda activate plant2
pip install gymnasium panda3d panda3d-gltf progressbar pygame sumolib einops
pip install -e ../metadrive
```

### Checkpoints

Download model checkpoints from HuggingFace and place them in `sdc/pdd-bench/checkpoints/`:

```bash
huggingface-cli download emb-ai/traffic-rule-bench --local-dir sdc/pdd-bench/checkpoints
```

---

## Evaluation

Run baselines on the generated scenes. Each baseline produces `episodes_<policy>.jsonl` and per-episode `replay.json` sidecars.

### Single baseline

```bash
cd sdc/pdd-bench/scripts/per_sign_bench

# single manifest file:
python run_benchmark_mini.py \
    --policy      idm \
    --run-name    idm_default \
    --manifest    /path/to/var_0.jsonl \
    --emit-replay-sidecar    # required for metrics

# or a directory with per-sign subdirs 
python run_benchmark_mini.py \
    --policy           idm \
    --run-name         idm_default \
    --benchmark-output /path/to/benchmark_output \
    --emit-replay-sidecar    # required for metrics
```

### Yield-sign scenarios

`yield_run_benchmark_mini_plant2.py` is an  eval runner for yield-sign (2.4) scenes.

```bash
cd sdc/pdd-bench/scripts/per_sign_bench

python yield_run_benchmark_mini_plant2.py \
    --policy      idm \
    --run-name    idm_yield \
    --manifest    /path/to/yield_manifest.jsonl \
    --sign-type   2.4 \
    --emit-replay-sidecar \
    --save-gifs                    # optional: record top-down GIFs
```

### All 17 baselines (multi-GPU cluster)

```bash
export CARL_CKPT=/path/to/carl.ckpt
export PLANT2_CKPT=/path/to/plant2.ckpt
export PLANT2_CKPT_FINETUNE=/path/to/plant2_finetune.ckpt

GPUS=0,1,2,3,4,5,6,7 \
BY_VAR_IDX_DIR=/path/to/scenes_by_var \
ROOT=eval_out \
bash run_17_baselines.sh    # run from repo root
```

### Available baselines

| Baseline | `--policy` | Notes |
|---|---|---|
| IDM (5 ego variants) | `idm` | + `--ego-variant default/s1..s4` |
| PPO | `ppo_expert` | |
| CaRL | `carl` | `--model-path` required |
| PlanT2 | `plant2` | `--model-path` required |
| IDM + rule overlay | `comprehensive_rule_expert` | + `--ego-variant` |
| PPO + rule overlay | `rule_compliant` | |
| CaRL + rule overlay | `carl_rule` | `--model-path` required |
| PlanT2 + rule overlay | `plant2_rule` | `--model-path` required |

---

## Computing Metrics

### Single-run metrics (quickest)

```bash
bash run_metrics_single_run.sh \
    --run-dir eval_out/runs/var_0/idm_default \
    --out-dir eval_out/metrics_idm_default \
    --policy  idm_default
```

Outputs:
- `metrics_per_episode.csv` — episode-level table
- `aggregations/agg_per_baseline.csv` — per-baseline summary
- `reports/report_cumulative.md` — markdown report table
- `reports/report_cumulative_categories.md` — per-category breakdown

### Full multi-baseline pipeline

```bash
# All steps in one command:
ROOT=eval_out bash run_full_metrics_pipeline.sh

# Skip consolidation if replay jsonl files already exist:
SKIP_CONSOLIDATE=1 ROOT=eval_out bash run_full_metrics_pipeline.sh
```

The pipeline runs:
1. `consolidate_replays.py` — merges `replay.json` sidecars → `<baseline>_replays.jsonl`
2. `build_episode_metrics_csv.py` — builds `metrics_per_episode.csv`
3. `build_oracle_baseline.py` — adds `oracle_rule` synthetic baseline
4. `aggregate_episode_metrics.py` — aggregates to per-baseline/sign/var CSVs

### Oracle baseline

```bash
python3 build_oracle_baseline.py \
    --csv eval_out/metrics_per_episode.csv
```
---

## Metrics

| Metric | Description |
|---|---|
| `target_compliant_event` | Ego obeyed the target sign within its zone (primary rule metric) |
| `arrived_dest` | Reached the destination |
| `route_completion` | Fraction of route covered |
| `total_violations` | Total traffic rule violations (all signs) |
| `success` | Arrived AND no collision AND compliant |
| `comfort` | Jerk / lateral acceleration composite |

---

## Citation

```bibtex
@article{trafficrulebench2025,
  title   = {TrafficRuleBench: A Benchmark for Evaluating Traffic Rule Compliance in Autonomous Driving},
  author  = {},
  journal = {arXiv},
  year    = {2025},
  url     = {https://github.com/emb-ai/traffic-rule-bench}
}
```
