# TrafficRuleBench: A Benchmark for Evaluating Traffic Rule Compliance in Autonomous Driving

<p align="center">
  <a href="https://arxiv.org/abs/TODO"><img src="https://img.shields.io/badge/arXiv-TODO-b31b1b.svg" alt="ArXiv"></a>
  <a href="https://huggingface.co/datasets/emb-ai/traffic-rule-bench"><img src="https://img.shields.io/badge/🤗%20Dataset-emb--ai%2Ftraffic--rule--bench-yellow" alt="Dataset"></a>
  <a href="https://huggingface.co/emb-ai/traffic-rule-bench-models"><img src="https://img.shields.io/badge/🤗%20Models-emb--ai%2Ftraffic--rule--bench--models-yellow" alt="Models"></a>
</p>

## Abstract

Autonomous driving planners are typically evaluated using aggregate metrics such as collision rate, route completion, and comfort, which do not explicitly measure compliance with traffic rules. As a result, planners can achieve high benchmark scores while still exhibiting unsafe or illegal behaviors, limiting their applicability to real-world deployment. To address this gap, we introduce **TrafficRuleBench**, a large-scale, rule-centric benchmark for systematic and interpretable evaluation of traffic-rule compliance in autonomous driving. Our framework combines real-map-based simulation for realistic road layouts with rule-targeted procedural scenario generation for scalable and balanced coverage of underrepresented rules. We implement traffic rules corresponding to **45 traffic signs**, each equipped with an automatic rule checker for detecting violations during closed-loop execution. This design yields **15,200 diverse road scenes** and **18 distinct testing scenario types**, enabling controlled evaluation of rule-specific planner behavior. We construct **5,400 testing scenes** and demonstrate that current autonomous driving planners can exhibit poor traffic-rule compliance despite strong performance on standard evaluation metrics. To address this limitation, we transform existing planners into rule-compliant trajectory experts via explicit traffic-sign constraints, enabling scalable generation of high-quality oracle trajectories for fine-tuning. Code and data are publicly available at [github.com/emb-ai/traffic-rule-bench](https://github.com/emb-ai/traffic-rule-bench) and [huggingface.co/datasets/emb-ai/traffic-rule-bench](https://huggingface.co/datasets/emb-ai/traffic-rule-bench).

## 🤗 Supported Baselines

| Family            | Baseline                    | `--policy`                    | Notes                              |
|-------------------|-----------------------------|-------------------------------|------------------------------------|
| Base              | IDM (5 ego variants)        | `idm`                         | + `--ego-variant default/s1..s4`   |
| Base              | PPO                         | `ppo_expert`                  |                                    |
| Base              | CaRL                        | `carl`                        | `--model-path` required            |
| Base              | PlanT2                      | `plant2`                      | `--model-path` required            |
| Fine-tuned        | PlanT2 fine-tuned on TrafficRuleBench | `plant2`             | `--model-path` to fine-tuned `.pt` |
| Rule-augmented    | IDM + rule overlay          | `comprehensive_rule_expert`   | + `--ego-variant`                  |
| Rule-augmented    | PPO + rule overlay          | `rule_compliant`              |                                    |
| Rule-augmented    | CaRL + rule overlay         | `carl_rule`                   | `--model-path` required            |
| Rule-augmented    | PlanT2 + rule overlay       | `plant2_rule`                 | `--model-path` required            |

**Submodules:**
- [MetaDrive](https://github.com/emb-ai/metadrive) — simulation backend with sign-aware extensions
- [PlanT2](https://github.com/emb-ai/plant2) — PlanT2 policy and training pipeline
- [CaRL](https://github.com/autonomousvision/CaRL) — CaRL policy

## 🚀 Quick start

### Environment Set Up

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/emb-ai/traffic-rule-bench
   cd traffic-rule-bench
   git submodule update --init --recursive
   ```

2. Create the main conda env:
   ```bash
   conda create --name metadrive_signs python=3.10
   conda activate metadrive_signs

   pip install -e metadrive
   pip install eclipse-sumo sumolib pyproj stable_baselines3
   pip install pandas "geopandas<1.0" gym timm
   pip install -e pdd-bench
   ```

3. (Optional) PlanT2 env for `plant2` / `plant2_rule` baselines:
   ```bash
   cd plant2
   conda env update -f environment.yml --prune
   conda activate plant2
   pip install gymnasium panda3d panda3d-gltf progressbar pygame sumolib einops
   pip install -e ../metadrive
   cd ..
   ```

### Checkpoints

TrafficRuleBench uses three checkpoint sets. **Base** CaRL and PlanT2 weights come from the original authors' releases; the **fine-tuned** PlanT2 weights come from our HuggingFace model hub.

| Model                   | Source                                                                                                  | Default location                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| CaRL (base)             | [autonomousvision/CaRL](https://github.com/autonomousvision/CaRL) — see their release/checkpoints      | `pdd-bench/checkpoints/CaRL/model_best.pth`            |
| PlanT2 (base, pretrain) | [emb-ai/plant2](https://github.com/emb-ai/plant2) / [autonomousvision/plant](https://github.com/autonomousvision/plant) | `pdd-bench/checkpoints/plant2/epoch%3D029_final_3.ckpt` |
| PlanT2 (fine-tuned)     | [🤗 emb-ai/traffic-rule-bench-models](https://huggingface.co/emb-ai/traffic-rule-bench-models)         | `pdd-bench/checkpoints/plant2/plant2_supervised_2nd_final.pt` |

Download the fine-tuned PlanT2 checkpoint:

```bash
pip install huggingface_hub
huggingface-cli download emb-ai/traffic-rule-bench-models --local-dir pdd-bench/checkpoints
```

For CaRL and the PlanT2 pretrain weights, follow the download instructions in each respective upstream repository and place the resulting files under `pdd-bench/checkpoints/`.

### Scenes & test manifests

SUMO road layouts (`.net.xml`) and per-sign test manifests (`.jsonl`) live in the HuggingFace dataset [emb-ai/traffic-rule-bench](https://huggingface.co/datasets/emb-ai/traffic-rule-bench). Download both into `pdd-bench/`:

```bash
huggingface-cli download emb-ai/traffic-rule-bench \
    --repo-type dataset \
    --local-dir pdd-bench
```

This produces:

```
pdd-bench/
├── scenes/{sign_code}/sign_NNNNNN/*.net.xml
└── test/{sign_code}/*.jsonl
```

Pass a manifest to the runner via `--manifest pdd-bench/test/<sign>/<file>.jsonl`. Scripts default to `pdd-bench/scenes` for `--scenes-root`.

### Run Evaluation

Each manifest in `pdd-bench/test/<sign>/<file>.jsonl` is a self-contained set of scenes for one sign. Run a baseline against any manifest with `run_benchmark_mini.py`. Each run produces `episodes_<policy>.jsonl` plus per-episode `replay.json` sidecars (needed for the metrics pipeline).

#### 1. One baseline on one sign

```bash
cd pdd-bench/scripts/per_sign_bench

python run_benchmark_mini.py \
    --policy   idm \
    --run-name idm_2_5 \
    --manifest ../../test/2_5/real_manifest.jsonl \
    --emit-replay-sidecar
```

Models that require checkpoints (`carl`, `plant2`, `*_rule`) need `--model-path`:

```bash
python run_benchmark_mini.py \
    --policy     plant2 \
    --run-name   plant2_2_5 \
    --manifest   ../../test/2_5/real_manifest.jsonl \
    --model-path ../../checkpoints/plant2/epoch%3D029_final_3.ckpt \
    --emit-replay-sidecar
```

#### 2. Loop over all signs / all manifests

```bash
cd pdd-bench/scripts/per_sign_bench

for f in ../../test/*/*.jsonl; do
    sign=$(basename "$(dirname "$f")")
    src=$(basename "$f" .jsonl)
    python run_benchmark_mini.py \
        --policy   idm \
        --run-name "idm_${sign}_${src}" \
        --manifest "$f" \
        --emit-replay-sidecar
done
```

Repeat the loop for each baseline you want to evaluate (`comprehensive_rule_expert`, `carl`, `plant2`, etc.).

#### 3. Yield-sign scenarios (sign 2.4)

Yield (2.4) uses a dedicated runner that adds yield-specific termination conditions and an optional top-down GIF recorder:

```bash
cd pdd-bench/scripts/per_sign_bench

python yield_run_benchmark_mini_plant2.py \
    --policy    idm \
    --run-name  idm_yield \
    --manifest  /path/to/2_4_manifest.jsonl \
    --sign-type 2.4 \
    --emit-replay-sidecar \
    --save-gifs                    # optional: record top-down GIFs
```

### Compute Metrics

#### 1. Single-run metrics (quickest)

```bash
bash pdd-bench/scripts/per_sign_bench/run_metrics_single_run.sh \
    --run-dir eval_out/runs/idm_2_5 \
    --out-dir eval_out/metrics_idm_2_5 \
    --policy  idm
```

Outputs:
- `metrics_per_episode.csv` — episode-level table
- `aggregations/agg_per_baseline.csv` — per-baseline summary
- `reports/report_cumulative.md` — markdown report table
- `reports/report_cumulative_categories.md` — per-category breakdown

#### 2. Full multi-baseline pipeline

```bash
ROOT=eval_out bash pdd-bench/scripts/per_sign_bench/run_full_metrics_pipeline.sh

# Skip consolidation if replay jsonl files already exist:
SKIP_CONSOLIDATE=1 ROOT=eval_out bash pdd-bench/scripts/per_sign_bench/run_full_metrics_pipeline.sh
```

The pipeline runs:
1. `consolidate_replays.py` — merges `replay.json` sidecars → `<baseline>_replays.jsonl`
2. `build_episode_metrics_csv.py` — builds `metrics_per_episode.csv`
3. `build_oracle_baseline.py` — adds `oracle_rule` synthetic baseline
4. `aggregate_episode_metrics.py` — aggregates to per-baseline and per-sign CSVs

#### 3. Oracle baseline only

```bash
python3 pdd-bench/scripts/per_sign_bench/build_oracle_baseline.py \
    --csv eval_out/metrics_per_episode.csv
```

### 📊 Metrics

| Metric | Description |
|---|---|
| `target_compliant_event` | Ego obeyed the target sign within its zone (primary rule metric) |
| `arrived_dest` | Reached the destination |
| `route_completion` | Percent of route covered (0–100) |
| `total_violations` | Total traffic rule violations (all signs) |
| `comfort` | Standard nuplan based kinematic smoothness ratio |

## ⭐ Citation

TODO
