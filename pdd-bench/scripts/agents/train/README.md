# Plant2 Training Pipeline

All scripts assume the `**plant2**` conda env and run from the repo root
(`/home/jovyan/shares/SR006.nfs2/arbelyaev/sdc`).

---

## Environment setup

To create the env (Python 3.10, CUDA 12.4):

```bash
conda env create -f plant2/environment.yml
conda activate plant2

# Install local MetaDrive from the submodule
pip install -e metadrive/
```

Key packages installed: `torch==2.6.0`, `pytorch-lightning==2.5.0`, `metadrive-simulator==0.4.3`,
`stable_baselines3==2.7.1`, `numpy==1.26.4`.

---

## Option A — Quick smoke test (mini benchmark data)

Repack existing expert `.pkl` files → `.pt`, fine-tune, then run benchmark.
Needs `pdd-bench/data/benchmark_mini` symlink pointing at the mini dataset.

```bash
tmux new -s plant2_ft
export SDC_AR=/home/jovyan/shares/SR006.nfs2/arbelyaev/sdc
export INIT_CKPT=$SDC_AR/../plant2/models/epoch%3D029_final_3.ckpt
bash pdd-bench/scripts/agents/train/run_plant2_mini_repack_train_eval.sh
```

Key overrides:

```bash
REPACK_LIMIT=5          # limit episodes for a fast sanity check
REPACK_NUM_WORKERS=4    # parallel repack workers
EPOCHS=5
```

---

## Option B — Full pipeline (collect → shard → train)

### 1. Collect trajectories in parallel

```bash
# v5 corpus (recommended)
N_WORKERS=8 OUTPUT_DIR=pdd-bench/outputs/benchmark_sign_trajectories_v5 \
  bash pdd-bench/scripts/agents/train/run_collect_v5_parallel.sh
```

Monitor: `tail -F pdd-bench/outputs/benchmark_sign_trajectories_v5/_collect_logs/*.log`

### 2. Shard `.pt` files for fast DataLoader

```bash
# Split across N nodes (single node = 1)
python pdd-bench/scripts/agents/train/prepare_shard_splits.py \
    --nodes 1 \
    --pt-dir pdd-bench/outputs/benchmark_sign_trajectories_v5 \
    --output-dir pdd-bench/outputs/shard_splits

python pdd-bench/scripts/agents/train/shard_plant2_pt.py \
    --input-dir  pdd-bench/outputs/benchmark_sign_trajectories_v5 \
    --output-dir pdd-bench/outputs/plant2_shards_v5 \
    --steps-per-shard 1024 --shuffle
```

### 3. Train

```bash
# 4 GPUs, 30 epochs (edit CUDA_VISIBLE_DEVICES / EPOCHS / BATCH_PER_GPU as needed)
export CKPT=/home/jovyan/shares/SR006.nfs2/arbelyaev/sdc/epoch%3D029_final_3.ckpt
export DATA_DIR=pdd-bench/outputs/benchmark_sign_trajectories_v5

bash pdd-bench/scripts/agents/train/run_plant2_train_benchmark_v5.sh
```

Or sharded training (faster I/O on large corpora):

```bash
DATA_DIR=pdd-bench/outputs/plant2_shards_v5 \
  bash pdd-bench/scripts/agents/train/run_plant2_train_benchmark_v5.sh
```

Logs: `pdd-bench/outputs/plant2_supervised_benchmark_v5/train_console.log`

---

## Option C — Train only (data already exists)

```bash
DATA_DIR=pdd-bench/outputs/benchmark_sign_trajectories_v4 \
EPOCHS=10 \
  bash pdd-bench/scripts/agents/train/run_plant2_train_only.sh
```

---

## Pretrained checkpoints

Ready-to-use checkpoints are published at
**[emb-ai/traffic-rule-bench-models](https://huggingface.co/emb-ai/traffic-rule-bench-models)**:

| HF path | Description |
|---|---|
| `plant2_pretrain/epoch=029_final_3.ckpt` | Plant2 base pretrained model (PlanT/HFLM) |
| `plant2_finetuned/plant2_supervised_2nd_final.pt` | Plant2 fine-tuned on traffic-rule benchmark |
| `carl/nuplan_51479_1B/model_best.pth` | CaRL nuPlan best checkpoint |

### Download and load

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import torch

# --- Plant2 pretrain (Lightning .ckpt) ---
ckpt_path = hf_hub_download(
    repo_id="emb-ai/traffic-rule-bench-models",
    filename="plant2_pretrain/epoch=029_final_3.ckpt",
)
# use as INIT_CKPT for fine-tuning scripts
# or load directly:
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)

# --- Plant2 fine-tuned (.pt) ---
ft_path = hf_hub_download(
    repo_id="emb-ai/traffic-rule-bench-models",
    filename="plant2_finetuned/plant2_supervised_2nd_final.pt",
)
ckpt = torch.load(ft_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# --- CaRL ---
carl_path = hf_hub_download(
    repo_id="emb-ai/traffic-rule-bench-models",
    filename="carl/nuplan_51479_1B/model_best.pth",
)
carl_ckpt = torch.load(carl_path, map_location="cpu", weights_only=False)
```

To use the pretrained checkpoint as a starting point for fine-tuning:

```bash
export INIT_CKPT=$(python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('emb-ai/traffic-rule-bench-models',
      'plant2_pretrain/epoch=029_final_3.ckpt'))
")
bash pdd-bench/scripts/agents/train/run_plant2_train_only.sh
```

---

## Training output

| Path | Content |
|---|---|
| `pdd-bench/outputs/<run>/plant2_supervised_2nd_final.pt` | Final weights |
| `pdd-bench/outputs/<run>/last.ckpt` | Last Lightning checkpoint |
| `pdd-bench/outputs/<run>/metrics_csv/` | Per-epoch CSV metrics |
| `pdd-bench/outputs/<run>/tensorboard/` | TensorBoard logs |

