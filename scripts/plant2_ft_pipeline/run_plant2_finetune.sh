#!/usr/bin/env bash
# Fine-tune PlanT2 on an explicit train/val SPLIT.
# Required env:
#   SPLIT  absolute path to split root (.../plant2_l1_fv_experts_split)
#          Must NOT be plant2_l1_parallel300*_split unless you intentionally set it.
# Optional env overrides:
#   DS, DS_VAL (default: $SPLIT/train, $SPLIT/val),
#   CUDA_VISIBLE_DEVICES, LEARNING_RATE, CHECKPOINT_ADDON, BATCH_SIZE, NUM_WORKERS,
#   LR_SCHEDULER (multistep|cosine_warmup), WARMUP_RATIO, DS_LOCAL, SEED, PYTHON
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

PLANT="$TRB_ROOT/plant2"
PLAN_T="$PLANT/PlanT"
CKPT="${CKPT0:-$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt}"
SHIM="${SHIM:-$PIPELINE_DIR/plant2_py_shims/run_lit_finetune.py}"
# Default: shared-disk arbelyaev-sdc (PlanT2 FT + pdd-bench). Override with PYTHON=...
# Activate: conda activate /home/jovyan/shares/SR006.nfs3/shepelev/conda_envs/arbelyaev-sdc
ARBELYAEV_PY="$PY"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ARBELYAEV_PY" ]]; then
    PY="$ARBELYAEV_PY"
  elif [[ -x /home/user/conda/envs/zinkovich-sdc/bin/python ]]; then
    PY=/home/user/conda/envs/zinkovich-sdc/bin/python
  else
    PY=/home/user/conda/bin/python
  fi
else
  PY="$PYTHON"
fi

if [[ -z "${SPLIT:-}" ]]; then
  echo "ERROR: SPLIT must be set explicitly (e.g. SPLIT=$SHEPELEV/plant2_l1_fv_experts_split)" >&2
  echo "Refusing to default to plant2_l1_parallel300_split." >&2
  exit 1
fi
if [[ "$SPLIT" == *"parallel300"* ]]; then
  echo "WARNING: SPLIT points at parallel300 tree: $SPLIT" >&2
  echo "Set ALLOW_PARALLEL300=1 to proceed." >&2
  if [[ "${ALLOW_PARALLEL300:-}" != "1" ]]; then
    exit 1
  fi
fi

export SEED="${SEED:-1}"
export CHECKPOINT_ADDON="${CHECKPOINT_ADDON:-arbelyaev_ft5}"
export DS="${DS:-$SPLIT/train}"
export DS_VAL="${DS_VAL:-$SPLIT/val}"
export DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_${SEED}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# GPUS>1 picks the first N devices unless the caller pinned them by hand.
GPUS="${GPUS:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$GPUS" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS - 1)))"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Avoid user-site transformers (older) + broken root-owned flash_attn ABI mismatch.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# The entry-point shim ships in this repo; shepelev's copy is only a fallback
# for nodes where the repo predates it. Nodes without nfs3 must not need it.
LIT_ENTRY="${LIT_ENTRY:-$SHIM}"
[ -f "$LIT_ENTRY" ] || LIT_ENTRY="$SHEPELEV/collected_trajectories/plant2_py_shims/run_lit_finetune.py"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1536}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine_warmup}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
# Loss-side rebalance for the stop bin (lit_module._weighted_egospeed_ce);
# 1.0 = off, the historical default.
STOP_SPEED_LOSS_WEIGHT="${STOP_SPEED_LOSS_WEIGHT:-1.0}"
export CACHE_SIZE_GB="${CACHE_SIZE_GB:-641}"
export CKPT_EVERY_N_EPOCHS="${CKPT_EVERY_N_EPOCHS:-5}"
# tok_emb holds one embedding per object class; no single batch contains every
# class, so plain DDP aborts on "unused parameters" at the first step.
export DDP_STRATEGY="${DDP_STRATEGY:-ddp_find_unused_parameters_true}"

mkdir -p "$DS_LOCAL"
mkdir -p "$PLAN_T/log" "$PLAN_T/checkpoints_ft"
cd "$PLAN_T"

echo "============================================================"
echo "PlanT2 fine-tune  $(date -Is)"
echo "  SPLIT    = $SPLIT"
echo "  CKPT     = $CKPT"
echo "  DS       = $DS"
echo "  DS_VAL   = $DS_VAL"
echo "  DS_LOCAL = $DS_LOCAL"
echo "  SEED     = $SEED"
echo "  GPU      = $CUDA_VISIBLE_DEVICES (gpus=$GPUS)"
echo "  LR       = $LEARNING_RATE"
echo "  SCHED    = $LR_SCHEDULER (warmup_ratio=$WARMUP_RATIO)"
echo "  BS       = $BATCH_SIZE  workers=$NUM_WORKERS  epochs=$MAX_EPOCHS"
echo "  ADDON    = $CHECKPOINT_ADDON"
echo "  CACHE_GB = $CACHE_SIZE_GB  ckpt_every=$CKPT_EVERY_N_EPOCHS"
echo "  STOPW    = $STOP_SPEED_LOSS_WEIGHT  ts_lookahead=${TS_LOOKAHEAD:-0} wps_stride=${WPS_STRIDE:-1}"
echo "  PYTHON   = $PY"
echo "  ENTRY    = $LIT_ENTRY (flash_attn disabled)"
echo "  NOUSERSITE=$PYTHONNOUSERSITE"
echo "============================================================"

test -f "$CKPT"
test -d "$DS/data"
test -d "$DS_VAL/data"
test -f "$SPLIT/split_meta.json"
test -f "$LIT_ENTRY"

# Hydra override grammar: '=' inside values must be escaped.
hydra_esc() { printf '%s' "$1" | sed 's/=/\\=/g'; }

"$PY" -u "$LIT_ENTRY" \
  resume=True \
  "resume_path=$(hydra_esc "$CKPT")" \
  "gpus=$GPUS" \
  use_caching=True \
  "lr_scheduler=$LR_SCHEDULER" \
  "warmup_ratio=$WARMUP_RATIO" \
  "model.training.learning_rate=$LEARNING_RATE" \
  "model.training.max_epochs=$MAX_EPOCHS" \
  "model.training.batch_size=$BATCH_SIZE" \
  "model.training.num_workers=$NUM_WORKERS" \
  model.training.augment=False \
  model.training.augment_parked=False \
  "model.training.stop_speed_loss_weight=$STOP_SPEED_LOSS_WEIGHT" \
  "user.working_dir=$(hydra_esc "$PLANT")" \
  "model.training.log_path=$(hydra_esc "$PLAN_T/log/ft_${CHECKPOINT_ADDON}_${SEED}")" \
  "expname=ft_${CHECKPOINT_ADDON}" \
  "wandb_name=ft_${CHECKPOINT_ADDON}_${SEED}"
