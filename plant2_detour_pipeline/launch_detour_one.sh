#!/usr/bin/env bash
# Launch ONE detour fine-tune. Generic so runs can be scheduled onto whatever
# GPUs are actually free (this box is shared -- other users' jobs come and go).
#
# Usage: launch_detour_one.sh <name> <gpu> <lr> [hydra.override=value ...]
#
# All runs share one dataset and one diskcache (/tmp/plant2_ds_cache_detour).
# That is safe: the cache is keyed by frame file path (dataset.py, via
# `labels[0].decode()`), not by dataset index, so runs with different
# oversampling factors reuse entries for the same frames instead of
# corrupting each other. Cache reads go through the atomic Cache.get() --
# the previous `key in cache` + `cache[key]` pattern was a TOCTOU race that
# killed a run mid-epoch once several trainings shared a cache at its size
# limit. Limit raised to 250G (disk has ~626G free) so it stops thrashing.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
PY="/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python"
# DETOUR_SPLIT overrides the training split. Use /tmp/detour_split_fixedroute
# (built by scripts/plant2_ft_pipeline/data/fix_route_target.py) for the
# leak-free target; the default below still points at the original split whose
# route input is byte-identical to the target.
SPLIT="${DETOUR_SPLIT:-$TRB_ROOT/plant2_detour_pipeline/detour_split_filtered}"
# timm fetches resnet18 weights from HF on model init; the proxy here fails
# intermittently and killed a run. The weights are already in the local HF
# cache, so force cache-only.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT0="$TRB_ROOT/stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt"
LOGDIR="$TRB_ROOT/plant2_detour_pipeline/logs"
DS_LOCAL="/tmp/plant2_ds_cache_detour"

NAME="${1:?usage: launch_detour_one.sh <name> <gpu> <lr> [overrides...]}"
GPU="${2:?}"
LR="${3:?}"
shift 3

ADDON="detour_${NAME}"
hydra_args=()
for o in "$@"; do hydra_args+=(--hydra-override "$o"); done

mkdir -p "$LOGDIR"
cd "$PIPELINE"
nohup "$PY" -u train/run_plant2_finetune.py \
  --split "$SPLIT" \
  --learning-rate "$LR" \
  --checkpoint-addon "$ADDON" \
  --cuda-device "$GPU" \
  --ds-local "$DS_LOCAL" \
  --cache-size-gb 250 \
  --batch-size 512 \
  --num-workers 4 \
  --max-epochs 20 \
  --ckpt-every-n-epochs 5 \
  --augment --no-filter-routes \
  --resume-ckpt "$CKPT0" \
  --wandb-mode offline \
  --python "$PY" \
  --hydra-override "user.working_dir=$TRB_ROOT/plant2" \
  "${hydra_args[@]}" \
  --hydra-run-dir "$TRB_ROOT/plant2/PlanT/outputs/PlanT2_train/${ADDON}" \
  --log "$LOGDIR/train_${ADDON}.log" \
  > "$LOGDIR/train_${ADDON}.nohup.log" 2>&1 < /dev/null &

echo $! > "$LOGDIR/train_${ADDON}.pid"
echo "[launch] $NAME gpu=$GPU lr=$LR pid=$(cat "$LOGDIR/train_${ADDON}.pid") overrides=$*"
