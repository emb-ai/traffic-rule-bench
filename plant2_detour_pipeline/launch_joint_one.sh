#!/usr/bin/env bash
# Launch ONE joint (2.5 stop + 4.2.x detour) fine-tune.
# Usage: launch_joint_one.sh <name> <gpu> <lr> <max_epochs> [hydra.override=value ...]
# The joint split is built by /tmp/build_joint.sh: stop and detour dumps merged,
# route target rebuilt offline (fix_route_target.py) and signs dumped at 90 m.
set -euo pipefail
TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
PY="/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python"
SPLIT="${JOINT_SPLIT:-/tmp/joint_split_fx}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT0="$TRB_ROOT/stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt"
LOGDIR="$TRB_ROOT/plant2_detour_pipeline/logs"
DS_LOCAL="/tmp/plant2_ds_cache_joint"

NAME="${1:?usage: launch_joint_one.sh <name> <gpu> <lr> <max_epochs> [overrides...]}"
GPU="${2:?}"; LR="${3:?}"; EPOCHS="${4:?}"; shift 4
ADDON="joint_${NAME}"
hydra_args=(); for o in "$@"; do hydra_args+=(--hydra-override "$o"); done
mkdir -p "$LOGDIR"; cd "$PIPELINE"
nohup "$PY" -u train/run_plant2_finetune.py \
  --split "$SPLIT" --learning-rate "$LR" --checkpoint-addon "$ADDON" \
  --cuda-device "$GPU" --ds-local "$DS_LOCAL" --cache-size-gb 200 \
  --batch-size 512 --num-workers 4 --max-epochs "$EPOCHS" --ckpt-every-n-epochs 3 \
  --augment --no-filter-routes --resume-ckpt "$CKPT0" --wandb-mode offline --python "$PY" \
  --hydra-override "user.working_dir=$TRB_ROOT/plant2" "${hydra_args[@]}" \
  --hydra-run-dir "$TRB_ROOT/plant2/PlanT/outputs/PlanT2_train/${ADDON}" \
  --log "$LOGDIR/train_${ADDON}.log" \
  > "$LOGDIR/train_${ADDON}.nohup.log" 2>&1 < /dev/null &
echo $! > "$LOGDIR/train_${ADDON}.pid"
echo "[launch] $NAME gpu=$GPU lr=$LR ep=$EPOCHS pid=$(cat "$LOGDIR/train_${ADDON}.pid") overrides=$*"
