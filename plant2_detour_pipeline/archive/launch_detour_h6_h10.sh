#!/usr/bin/env bash
# H6-H10: attack the route-copying shortcut (see DETOUR_HYPOTHESES.md).
#
# All five share the SAME data (detour_split_filtered) and the SAME warm
# diskcache (/tmp/plant2_ds_cache_detour) -- verified safe to share: the
# cache is keyed by frame file path (dataset.py:298 `labels[0].decode()`),
# not by dataset index, so runs with different oversampling factors (hence
# different dataset lengths) hit the same keys for the same frames instead
# of corrupting each other. Nothing here writes into the shared data dirs.
#
# GPUs 0 and 3 are left alone (other users' processes); 7 is running H5.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
PY="/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python"
SPLIT="$TRB_ROOT/plant2_detour_pipeline/detour_split_filtered"
CKPT0="$TRB_ROOT/stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt"
LOGDIR="$TRB_ROOT/plant2_detour_pipeline/logs"
DS_LOCAL="/tmp/plant2_ds_cache_detour"

mkdir -p "$LOGDIR"

launch() {
  local name="$1" gpu="$2" lr="$3"; shift 3
  local extra=("$@")
  local addon="detour_${name}"
  local hydra_args=()
  for o in "${extra[@]}"; do hydra_args+=(--hydra-override "$o"); done

  echo "[launch] $name gpu=$gpu lr=$lr extra=${extra[*]:-none}"
  (
    cd "$PIPELINE"
    nohup "$PY" -u train/run_plant2_finetune.py \
      --split "$SPLIT" \
      --learning-rate "$lr" \
      --checkpoint-addon "$addon" \
      --cuda-device "$gpu" \
      --ds-local "$DS_LOCAL" \
      --cache-size-gb 80 \
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
      --hydra-run-dir "$TRB_ROOT/plant2/PlanT/outputs/PlanT2_train/${addon}" \
      --log "$LOGDIR/train_${addon}.log" \
      > "$LOGDIR/train_${addon}.nohup.log" 2>&1 < /dev/null &
    echo $! > "$LOGDIR/train_${addon}.pid"
  )
}

# H6: pure route dropout — can the model plan without leaning on the route?
launch h6_routedrop50_lr1e4_ep20 1 1e-4 \
  "model.training.route_dropout_p=0.5"

# H7: route kept but unreliable (lateral jitter) instead of removed outright.
launch h7_routenoise15_lr1e4_ep20 2 1e-4 \
  "model.training.route_lateral_noise_m=1.5"

# H8: dose-response — is more shortcut removal monotonically better?
launch h8_routedrop80_lr1e4_ep20 4 1e-4 \
  "model.training.route_dropout_p=0.8"

# H9: shortcut removal + H5's frame oversampling near the cone.
launch h9_routedrop50_oversample8d5_lr1e4_ep20 5 1e-4 \
  "model.training.route_dropout_p=0.5" \
  "model.training.oversample_cone_distance_m=5" \
  "model.training.oversample_factor=8"

# H10: shortcut removal + H2's per-sample lateral loss reweighting.
launch h10_routedrop50_alpha30_lr1e4_ep20 6 1e-4 \
  "model.training.route_dropout_p=0.5" \
  "model.waypoints.lateral_reweight_alpha=30"

sleep 3
echo
echo "PIDs:"
for n in h6_routedrop50_lr1e4_ep20 h7_routenoise15_lr1e4_ep20 h8_routedrop80_lr1e4_ep20 \
         h9_routedrop50_oversample8d5_lr1e4_ep20 h10_routedrop50_alpha30_lr1e4_ep20; do
  printf '  %-45s %s\n' "$n" "$(cat "$LOGDIR/train_detour_${n}.pid" 2>/dev/null || echo '?')"
done
