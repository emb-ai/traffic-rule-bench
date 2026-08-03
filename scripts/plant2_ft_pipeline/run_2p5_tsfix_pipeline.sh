#!/usr/bin/env bash
# Pipeline: retrofit 2.5 target_speed → fresh diskcache → FT (2 LRs) → eval Sign SR.
#
# Uses a dedicated DS_LOCAL so we do not fight /tmp/plant2_ds_cache_spatial_aug.
# New checkpoint addons: fvexp30_spatial_2p5_tsfix_lr{1e4,1e5}
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

CT="$PIPELINE_DIR"
PLAN_T="$TRB_ROOT/plant2/PlanT"
PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
SPLIT="$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5"
CKPT0="$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt"
SHIM="$PIPELINE_DIR/plant2_py_shims/run_lit_finetune.py"
SIGNS_DIR="$SHEPELEV/traffic-rule-bench/pdd-bench/scripts/per_sign_bench/plant2_rule_test"

export DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_2p5_tsfix}"
export CACHE_SIZE_GB="${CACHE_SIZE_GB:-400}"
export DS="$SPLIT/train"
export DS_VAL="$SPLIT/val"
export SPLIT
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export MAX_EPOCHS="${MAX_EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-1344}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CKPT_EVERY_N_EPOCHS="${CKPT_EVERY_N_EPOCHS:-5}"
export LR_SCHEDULER=cosine_warmup
export WARMUP_RATIO=0.1
export SEED=1

METRICS_ROOT="${METRICS_ROOT:-$SHEPELEV/plant2_ft_metrics/spatial_2p5_tsfix_eval_sign25}"
LOGDIR="$CT/logs_pipeline_2p5_tsfix"
mkdir -p "$LOGDIR" "$METRICS_ROOT" "$DS_LOCAL" "$PLAN_T/checkpoints_ft"
PIPE_LOG="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE_LOG"; }

hydra_esc() { printf '%s' "$1" | sed 's/=/\\=/g'; }

verify_measurements() {
  "$PY" - <<'PY'
import gzip, json
from pathlib import Path
import sys
sys.path.insert(0, "$TRB_ROOT/plant2/PlanT")
from util.sign_id import load_split_meta_route2sign, load_uid2sign, resolve_route_sign

root = Path("/home/jovyan/shares/SR006.nfs3/shepelev/plant2_l1_fv_experts_split_signs")
extra = load_split_meta_route2sign(root / "split_meta.json")
uid = load_uid2sign()
broken = []
n = 0
for split in ("train", "val"):
    data = root / split / "data"
    for p in data.iterdir():
        if not p.is_dir():
            continue
        s = extra.get(p.name) or resolve_route_sign(p.name, uid)
        if s != "2.5":
            continue
        n += 1
        files = sorted((p / "measurements").glob("*.json.gz"))
        if not files:
            broken.append((p.name, "no_meas"))
            continue
        # sparse sample
        idxs = list(range(0, len(files), max(1, len(files) // 15)))[:15]
        all20 = True
        for i in idxs:
            with gzip.open(files[i], "rt") as f:
                d = json.load(f)
            if abs(float(d["target_speed"]) - 20.0) > 1e-6:
                all20 = False
                break
        if all20:
            broken.append((p.name, "target_all_20"))
print(f"n_2.5={n} broken={len(broken)}")
for b in broken[:10]:
    print(" BROKEN", b)
raise SystemExit(1 if broken else 0)
PY
}

# ---------- 1) retrofit ----------
if [[ "${SKIP_RETROFIT:-0}" == "1" ]]; then
  log "STAGE retrofit SKIPPED (SKIP_RETROFIT=1)"
else
  log "STAGE retrofit --signs 2.5"
  "$PY" -u "$CT/retrofit_target_speed_expert.py" --signs 2.5 --workers 32 \
    2>&1 | tee -a "$LOGDIR/retrofit.log"
fi
log "STAGE verify measurements"
verify_measurements

# ---------- 2) extract+patch 2.5 keys from big cache (no full iterkeys / prefill) ----------
SRC_CACHE="${SRC_CACHE:-/tmp/plant2_ds_cache_spatial_aug}"
log "STAGE extract_patch src=$SRC_CACHE dst=$DS_LOCAL"
"$PY" -u "$CT/extract_patch_2p5_cache.py" \
  --src "$SRC_CACHE" \
  --dst "$DS_LOCAL" \
  --split "$SPLIT" \
  --cache-size-gb "$CACHE_SIZE_GB" \
  --reset-dst \
  --materialize-missing \
  2>&1 | tee -a "$LOGDIR/extract_patch.log"
log "extract done volume=$(du -sh "$DS_LOCAL" 2>/dev/null | awk '{print $1}') keys=$(
  "$PY" -c "from diskcache import Cache; c=Cache('$DS_LOCAL'); print(len(c)); c.close()"
)"

# ---------- 3) FT ----------
log "STAGE train"
declare -a JOBS=(
  "0|1e-4|fvexp30_spatial_2p5_tsfix_lr1e4"
  "1|1e-5|fvexp30_spatial_2p5_tsfix_lr1e5"
)

pids=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r GPU LR ADDON <<<"$spec"
  LR_TAG=$(echo "$LR" | tr -d '-')
  LOG="/tmp/plant2_ft_2p5_tsfix_lr${LR_TAG}.log"
  CKPT_DIR="$PLAN_T/checkpoints_ft/$ADDON"
  mkdir -p "$CKPT_DIR"
  (
    set -euo pipefail
    cd "$PLAN_T"
    export CUDA_VISIBLE_DEVICES=$GPU
    export LEARNING_RATE=$LR
    export CHECKPOINT_ADDON=$ADDON
    export DS_LOCAL CACHE_SIZE_GB MAX_EPOCHS BATCH_SIZE NUM_WORKERS
    export CKPT_EVERY_N_EPOCHS LR_SCHEDULER WARMUP_RATIO SEED WANDB_MODE
    export PYTHONNOUSERSITE=1
    echo "FT_START $(date -Is) gpu=$GPU lr=$LR addon=$ADDON" | tee "$LOG"
    "$PY" -u "$SHIM" \
      resume=True \
      "resume_path=$(hydra_esc "$CKPT0")" \
      gpus=1 \
      use_caching=True \
      "lr_scheduler=$LR_SCHEDULER" \
      "warmup_ratio=$WARMUP_RATIO" \
      "model.training.learning_rate=$LR" \
      "model.training.max_epochs=$MAX_EPOCHS" \
      "model.training.batch_size=$BATCH_SIZE" \
      "model.training.num_workers=$NUM_WORKERS" \
      model.training.augment=True \
      model.training.augment_parked=False \
      '+model.training.filter_routes=False' \
      "model.training.log_path=$(hydra_esc "$PLAN_T/log/ft_${ADDON}_${SEED}")" \
      "expname=ft_${ADDON}" \
      "wandb_name=ft_${ADDON}_${SEED}" \
      2>&1 | tee -a "$LOG"
    echo "FT_EXIT=$? $(date -Is)" | tee -a "$LOG"
  ) &
  pids+=($!)
  log "started FT gpu=$GPU lr=$LR addon=$ADDON pid=${pids[-1]} log=$LOG"
done

fail_train=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail_train=$((fail_train + 1))
done
log "FT finished fail=$fail_train"
[[ "$fail_train" -eq 0 ]] || exit 1

# ---------- 4) eval best + last ----------
log "STAGE eval"
pick_ckpt() {
  local dir="$1" prefer_best="$2"
  if [[ "$prefer_best" == best ]]; then
    ls -1t "$dir"/best_*.ckpt 2>/dev/null | head -1 || true
  else
    ls -1t "$dir"/epoch=*.ckpt 2>/dev/null | head -1 || true
  fi
}

declare -a EVAL_JOBS=()
gpu=0
for addon in fvexp30_spatial_2p5_tsfix_lr1e4 fvexp30_spatial_2p5_tsfix_lr1e5; do
  lr_tag=${addon##*_lr}
  for slot in best last; do
    ckpt=$(pick_ckpt "$PLAN_T/checkpoints_ft/$addon" "$slot")
    if [[ -z "$ckpt" ]]; then
      log "WARN missing ckpt addon=$addon slot=$slot"
      continue
    fi
    bn=$(basename "$ckpt")
    if [[ "$bn" =~ best_([0-9]+)_ ]]; then
      tag="fvexp30_spatial_2p5_tsfix_lr${lr_tag}_best${BASH_REMATCH[1]}_sign25"
    elif [[ "$bn" =~ epoch=([0-9]+)_ ]]; then
      tag="fvexp30_spatial_2p5_tsfix_lr${lr_tag}_ep${BASH_REMATCH[1]}_sign25"
    else
      tag="fvexp30_spatial_2p5_tsfix_lr${lr_tag}_${slot}_sign25"
    fi
    EVAL_JOBS+=("$gpu|$tag|$ckpt")
    gpu=$(( (gpu + 1) % 7 ))
  done
done

eval_one() {
  local gpu="$1" tag="$2" ckpt="$3"
  local logf="$METRICS_ROOT/$tag/logs/eval_checkpoint.log"
  mkdir -p "$METRICS_ROOT/$tag/logs" "$SIGNS_DIR/output/$tag"
  ln -sfn "$SIGNS_DIR/output/$tag" "$METRICS_ROOT/$tag/signs"
  printf '%s\n' "$ckpt" > "$METRICS_ROOT/$tag/ckpt.txt"
  if [[ -f "$SIGNS_DIR/output/$tag/_summary/summary.md" ]]; then
    log "EVAL SKIP $tag"
    return 0
  fi
  log "EVAL START tag=$tag gpu=$gpu ckpt=$ckpt"
  (
    cd "$SIGNS_DIR"
    unset PYTHONPATH
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
    export PYTHONNOUSERSITE=1 PER_SIGN_COMPLIANT_NPC=1 SDL_AUDIODRIVER=dummy
    "$PY" -u eval_checkpoint_on_test.py \
      --policies plant2 --model-paths "plant2:$ckpt" \
      --jobs 8 --scenes-per-job 20 --only 2.5 --keep-going --run-name "$tag"
    "$PY" summarize_reports.py --run-name "$tag" --baseline plant2_default \
      --out-dir "output/$tag/_summary"
  ) >>"$logf" 2>&1
  if [[ -f "$SIGNS_DIR/output/$tag/_summary/summary.md" ]]; then
    log "EVAL DONE $tag"
    cat "$SIGNS_DIR/output/$tag/_summary/summary.md" | tee -a "$PIPE_LOG"
    return 0
  fi
  log "EVAL FAIL $tag"
  tail -40 "$logf" | tee -a "$PIPE_LOG" || true
  return 1
}

epids=()
for spec in "${EVAL_JOBS[@]}"; do
  IFS='|' read -r gpu tag ckpt <<<"$spec"
  eval_one "$gpu" "$tag" "$ckpt" &
  epids+=($!)
done
fail_eval=0
for pid in "${epids[@]}"; do
  wait "$pid" || fail_eval=$((fail_eval + 1))
done
log "ALL DONE fail_eval=$fail_eval metrics=$METRICS_ROOT"
exit "$fail_eval"
