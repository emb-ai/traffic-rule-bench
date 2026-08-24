#!/usr/bin/env bash
# PlanT STOP class_emb — lr=3e-4 @ 20 and 30 epochs (parallel).
# Eval last_ft only; append metrics into ep10 RESULTS.md (fixed path).
#
# Reuses signfix dump/split + prefill. Does NOT re-dump, does NOT touch
# plant2_stop_pipeline_signfix/ or plant2_stop_pipeline_lrgrid_ep30/ artifacts,
# does NOT write into the zinkovich tree.
#
# Architecture: shared class_emb + attr_emb. Pretrain CKPT0 has tok_emb;
# lit_finetune loads with strict=False.
#
# Designed for: setsid nohup bash run_ep20_30.sh >>logs/nohup.out 2>&1 &
# Skip-if-complete per stage so a crash mid-run can resume.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
STOP_DATA="${STOP_DATA:-$TRB_ROOT/stop_data}"
PRIORITY="$TRB_ROOT/pdd-bench/scripts/per_sign_bench/priority_bench"
SPLIT="${SPLIT:-$TRB_ROOT/plant2_stop_pipeline_signfix/plant2_l1_stop_split}"
SIGNFIX_WORK="$TRB_ROOT/plant2_stop_pipeline_signfix"
EP10_WORK="$TRB_ROOT/plant2_stop_pipeline_lrgrid_ep10_classemb"
WORK="$TRB_ROOT/plant2_stop_pipeline_lrgrid_ep20_30_classemb"
RESULTS_MD="$EP10_WORK/RESULTS.md"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
PY="${PY:-/home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python}"
CKPT0="${CKPT0:-$STOP_DATA/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt}"
TEST_MANIFEST="$STOP_DATA/output/ts_test/real_manifest.jsonl"
SCENES="$STOP_DATA/scenes"
TRAIN_EXPERTS="$STOP_DATA/trajectories/debug_train_400/experts/experts_scene_uid_top1.jsonl"

export TRB_ROOT
export SHEPELEV="$WORK"
export PIPELINE_DIR="$PIPELINE"
export PLAN_T="$TRB_ROOT/plant2/PlanT"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export PER_SIGN_COMPLIANT_NPC=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export PLANT2_DUMP_SIGN_CLASSES="${PLANT2_DUMP_SIGN_CLASSES:-2.5}"

PP_BELYAEV="$TRB_ROOT/metadrive:$TRB_ROOT/pdd-bench:$TRB_ROOT/pdd-bench/scripts/per_sign_bench"
export PYTHONPATH="$PP_BELYAEV"
EVAL_PRIORITY="$PRIORITY"

DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_stop_signfix}"
CACHE_SIZE_GB="${CACHE_SIZE_GB:-400}"
# Lean intermediate saves (NFS nearly full): every 5 ep + last + best.
CKPT_EVERY="${CKPT_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
JOBS_EVAL_SOLO="${JOBS_EVAL_SOLO:-8}"
JOBS_EVAL_PARALLEL="${JOBS_EVAL_PARALLEL:-2}"

# Fixed LR; two epoch budgets. Avoid busy GPUs 0/1.
LR_VAL="${LR_VAL:-3e-4}"
GPU_EP20="${GPU_EP20:-3}"
GPU_EP30="${GPU_EP30:-4}"
EVAL_GPUS="${EVAL_GPUS:-6 7}"

# tag:epochs pairs (addon = stop_classemb_<tag>)
EXP_GRID=(
  "lr3e4_ep20:20"
  "lr3e4_ep30:30"
)

LOGDIR="$WORK/logs"
MASTER_LOG="$LOGDIR/pipeline_master.log"
STATUS="$WORK/STATUS.txt"
CURRENT_STAGE="$WORK/CURRENT_STAGE.txt"
TRAIN_META="$WORK/train_meta"
EVAL_ROOT="$WORK/eval"
LOCKDIR="$LOGDIR/locks"
PIDDIR="$LOGDIR/pids"

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  STATUS="$LOGDIR/preflight_status.txt"
fi

export WORK SPLIT DS_LOCAL CACHE_SIZE_GB CKPT_EVERY
export JOBS_EVAL_SOLO JOBS_EVAL_PARALLEL LOGDIR TEST_MANIFEST SCENES CKPT0
export STOP_DATA TRAIN_META EVAL_ROOT BATCH_SIZE NUM_WORKERS
export PY PLAN_T PIPELINE TRB_ROOT RESULTS_MD LR_VAL EP10_WORK

mkdir -p "$LOGDIR" "$WORK" "$TRAIN_META" "$EVAL_ROOT" "$PLAN_T/checkpoints_ft" "$LOCKDIR" "$PIDDIR"

ts() { date -Is; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }
cur_stage() { cat "$CURRENT_STAGE" 2>/dev/null || echo unknown; }

with_status_lock() {
  local lock="$LOCKDIR/status.lock"
  (
    flock 9
    "$@"
  ) 9>"$lock"
}

FAIL_RECORDED=0
fail() {
  FAIL_RECORDED=1
  log "FAIL stage=$(cur_stage): $*"
  with_status_lock bash -c "echo \"FAIL $(cur_stage) $(ts) $*\" >> \"$STATUS\""
  exit 1
}
set_stage() { echo "$1" > "$CURRENT_STAGE"; log "stage=$1"; }
ok_stage() {
  log "OK stage=$1"
  with_status_lock bash -c "echo \"OK $1 $(ts)\" >> \"$STATUS\""
}
skip_stage() {
  log "SKIP stage=$1 reason=$2"
  with_status_lock bash -c "echo \"SKIP $1 $(ts) $2\" >> \"$STATUS\""
}

on_exit() {
  local rc=$?
  [[ "$rc" -eq 0 || "$FAIL_RECORDED" -eq 1 ]] && return 0
  log "FAIL stage=$(cur_stage): abnormal exit rc=$rc"
  with_status_lock bash -c "echo \"FAIL $(cur_stage) $(ts) abnormal exit rc=$rc\" >> \"$STATUS\""
}
on_signal() {
  FAIL_RECORDED=1
  log "FAIL stage=$(cur_stage): killed by SIG$1 — terminating children"
  local f pid
  for f in "$PIDDIR"/*.pid; do
    [[ -f "$f" ]] || continue
    pid="$(cat "$f" 2>/dev/null || true)"
    [[ -n "${pid:-}" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  with_status_lock bash -c "echo \"FAIL $(cur_stage) $(ts) killed by SIG$1\" >> \"$STATUS\""
  exit 143
}
trap on_exit EXIT
trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

if [[ "$CHECK_ONLY" -eq 0 ]]; then
  if [[ -f "$STATUS" ]]; then
    cp -f "$STATUS" "$LOGDIR/STATUS.prev.$(date +%Y%m%d_%H%M%S)" || true
  fi
  {
    echo "RUNNING $(ts) mode=parallel_ep20_30 lr=$LR_VAL"
    if [[ -f "$STATUS" ]]; then
      grep -E '^(OK|SKIP) ' "$STATUS" || true
    fi
  } > "$STATUS.tmp"
  mv -f "$STATUS.tmp" "$STATUS"
fi
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "=== class_emb ep20/ep30 preflight (--check-only) ==="
else
  log "=== PlanT STOP class_emb lr=${LR_VAL} ep20+ep30 PARALLEL start ==="
fi
log "WORK=$WORK PY=$PY CKPT0=$CKPT0"
log "RESULTS_MD=$RESULTS_MD (fixed; ep10 rows preserved via merge)"
log "SPLIT=$SPLIT DS_LOCAL=$DS_LOCAL"
log "TEST_MANIFEST=$TEST_MANIFEST SCENES=$SCENES"
log "GPU map: ep20=$GPU_EP20 ep30=$GPU_EP30 EVAL_GPUS=$EVAL_GPUS"
log "EXP_GRID=${EXP_GRID[*]}"

[[ -x "$PY" ]] || fail "python missing: $PY"
[[ -f "$CKPT0" ]] || fail "ckpt missing: $CKPT0"
[[ -f "$TEST_MANIFEST" ]] || fail "test manifest missing: $TEST_MANIFEST"
[[ -d "$SCENES" ]] || fail "scenes missing: $SCENES"
[[ -f "$SPLIT/split_meta.json" ]] || fail "split missing: $SPLIT"
[[ -d "$SPLIT/train/data" && -d "$SPLIT/val/data" ]] || fail "split train/val data missing"
[[ -d "$EP10_WORK" ]] || fail "ep10 work missing: $EP10_WORK"

# Architecture gate: must be class_emb + attr_emb (no tok_emb / sign_emb).
"$PY" - <<'PY' || fail "model architecture check failed (need class_emb, no tok_emb/sign_emb)"
from pathlib import Path
import os, re
p = Path(os.environ["PLAN_T"]) / "model.py"
text = p.read_text()
assert "self.class_emb" in text, "missing self.class_emb"
assert "self.attr_emb" in text, "missing self.attr_emb"
assert not re.search(r"self\.tok_emb\b", text), "unexpected self.tok_emb"
assert not re.search(r"self\.sign_emb\b", text), "unexpected self.sign_emb"
print("arch OK: class_emb + attr_emb (no tok_emb/sign_emb)")
PY

"$PY" - <<'PY' || fail "split validation failed"
import json, os
from pathlib import Path
split = Path(os.environ["SPLIT"])
meta = json.loads((split / "split_meta.json").read_text())
assert meta["per_sign"]["2.5"]["n_train"] == 294, meta
assert meta["per_sign"]["2.5"]["n_val"] == 50, meta
n_train = sum(1 for p in (split / "train" / "data").iterdir() if p.is_dir())
n_val = sum(1 for p in (split / "val" / "data").iterdir() if p.is_dir())
assert n_train == 294 and n_val == 50, (n_train, n_val)
print(f"split OK train={n_train} val={n_val}")
PY

DATA_CHECK="$SIGNFIX_WORK/check_stop_data.py"
[[ -f "$DATA_CHECK" ]] || fail "data check missing: $DATA_CHECK"
"$PY" "$DATA_CHECK" \
  --scenes "$SCENES" \
  --test-manifest "$TEST_MANIFEST" \
  --train-experts "$TRAIN_EXPERTS" \
  --ckpt "$CKPT0" \
  2>&1 | tee -a "$MASTER_LOG" \
  || fail "stop-data validation failed"

code_preflight() {
  local dir="$1" script="$2"
  ( cd "$dir" && PYTHONPATH="$PP_BELYAEV" "$PY" "$script" --help ) >/dev/null 2>&1
}
code_preflight "$PRIORITY" "run_benchmark.py" \
  || fail "code preflight: $PRIORITY/run_benchmark.py"
code_preflight "$PRIORITY" "eval_pipeline.py" \
  || fail "code preflight: $PRIORITY/eval_pipeline.py"
log "preflight OK: data + entrypoints"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "=== PREFLIGHT OK (--check-only), no stage run ==="
  echo "PREFLIGHT_OK $(ts)" > "$STATUS"
  exit 0
fi

prefill_ready() {
  [[ -d "$DS_LOCAL" ]] || return 1
  [[ -f "$DS_LOCAL/cache.db" || -f "$DS_LOCAL/cache.db-wal" || -f "$DS_LOCAL/cache.db-shm" ]] || return 1
}

stage_prefill() {
  if prefill_ready; then
    skip_stage prefill "existing diskcache at $DS_LOCAL"
    return 0
  fi
  set_stage prefill
  mkdir -p "$DS_LOCAL"
  log "prefill: train+val -> $DS_LOCAL (rebuild)"
  cd "$PIPELINE"
  "$PY" -u data/prefill_diskcache.py parallel \
    --ds "$SPLIT/train" \
    --ds-val "$SPLIT/val" \
    --ds-local "$DS_LOCAL" \
    --cache-size-gb "$CACHE_SIZE_GB" \
    --augment \
    --max-workers 16 \
    --python "$PY" \
    --log-dir "$LOGDIR/prefill" \
    2>&1 | tee "$LOGDIR/04_prefill.log"
  prefill_ready || fail "prefill validation failed"
  ok_stage prefill
}

# Distinct addons: stop_classemb_lr3e4_ep20 / stop_classemb_lr3e4_ep30
addon_for() { echo "stop_classemb_${1}"; }

ckpt_dir_for() {
  local addon
  addon="$(addon_for "$1")"
  echo "$PLAN_T/checkpoints_ft/$addon"
}

gpu_for_tag() {
  case "$1" in
    lr3e4_ep20) echo "$GPU_EP20" ;;
    lr3e4_ep30) echo "$GPU_EP30" ;;
    *) fail "no GPU mapping for $1" ;;
  esac
}

epochs_for_tag() {
  case "$1" in
    lr3e4_ep20) echo 20 ;;
    lr3e4_ep30) echo 30 ;;
    *) fail "no epochs for $1" ;;
  esac
}

find_last_ckpt() {
  local dir="$1"
  ls -1t "$dir"/last_ft_*.ckpt 2>/dev/null | head -1 || true
}

find_best_ckpt() {
  local dir="$1"
  ls -1t "$dir"/best_*.ckpt 2>/dev/null | head -1 || true
}

status_has_ok() {
  local stage="$1"
  grep -qE "^OK ${stage} " "$STATUS" 2>/dev/null
}

# last_ft exists AND Lightning epoch >= max_epochs-1
train_ckpt_complete() {
  local ckpt_dir="$1"
  local max_epochs="$2"
  local last
  last="$(find_last_ckpt "$ckpt_dir")"
  [[ -n "$last" ]] || return 1
  MAX_EPOCHS="$max_epochs" LAST_CKPT="$last" "$PY" - <<'PY'
import os, sys
import torch
path = os.environ["LAST_CKPT"]
want = int(os.environ["MAX_EPOCHS"]) - 1
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"ckpt_read_fail {e}", flush=True)
    sys.exit(1)
epoch = int(ckpt.get("epoch", -1))
print(f"last_ft epoch={epoch} need>={want}", flush=True)
sys.exit(0 if epoch >= want else 1)
PY
}

train_is_complete() {
  local tag="$1"
  local epochs ckpt_dir
  epochs="$(epochs_for_tag "$tag")"
  ckpt_dir="$(ckpt_dir_for "$tag")"
  if status_has_ok "train_${tag}"; then
    return 0
  fi
  train_ckpt_complete "$ckpt_dir" "$epochs"
}

write_train_meta() {
  local tag="$1" lr="$2" epochs="$3" ckpt_dir="$4" train_log="$5"
  export EXP_TAG="$tag" LR_VAL_META="$lr" MAX_EPOCHS="$epochs" \
    CKPT_DIR_META="$ckpt_dir" TRAIN_LOG="$train_log" TRAIN_META
  "$PY" - <<'PY'
import json, os, re
from pathlib import Path

ckpt_dir = Path(os.environ["CKPT_DIR_META"])
train_log = Path(os.environ["TRAIN_LOG"])
out = Path(os.environ["TRAIN_META"]) / f"{os.environ['EXP_TAG']}.json"

last = sorted(ckpt_dir.glob("last_ft_*.ckpt"), key=lambda p: p.stat().st_mtime)
best = sorted(ckpt_dir.glob("best_*.ckpt"), key=lambda p: p.stat().st_mtime)
best_epoch = None
if best:
    m = re.search(r"best_(\d+)_", best[-1].name)
    if m:
        best_epoch = int(m.group(1))

final_train = final_val = None
if train_log.is_file():
    text = train_log.read_text(errors="replace")
    for pat, key in (
        (r"train_loss[=:]\s*([0-9.eE+-]+)", "train"),
        (r"val_loss[=:]\s*([0-9.eE+-]+)", "val"),
        (r"'train_loss':\s*([0-9.eE+-]+)", "train"),
        (r"'val_loss':\s*([0-9.eE+-]+)", "val"),
    ):
        hits = re.findall(pat, text)
        if hits:
            if key == "train":
                final_train = float(hits[-1])
            else:
                final_val = float(hits[-1])

meta = {
    "lr_tag": os.environ["EXP_TAG"],
    "lr": os.environ["LR_VAL_META"],
    "max_epochs": int(os.environ["MAX_EPOCHS"]),
    "addon": ckpt_dir.name,
    "ckpt_dir": str(ckpt_dir),
    "ckpt_last": str(last[-1]) if last else None,
    "ckpt_best": str(best[-1]) if best else None,
    "best_epoch": best_epoch,
    "final_train_loss": final_train,
    "final_val_loss": final_val,
    "train_log": str(train_log),
}
out.write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY
}

eval_ready() {
  local out="$1"
  [[ -f "$out/reports/report_cumulative.md" && -f "$out/reports/cumulative.json" ]]
}

acquire_eval_gpu() {
  local g lock fd
  while true; do
    for g in $EVAL_GPUS; do
      lock="$LOCKDIR/eval_gpu_${g}.lock"
      exec {fd}<>"$lock"
      if flock -n "$fd"; then
        echo "$g:$fd"
        return 0
      fi
      exec {fd}>&-
    done
    sleep 5
  done
}

release_eval_gpu() {
  local fd="$1"
  flock -u "$fd" 2>/dev/null || true
  exec {fd}>&- 2>/dev/null || true
}

count_active_evals() {
  local n=0 f
  for f in "$PIDDIR"/eval_*.pid; do
    [[ -f "$f" ]] || continue
    if kill -0 "$(cat "$f")" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

run_eval_last() {
  local tag="$1" ckpt="$2" gpu="$3" jobs="$4"
  local out="$EVAL_ROOT/${tag}_last"
  local logf="$LOGDIR/eval_${tag}_last.log"
  if eval_ready "$out"; then
    skip_stage "eval_${tag}_last" "report already at $out/reports"
    return 0
  fi
  if [[ -d "$out" ]] && ! eval_ready "$out"; then
    log "eval: clearing incomplete $out"
    rm -rf "$out"
  fi
  set_stage "eval_${tag}_last"
  [[ -f "$ckpt" ]] || fail "missing ckpt for eval: $ckpt"
  mkdir -p "$out"
  log "eval: tag=$tag kind=last ckpt=$ckpt out=$out jobs=$jobs gpu=$gpu"
  cd "$EVAL_PRIORITY"
  local rc=0
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$PY" -u eval_pipeline.py \
    --policies plant2 \
    --model-paths "plant2:$ckpt" \
    --manifest "$TEST_MANIFEST" \
    --scenes-root "$SCENES" \
    --out-dir "$out" \
    --plant2-action-mode pid \
    --jobs "$jobs" \
    --backends sumo \
    2>&1 | tee "$logf" || rc=$?

  if [[ "$rc" -ne 0 ]] || ! eval_ready "$out"; then
    if [[ "$jobs" -gt 1 ]]; then
      log "eval flaky/failed with jobs=$jobs; retrying jobs=1"
      rm -rf "$out"
      mkdir -p "$out"
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PY" -u eval_pipeline.py \
        --policies plant2 \
        --model-paths "plant2:$ckpt" \
        --manifest "$TEST_MANIFEST" \
        --scenes-root "$SCENES" \
        --out-dir "$out" \
        --plant2-action-mode pid \
        --jobs 1 \
        --backends sumo \
        2>&1 | tee -a "$logf" \
        || fail "eval ${tag}_last failed; see $logf"
    else
      fail "eval ${tag}_last failed; see $logf"
    fi
  fi
  eval_ready "$out" || fail "eval ${tag}_last produced no report"
  ok_stage "eval_${tag}_last"
}

update_results() {
  WORK="$WORK" RESULTS_MD="$RESULTS_MD" "$PY" "$WORK/update_results.py" \
    2>&1 | tee -a "$MASTER_LOG" \
    || log "WARN: update_results.py failed (non-fatal)"
}

# last-only eval for one experiment on an acquired eval GPU.
eval_tag_last() {
  local tag="$1"
  local epochs ckpt_dir last_ckpt
  local lease gpu fd jobs n_active
  epochs="$(epochs_for_tag "$tag")"
  ckpt_dir="$(ckpt_dir_for "$tag")"
  last_ckpt="$(find_last_ckpt "$ckpt_dir")"
  [[ -n "$last_ckpt" ]] || fail "no last_ft for $tag"

  write_train_meta "$tag" "$LR_VAL" "$epochs" "$ckpt_dir" "$LOGDIR/train_${tag}.log" || true

  lease="$(acquire_eval_gpu)"
  gpu="${lease%%:*}"
  fd="${lease##*:}"
  n_active="$(count_active_evals)"
  if [[ "$n_active" -ge 1 ]]; then
    jobs="$JOBS_EVAL_PARALLEL"
  else
    jobs="$JOBS_EVAL_SOLO"
  fi

  echo $$ > "$PIDDIR/eval_${tag}.pid"
  log "eval_tag_last: $tag on gpu=$gpu jobs=$jobs (active_evals=$n_active)"
  local rc=0
  run_eval_last "$tag" "$last_ckpt" "$gpu" "$jobs" || rc=$?
  update_results
  rm -f "$PIDDIR/eval_${tag}.pid"
  release_eval_gpu "$fd"
  return "$rc"
}

stage_train_one_bg() {
  local tag="$1" epochs="$2" gpu="$3"
  local addon ckpt_dir train_log
  addon="$(addon_for "$tag")"
  ckpt_dir="$(ckpt_dir_for "$tag")"
  train_log="$LOGDIR/train_${tag}.log"

  if train_is_complete "$tag"; then
    skip_stage "train_${tag}" "already complete (STATUS OK or last_ft epoch>=$((epochs - 1)))"
    write_train_meta "$tag" "$LR_VAL" "$epochs" "$ckpt_dir" "$train_log" || true
    echo "SKIP" > "$PIDDIR/train_${tag}.result"
    return 0
  fi

  if [[ -d "$ckpt_dir" ]] && ! train_ckpt_complete "$ckpt_dir" "$epochs"; then
    log "train_${tag}: incomplete ckpt dir (no Lightning resume in lit_finetune); clearing $ckpt_dir"
    rm -rf "$ckpt_dir"
  fi

  set_stage "train_${tag}"
  log "train: lr=$LR_VAL epochs=$epochs addon=$addon gpu=$gpu (background)"
  mkdir -p "$ckpt_dir"
  (
    set -o pipefail
    cd "$PIPELINE"
    "$PY" -u train/run_plant2_finetune.py \
      --split "$SPLIT" \
      --learning-rate "$LR_VAL" \
      --checkpoint-addon "$addon" \
      --cuda-device "$gpu" \
      --ds-local "$DS_LOCAL" \
      --cache-size-gb "$CACHE_SIZE_GB" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-epochs "$epochs" \
      --ckpt-every-n-epochs "$CKPT_EVERY" \
      --augment \
      --no-filter-routes \
      --resume-ckpt "$CKPT0" \
      --wandb-mode offline \
      --python "$PY" \
      --hydra-override "user.working_dir=$TRB_ROOT/plant2" \
      --log "$train_log" \
      2>&1 | tee -a "$train_log"
  ) &
  local pid=$!
  echo "$pid" > "$PIDDIR/train_${tag}.pid"
  log "train_${tag}: launched pid=$pid gpu=$gpu log=$train_log"
}

wait_train_and_record() {
  local tag="$1" epochs="$2"
  local pid_file="$PIDDIR/train_${tag}.pid"
  local result_file="$PIDDIR/train_${tag}.result"
  local ckpt_dir train_log last pid rc

  if [[ -f "$result_file" ]] && [[ "$(cat "$result_file")" == "SKIP" ]]; then
    return 0
  fi
  [[ -f "$pid_file" ]] || fail "train_${tag}: missing pid file"
  pid="$(cat "$pid_file")"
  set +e
  wait "$pid"
  rc=$?
  set -e
  ckpt_dir="$(ckpt_dir_for "$tag")"
  train_log="$LOGDIR/train_${tag}.log"
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL" > "$result_file"
    fail "train_${tag} exited rc=$rc; see $train_log"
  fi
  last="$(find_last_ckpt "$ckpt_dir")"
  [[ -n "$last" ]] || fail "train_${tag} produced no last_ft_*.ckpt under $ckpt_dir"
  train_ckpt_complete "$ckpt_dir" "$epochs" \
    || fail "train_${tag}: last_ft present but epoch incomplete; see $train_log"
  write_train_meta "$tag" "$LR_VAL" "$epochs" "$ckpt_dir" "$train_log" \
    || fail "train_${tag}: could not write train_meta"
  log "train_${tag}: last=$last best=$(find_best_ckpt "$ckpt_dir")"
  ok_stage "train_${tag}"
  echo "OK" > "$result_file"
  rm -f "$pid_file"
}

# ---------- run ----------
stage_prefill
update_results

set_stage "parallel_trains"
declare -a EVAL_BG_PIDS=()
declare -a TRAIN_TAGS=()
declare -a TRAIN_EPOCHS=()

for entry in "${EXP_GRID[@]}"; do
  tag="${entry%%:*}"
  epochs="${entry#*:}"
  gpu="$(gpu_for_tag "$tag")"
  log "=== experiment $tag lr=$LR_VAL epochs=$epochs gpu=$gpu ==="
  TRAIN_TAGS+=("$tag")
  TRAIN_EPOCHS+=("$epochs")

  if train_is_complete "$tag"; then
    stage_train_one_bg "$tag" "$epochs" "$gpu"
    if ! eval_ready "$EVAL_ROOT/${tag}_last"; then
      log "eval: scheduling immediate last-eval for completed $tag"
      (
        eval_tag_last "$tag"
      ) &
      EVAL_BG_PIDS+=("$!")
      echo "$!" > "$PIDDIR/evalwrap_${tag}.pid"
    else
      skip_stage "eval_${tag}_last" "already ready"
    fi
  else
    stage_train_one_bg "$tag" "$epochs" "$gpu"
  fi
done

for i in "${!TRAIN_TAGS[@]}"; do
  tag="${TRAIN_TAGS[$i]}"
  epochs="${TRAIN_EPOCHS[$i]}"
  result_file="$PIDDIR/train_${tag}.result"
  if [[ -f "$result_file" ]] && [[ "$(cat "$result_file")" == "SKIP" ]]; then
    log "train_${tag}: was already complete; eval handled separately"
    continue
  fi
  log "waiting for train_${tag} ..."
  wait_train_and_record "$tag" "$epochs"
  log "train_${tag} done — launching last-only eval"
  (
    eval_tag_last "$tag"
  ) &
  EVAL_BG_PIDS+=("$!")
  echo "$!" > "$PIDDIR/evalwrap_${tag}.pid"
done

set_stage "parallel_evals"
log "waiting for ${#EVAL_BG_PIDS[@]} eval worker(s) ..."
eval_fail=0
for pid in "${EVAL_BG_PIDS[@]:-}"; do
  [[ -n "$pid" ]] || continue
  set +e
  wait "$pid"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "WARN: eval worker pid=$pid exited rc=$rc"
    eval_fail=1
  fi
done

for entry in "${EXP_GRID[@]}"; do
  tag="${entry%%:*}"
  if ! eval_ready "$EVAL_ROOT/${tag}_last"; then
    log "final-pass last-eval for $tag"
    eval_tag_last "$tag" || eval_fail=1
  fi
done

update_results
if [[ "$eval_fail" -ne 0 ]]; then
  fail "one or more evals failed; see $LOGDIR/eval_*.log"
fi

log "=== EP20/EP30 GRID COMPLETE (parallel, last-only eval) ==="
echo "SUCCESS $(ts)" > "$STATUS"
echo "results=$RESULTS_MD" >> "$STATUS"
echo "done" > "$CURRENT_STAGE"
update_results
