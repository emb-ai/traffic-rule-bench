#!/usr/bin/env bash
# PlanT STOP class_emb — lr=3e-4 @ 40 epochs.
# Eval last_ft only; append metrics into ep10 RESULTS.md (fixed path).
#
# Reuses signfix dump/split + prefill. Does NOT re-dump, does NOT touch
# plant2_stop_pipeline_signfix/ or prior ep10/20/30 artifacts,
# does NOT write into the zinkovich tree.
#
# Architecture: shared class_emb + attr_emb. Pretrain CKPT0 has tok_emb;
# lit_finetune loads with strict=False.
#
# Designed for: setsid nohup bash run_ep40.sh >>logs/nohup.out 2>&1 &
# Skip-if-complete per stage so a crash mid-run can resume.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
STOP_DATA="${STOP_DATA:-$TRB_ROOT/stop_data}"
PRIORITY="$TRB_ROOT/pdd-bench/scripts/per_sign_bench/priority_bench"
SPLIT="${SPLIT:-$TRB_ROOT/plant2_stop_pipeline_signfix/plant2_l1_stop_split}"
SIGNFIX_WORK="$TRB_ROOT/plant2_stop_pipeline_signfix"
EP10_WORK="$TRB_ROOT/plant2_stop_pipeline_lrgrid_ep10_classemb"
EP20_30_WORK="$TRB_ROOT/plant2_stop_pipeline_lrgrid_ep20_30_classemb"
WORK="$TRB_ROOT/plant2_stop_pipeline_lrgrid_ep40_classemb"
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
JOBS_EVAL="${JOBS_EVAL:-8}"

# Fixed LR; single epoch budget. Avoid busy GPUs 0/1/2.
LR_VAL="${LR_VAL:-3e-4}"
GPU_TRAIN="${GPU_TRAIN:-3}"
GPU_EVAL="${GPU_EVAL:-5}"

TAG="lr3e4_ep40"
EPOCHS=40
ADDON="stop_classemb_${TAG}"

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
export JOBS_EVAL LOGDIR TEST_MANIFEST SCENES CKPT0
export STOP_DATA TRAIN_META EVAL_ROOT BATCH_SIZE NUM_WORKERS
export PY PLAN_T PIPELINE TRB_ROOT RESULTS_MD LR_VAL EP10_WORK EP20_30_WORK

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
    echo "RUNNING $(ts) mode=ep40 lr=$LR_VAL epochs=$EPOCHS"
    if [[ -f "$STATUS" ]]; then
      grep -E '^(OK|SKIP) ' "$STATUS" || true
    fi
  } > "$STATUS.tmp"
  mv -f "$STATUS.tmp" "$STATUS"
fi
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "=== class_emb ep40 preflight (--check-only) ==="
else
  log "=== PlanT STOP class_emb lr=${LR_VAL} ep40 start ==="
fi
log "WORK=$WORK PY=$PY CKPT0=$CKPT0"
log "RESULTS_MD=$RESULTS_MD (fixed; prior rows preserved via merge)"
log "SPLIT=$SPLIT DS_LOCAL=$DS_LOCAL"
log "TEST_MANIFEST=$TEST_MANIFEST SCENES=$SCENES"
log "GPU map: train=$GPU_TRAIN eval=$GPU_EVAL"
log "TAG=$TAG ADDON=$ADDON EPOCHS=$EPOCHS"

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

ckpt_dir_for() {
  echo "$PLAN_T/checkpoints_ft/$ADDON"
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
  local ckpt_dir
  ckpt_dir="$(ckpt_dir_for)"
  if status_has_ok "train_${TAG}"; then
    return 0
  fi
  train_ckpt_complete "$ckpt_dir" "$EPOCHS"
}

write_train_meta() {
  local ckpt_dir train_log
  ckpt_dir="$(ckpt_dir_for)"
  train_log="$LOGDIR/train_${TAG}.log"
  export EXP_TAG="$TAG" LR_VAL_META="$LR_VAL" MAX_EPOCHS="$EPOCHS" \
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

update_results() {
  WORK="$WORK" RESULTS_MD="$RESULTS_MD" EP20_30_WORK="$EP20_30_WORK" \
    "$PY" "$WORK/update_results.py" \
    2>&1 | tee -a "$MASTER_LOG" \
    || log "WARN: update_results.py failed (non-fatal)"
}

run_eval_last() {
  local ckpt="$1" gpu="$2" jobs="$3"
  local out="$EVAL_ROOT/${TAG}_last"
  local logf="$LOGDIR/eval_${TAG}_last.log"
  if eval_ready "$out"; then
    skip_stage "eval_${TAG}_last" "report already at $out/reports"
    return 0
  fi
  if [[ -d "$out" ]] && ! eval_ready "$out"; then
    log "eval: clearing incomplete $out"
    rm -rf "$out"
  fi
  set_stage "eval_${TAG}_last"
  [[ -f "$ckpt" ]] || fail "missing ckpt for eval: $ckpt"
  mkdir -p "$out"
  log "eval: tag=$TAG kind=last ckpt=$ckpt out=$out jobs=$jobs gpu=$gpu"
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
        || fail "eval ${TAG}_last failed; see $logf"
    else
      fail "eval ${TAG}_last failed; see $logf"
    fi
  fi
  eval_ready "$out" || fail "eval ${TAG}_last produced no report"
  ok_stage "eval_${TAG}_last"
}

stage_train() {
  local ckpt_dir train_log
  ckpt_dir="$(ckpt_dir_for)"
  train_log="$LOGDIR/train_${TAG}.log"

  if train_is_complete; then
    skip_stage "train_${TAG}" "already complete (STATUS OK or last_ft epoch>=$((EPOCHS - 1)))"
    write_train_meta || true
    return 0
  fi

  if [[ -d "$ckpt_dir" ]] && ! train_ckpt_complete "$ckpt_dir" "$EPOCHS"; then
    log "train_${TAG}: incomplete ckpt dir (no Lightning resume in lit_finetune); clearing $ckpt_dir"
    rm -rf "$ckpt_dir"
  fi

  set_stage "train_${TAG}"
  log "train: lr=$LR_VAL epochs=$EPOCHS addon=$ADDON gpu=$GPU_TRAIN"
  mkdir -p "$ckpt_dir"
  (
    set -o pipefail
    cd "$PIPELINE"
    "$PY" -u train/run_plant2_finetune.py \
      --split "$SPLIT" \
      --learning-rate "$LR_VAL" \
      --checkpoint-addon "$ADDON" \
      --cuda-device "$GPU_TRAIN" \
      --ds-local "$DS_LOCAL" \
      --cache-size-gb "$CACHE_SIZE_GB" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-epochs "$EPOCHS" \
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
  echo "$pid" > "$PIDDIR/train_${TAG}.pid"
  log "train_${TAG}: launched pid=$pid gpu=$GPU_TRAIN log=$train_log"

  set +e
  wait "$pid"
  local rc=$?
  set -e
  rm -f "$PIDDIR/train_${TAG}.pid"

  if [[ "$rc" -ne 0 ]]; then
    fail "train_${TAG} exited rc=$rc; see $train_log"
  fi
  local last
  last="$(find_last_ckpt "$ckpt_dir")"
  [[ -n "$last" ]] || fail "train_${TAG} produced no last_ft_*.ckpt under $ckpt_dir"
  train_ckpt_complete "$ckpt_dir" "$EPOCHS" \
    || fail "train_${TAG}: last_ft present but epoch incomplete; see $train_log"
  write_train_meta || fail "train_${TAG}: could not write train_meta"
  log "train_${TAG}: last=$last best=$(find_best_ckpt "$ckpt_dir")"
  ok_stage "train_${TAG}"
}

# ---------- run ----------
stage_prefill
update_results

stage_train

set_stage "eval_${TAG}_last"
last_ckpt="$(find_last_ckpt "$(ckpt_dir_for)")"
[[ -n "$last_ckpt" ]] || fail "no last_ft for $TAG"
write_train_meta || true
run_eval_last "$last_ckpt" "$GPU_EVAL" "$JOBS_EVAL"
update_results

log "=== EP40 COMPLETE (last-only eval) ==="
echo "SUCCESS $(ts)" > "$STATUS"
echo "results=$RESULTS_MD" >> "$STATUS"
echo "done" > "$CURRENT_STAGE"
update_results
