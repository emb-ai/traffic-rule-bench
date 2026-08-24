#!/usr/bin/env bash
# STOP / priority_bench end-to-end: dump → split → prefill → train → eval+GIFs
# Designed to run under nohup; each stage logs under $WORK/logs/.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
ZINK_PER_SIGN="/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench/scripts/per_sign_bench"
PRIORITY="$ZINK_PER_SIGN/priority_bench"
TRAIN_TRAJ="$PRIORITY/data/stop/trajectories/debug_train_400"
TEST_MANIFEST="$PRIORITY/data/stop/output/ts_test/real_manifest.jsonl"
SCENES="$PRIORITY/data/stop/scenes"
WORK="$TRB_ROOT/plant2_stop_pipeline_debug400"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
PY="${PY:-/home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python}"
CKPT0="${CKPT0:-/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt}"

export TRB_ROOT
export SHEPELEV="$WORK"
export PIPELINE_DIR="$PIPELINE"
export PLAN_T="$TRB_ROOT/plant2/PlanT"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export PER_SIGN_COMPLIANT_NPC=1
export WANDB_MODE="${WANDB_MODE:-offline}"
# Prefer local metadrive (rebbutle-wip) over env site-packages path
export PYTHONPATH="$TRB_ROOT/metadrive:$TRB_ROOT/pdd-bench:$TRB_ROOT/pdd-bench/scripts/per_sign_bench:${PYTHONPATH:-}"

DUMP_TRAIN="$WORK/plant2_l1_stop_train"
DUMP_TEST="$WORK/plant2_l1_stop_test"
TEST_TRAJ="$WORK/trajectories_test"
SPLIT="$WORK/plant2_l1_stop_split"
DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_stop_debug400}"
CACHE_SIZE_GB="${CACHE_SIZE_GB:-400}"
ADDON="${CHECKPOINT_ADDON:-stop_debug400_lr3e4_ep20}"
N_DUMP_WORKERS="${N_DUMP_WORKERS:-16}"
N_VAL="${N_VAL:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-3e-4}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_EVAL="${GPU_EVAL:-1}"
N_GIFS="${N_GIFS:-5}"

LOGDIR="$WORK/logs"
mkdir -p "$LOGDIR" "$WORK" "$PLAN_T/checkpoints_ft"
MASTER_LOG="$LOGDIR/pipeline_master.log"
STATUS="$WORK/STATUS.txt"

ts() { date -Is; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }
fail() { log "FAIL: $*"; echo "FAILED: $*" > "$STATUS"; exit 1; }
ok_stage() { log "OK stage=$1"; echo "OK $1 $(ts)" >> "$STATUS"; }

echo "RUNNING $(ts)" > "$STATUS"
log "=== STOP pipeline start ==="
log "WORK=$WORK PY=$PY CKPT0=$CKPT0"
log "TRAIN_TRAJ=$TRAIN_TRAJ"
log "TEST_MANIFEST=$TEST_MANIFEST"

[[ -x "$PY" ]] || fail "python missing: $PY"
[[ -f "$CKPT0" ]] || fail "ckpt missing: $CKPT0"
[[ -d "$TRAIN_TRAJ" ]] || fail "train traj missing: $TRAIN_TRAJ"
[[ -f "$TEST_MANIFEST" ]] || fail "test manifest missing: $TEST_MANIFEST"
[[ -d "$SCENES" ]] || fail "scenes missing: $SCENES"

# ---------- 1) select experts on train trajectories ----------
stage_select_train() {
  local out="$TRAIN_TRAJ/experts"
  local top1="$out/experts_scene_uid_top1.jsonl"
  if [[ -s "$top1" ]]; then
    log "select_train: reuse existing $top1 ($(wc -l < "$top1") lines)"
    return 0
  fi
  log "select_train: selecting experts → $out"
  cd "$PRIORITY/collect_trajectories"
  "$PY" -u select_experts_coverage.py \
    --root "$TRAIN_TRAJ" \
    --catalog "$TRAIN_TRAJ/catalog.jsonl" \
    --signs 2.5 \
    --horizon 1500 \
    --out-dir "$out" \
    2>&1 | tee "$LOGDIR/01_select_train.log"
  [[ -s "$top1" ]] || fail "select_train produced no $top1"
  log "select_train: $(wc -l < "$top1") experts"
}

# ---------- 2) dump train plant2 L1 (sharded) ----------
stage_dump_train() {
  local experts="$TRAIN_TRAJ/experts/experts_scene_uid_top1.jsonl"
  local n
  n=$(wc -l < "$experts")
  mkdir -p "$DUMP_TRAIN/logs"
  log "dump_train: n=$n → $DUMP_TRAIN workers=$N_DUMP_WORKERS"
  local shard=$(( (n + N_DUMP_WORKERS - 1) / N_DUMP_WORKERS ))
  [[ "$shard" -lt 1 ]] && shard=1
  local pids=() fails=0
  local i=0 start=0
  while [[ "$start" -lt "$n" ]]; do
    local count=$shard
    if (( start + count > n )); then count=$(( n - start )); fi
    local slog="$DUMP_TRAIN/logs/shard_${i}_s${start}_c${count}.log"
    (
      cd "$TRB_ROOT/pdd-bench/scripts/per_sign_bench"
      "$PY" -u expert_replay_for_plant2.py \
        --experts "$experts" \
        --scenes-root "$SCENES" \
        --save-plant2-dir "$DUMP_TRAIN" \
        --start "$start" \
        --count "$count" \
        --max-steps 1500 \
        --backends sumo
    ) >"$slog" 2>&1 &
    pids+=($!)
    log "dump_train: spawn shard=$i start=$start count=$count pid=${pids[-1]} log=$slog"
    start=$(( start + count ))
    i=$(( i + 1 ))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fails=$(( fails + 1 ))
      log "dump_train: worker pid=$pid FAILED"
    fi
  done
  local n_routes
  n_routes=$(find "$DUMP_TRAIN/data" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  log "dump_train: routes=$n_routes fails=$fails"
  [[ "$n_routes" -gt 0 ]] || fail "dump_train produced no routes"
  # tolerate a few shard failures if most routes exist
  if [[ "$fails" -gt 0 && "$n_routes" -lt $(( n / 2 )) ]]; then
    fail "dump_train too many failures ($fails) with only $n_routes routes"
  fi
}

# ---------- 3) collect + select + dump TEST from manifest ----------
stage_collect_test() {
  mkdir -p "$TEST_TRAJ"
  log "collect_test: manifest=$TEST_MANIFEST → $TEST_TRAJ"
  cd "$PRIORITY/collect_trajectories"
  "$PY" -u expert_replay_priority.py \
    --sign stop \
    --manifest "$TEST_MANIFEST" \
    --scenes-root "$SCENES" \
    --policy comprehensive_rule_expert \
    --output-dir "$TEST_TRAJ" \
    --max-steps 1500 \
    --resume \
    2>&1 | tee "$LOGDIR/03_collect_test.log"
}

stage_select_test() {
  local out="$TEST_TRAJ/experts"
  local top1="$out/experts_scene_uid_top1.jsonl"
  log "select_test: → $out"
  cd "$PRIORITY/collect_trajectories"
  "$PY" -u select_experts_coverage.py \
    --root "$TEST_TRAJ" \
    --manifest "$TEST_MANIFEST" \
    --signs 2.5 \
    --horizon 1500 \
    --min-join-rate 0.0 \
    --out-dir "$out" \
    2>&1 | tee "$LOGDIR/03b_select_test.log"
  [[ -s "$top1" ]] || fail "select_test produced no $top1"
}

stage_dump_test() {
  local experts="$TEST_TRAJ/experts/experts_scene_uid_top1.jsonl"
  mkdir -p "$DUMP_TEST/logs"
  log "dump_test: → $DUMP_TEST"
  cd "$TRB_ROOT/pdd-bench/scripts/per_sign_bench"
  "$PY" -u expert_replay_for_plant2.py \
    --experts "$experts" \
    --scenes-root "$SCENES" \
    --save-plant2-dir "$DUMP_TEST" \
    --max-steps 1500 \
    --backends sumo \
    2>&1 | tee "$LOGDIR/03c_dump_test.log"
  local n_routes
  n_routes=$(find "$DUMP_TEST/data" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  log "dump_test: routes=$n_routes"
  [[ "$n_routes" -gt 0 ]] || fail "dump_test produced no routes"
}

# ---------- 4) split train dump into train/val (50 val) ----------
stage_split() {
  log "split: $DUMP_TRAIN → $SPLIT (n_val=$N_VAL)"
  DUMP_TRAIN="$DUMP_TRAIN" SPLIT="$SPLIT" N_VAL="$N_VAL" \
    "$PY" -u - <<'PY' 2>&1 | tee "$LOGDIR/04_split.log"
import json, os, random, shutil
from pathlib import Path

dump = Path(os.environ["DUMP_TRAIN"])
out = Path(os.environ["SPLIT"])
n_val = int(os.environ["N_VAL"])
seed = 42

src = dump / "data"
routes = sorted([p for p in src.iterdir() if p.is_dir()])
if len(routes) < n_val + 1:
    raise SystemExit(f"need >{n_val} routes, got {len(routes)}")
rng = random.Random(seed)
rng.shuffle(routes)
val_routes = routes[:n_val]
train_routes = routes[n_val:]

if out.exists():
    shutil.rmtree(out)
for split, items in (("train", train_routes), ("val", val_routes)):
    data = out / split / "data"
    data.mkdir(parents=True)
    slurm = out / split / "slurm" / "run_files" / "logs"
    slurm.mkdir(parents=True)
    (slurm / "qsub_out2025_07.log").write_text("dummy\n")
    for p in items:
        dst = data / p.name
        for root, dirs, files in os.walk(p):
            rel = Path(root).relative_to(p)
            (dst / rel).mkdir(parents=True, exist_ok=True)
            for fn in files:
                s = Path(root) / fn
                d = dst / rel / fn
                try:
                    os.link(s, d)
                except OSError:
                    shutil.copy2(s, d)

meta = {
    "seed": seed,
    "n_val_requested": n_val,
    "source": str(dump),
    "per_sign": {"2.5": {"n": len(routes), "n_train": len(train_routes), "n_val": len(val_routes), "mode": "fixed50"}},
    "train_counts": {"2.5": len(train_routes)},
    "val": {"2.5": [p.name for p in val_routes]},
    "train": {"2.5": [p.name for p in train_routes]},
}
(out / "split_meta.json").write_text(json.dumps(meta, indent=2))
print(f"DONE train={len(train_routes)} val={len(val_routes)} out={out}")
PY
}

# ---------- 5) prefill diskcache on split ----------
stage_prefill() {
  mkdir -p "$DS_LOCAL"
  log "prefill: train+val → $DS_LOCAL"
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
    2>&1 | tee "$LOGDIR/05_prefill.log"
}

# ---------- 6) train ----------
stage_train() {
  log "train: lr=$LR epochs=$MAX_EPOCHS addon=$ADDON gpu=$GPU_TRAIN"
  cd "$PIPELINE"
  "$PY" -u train/run_plant2_finetune.py \
    --split "$SPLIT" \
    --learning-rate "$LR" \
    --checkpoint-addon "$ADDON" \
    --cuda-device "$GPU_TRAIN" \
    --ds-local "$DS_LOCAL" \
    --cache-size-gb "$CACHE_SIZE_GB" \
    --batch-size 512 \
    --num-workers 4 \
    --max-epochs "$MAX_EPOCHS" \
    --ckpt-every-n-epochs 5 \
    --augment \
    --no-filter-routes \
    --resume-ckpt "$CKPT0" \
    --wandb-mode offline \
    --python "$PY" \
    --log "$LOGDIR/06_train.log" \
    2>&1 | tee -a "$LOGDIR/06_train.log"
}

# ---------- 7) eval on test + 5 GIFs ----------
stage_eval() {
  local ckpt_dir="$PLAN_T/checkpoints_ft/$ADDON"
  local ckpt
  ckpt=$(ls -1t "$ckpt_dir"/best_*.ckpt 2>/dev/null | head -1 || true)
  if [[ -z "$ckpt" ]]; then
    ckpt=$(ls -1t "$ckpt_dir"/epoch=*.ckpt 2>/dev/null | head -1 || true)
  fi
  if [[ -z "$ckpt" ]]; then
    ckpt=$(ls -1t "$ckpt_dir"/*.ckpt 2>/dev/null | head -1 || true)
  fi
  [[ -n "$ckpt" && -f "$ckpt" ]] || fail "no checkpoint under $ckpt_dir"
  log "eval: ckpt=$ckpt"
  local eval_out="$WORK/eval_test"
  mkdir -p "$eval_out"
  # Full test eval (no gifs for speed), then 5 GIF scenes
  cd "$PRIORITY"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" \
  "$PY" -u eval_pipeline.py \
    --policies plant2 \
    --model-paths "plant2:$ckpt" \
    --manifest "$TEST_MANIFEST" \
    --scenes-root "$SCENES" \
    --out-dir "$eval_out/full" \
    --jobs 8 \
    --backends sumo \
    2>&1 | tee "$LOGDIR/07_eval_full.log"

  CUDA_VISIBLE_DEVICES="$GPU_EVAL" \
  "$PY" -u eval_pipeline.py \
    --policies plant2 \
    --model-paths "plant2:$ckpt" \
    --manifest "$TEST_MANIFEST" \
    --scenes-root "$SCENES" \
    --out-dir "$eval_out/gifs5" \
    --max-scenes "$N_GIFS" \
    --save-gifs \
    --jobs 1 \
    --backends sumo \
    2>&1 | tee "$LOGDIR/07_eval_gifs.log"

  find "$eval_out/gifs5" -name '*.gif' 2>/dev/null | tee "$WORK/gif_paths.txt" | tee -a "$MASTER_LOG"
  echo "$ckpt" > "$WORK/final_ckpt.txt"
  log "eval done; gifs listed in $WORK/gif_paths.txt"
}

# ---- run ----
stage_select_train && ok_stage select_train
stage_dump_train && ok_stage dump_train
stage_collect_test && ok_stage collect_test
stage_select_test && ok_stage select_test
stage_dump_test && ok_stage dump_test
stage_split && ok_stage split
stage_prefill && ok_stage prefill
stage_train && ok_stage train
stage_eval && ok_stage eval

log "=== PIPELINE COMPLETE ==="
echo "SUCCESS $(ts)" > "$STATUS"
echo "ckpt=$(cat "$WORK/final_ckpt.txt")" >> "$STATUS"
echo "gifs=$WORK/gif_paths.txt" >> "$STATUS"
