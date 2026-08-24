#!/usr/bin/env bash
# Continue STOP/priority_bench pipeline from an already completed train dump.
# Runs split -> prefill -> train -> eval(test + 5 GIFs), skipping completed stages.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
ZINK_PER_SIGN="/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench/scripts/per_sign_bench"
PRIORITY="$ZINK_PER_SIGN/priority_bench"
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
export PYTHONPATH="$TRB_ROOT/metadrive:$TRB_ROOT/pdd-bench:$TRB_ROOT/pdd-bench/scripts/per_sign_bench:${PYTHONPATH:-}"

DUMP_TRAIN="$WORK/plant2_l1_stop_train"
SPLIT="$WORK/plant2_l1_stop_split"
DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_stop_debug400}"
CACHE_SIZE_GB="${CACHE_SIZE_GB:-400}"
ADDON="${CHECKPOINT_ADDON:-stop_debug400_lr3e4_ep20}"
N_VAL="${N_VAL:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-3e-4}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_EVAL="${GPU_EVAL:-1}"
N_GIFS="${N_GIFS:-5}"
CKPT_DIR="$PLAN_T/checkpoints_ft/$ADDON"
GIF_PATHS="$WORK/gif_paths.txt"

LOGDIR="$WORK/logs"
MASTER_LOG="$LOGDIR/continue_master.log"
STATUS="$WORK/CONTINUE_STATUS.txt"
CURRENT_STAGE="$WORK/CONTINUE_CURRENT_STAGE.txt"

# Python heredocs read these via os.environ; keep them exported so skip/validate
# checks do not KeyError when a stage prefix is missing.
export WORK DUMP_TRAIN SPLIT DS_LOCAL CACHE_SIZE_GB ADDON N_VAL MAX_EPOCHS LR
export GPU_TRAIN GPU_EVAL N_GIFS LOGDIR TEST_MANIFEST SCENES CKPT0 CKPT_DIR GIF_PATHS

mkdir -p "$LOGDIR" "$WORK" "$PLAN_T/checkpoints_ft"

ts() { date -Is; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }
fail() { log "FAIL: $*"; echo "FAILED: $*" > "$STATUS"; exit 1; }
set_stage() { echo "$1" > "$CURRENT_STAGE"; log "stage=$1"; }
ok_stage() { log "OK stage=$1"; echo "OK $1 $(ts)" >> "$STATUS"; }
skip_stage() { log "SKIP stage=$1 reason=$2"; echo "SKIP $1 $(ts) $2" >> "$STATUS"; }

echo "RUNNING $(ts)" > "$STATUS"
log "=== CONTINUE STOP pipeline from train dump ==="
log "WORK=$WORK PY=$PY CKPT0=$CKPT0"
log "DUMP_TRAIN=$DUMP_TRAIN"
log "TEST_MANIFEST=$TEST_MANIFEST"
log "SCENES=$SCENES"

[[ -x "$PY" ]] || fail "python missing: $PY"
[[ -f "$CKPT0" ]] || fail "ckpt missing: $CKPT0"
[[ -d "$DUMP_TRAIN/data" ]] || fail "train dump missing: $DUMP_TRAIN/data"
[[ -f "$TEST_MANIFEST" ]] || fail "test manifest missing: $TEST_MANIFEST"
[[ -d "$SCENES" ]] || fail "scenes missing: $SCENES"

split_ready() {
  [[ -f "$SPLIT/split_meta.json" ]] || return 1
  SPLIT="$SPLIT" N_VAL="$N_VAL" "$PY" - <<'PY'
import json, os
from pathlib import Path
split = Path(os.environ["SPLIT"])
n_val = int(os.environ["N_VAL"])
meta = json.loads((split / "split_meta.json").read_text())
assert meta["per_sign"]["2.5"]["n_val"] == n_val, meta["per_sign"]["2.5"]
assert (split / "train" / "data").is_dir()
assert (split / "val" / "data").is_dir()
assert any((split / "train" / "data").iterdir())
assert len([p for p in (split / "val" / "data").iterdir() if p.is_dir()]) == n_val
PY
}

prefill_ready() {
  [[ -d "$DS_LOCAL" ]] || return 1
  [[ -f "$DS_LOCAL/cache.db" || -f "$DS_LOCAL/cache.db-wal" || -f "$DS_LOCAL/cache.db-shm" ]] || return 1
}

latest_ckpt() {
  CKPT_DIR="${CKPT_DIR:-$PLAN_T/checkpoints_ft/$ADDON}" "$PY" - <<'PY'
import os
from pathlib import Path
ckpt_dir = Path(os.environ["CKPT_DIR"])
if not ckpt_dir.is_dir():
    raise SystemExit(1)
candidates = []
for pattern in ("best_*.ckpt", "epoch=*.ckpt", "*.ckpt"):
    candidates.extend(ckpt_dir.glob(pattern))
if not candidates:
    raise SystemExit(1)
latest = max({p.resolve() for p in candidates}, key=lambda p: p.stat().st_mtime)
print(latest)
PY
}

train_ready() {
  latest_ckpt >/dev/null 2>&1
}

eval_ready() {
  [[ -f "$GIF_PATHS" ]] || return 1
  GIF_PATHS="$GIF_PATHS" N_GIFS="$N_GIFS" "$PY" - <<'PY'
import os
from pathlib import Path
gif_paths = Path(os.environ["GIF_PATHS"])
n_gifs = int(os.environ["N_GIFS"])
paths = [line.strip() for line in gif_paths.read_text().splitlines() if line.strip()]
assert len(paths) >= n_gifs
for path in paths[:n_gifs]:
    assert Path(path).is_file()
PY
}

stage_split() {
  if split_ready; then
    skip_stage split "existing split with 50 val"
    return 0
  fi
  set_stage split
  log "split: $DUMP_TRAIN -> $SPLIT (n_val=$N_VAL)"
  DUMP_TRAIN="$DUMP_TRAIN" SPLIT="$SPLIT" N_VAL="$N_VAL" \
    "$PY" -u - <<'PY' 2>&1 | tee "$LOGDIR/08_split_continue.log"
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
  split_ready || fail "split validation failed"
  ok_stage split
}

stage_prefill() {
  if prefill_ready; then
    skip_stage prefill "existing diskcache detected"
    return 0
  fi
  set_stage prefill
  mkdir -p "$DS_LOCAL"
  log "prefill: train+val -> $DS_LOCAL"
  cd "$PIPELINE"
  "$PY" -u data/prefill_diskcache.py parallel \
    --ds "$SPLIT/train" \
    --ds-val "$SPLIT/val" \
    --ds-local "$DS_LOCAL" \
    --cache-size-gb "$CACHE_SIZE_GB" \
    --augment \
    --max-workers 16 \
    --python "$PY" \
    --log-dir "$LOGDIR/prefill_continue" \
    2>&1 | tee "$LOGDIR/09_prefill_continue.log"
  prefill_ready || fail "prefill validation failed"
  ok_stage prefill
}

stage_train() {
  if train_ready; then
    skip_stage train "checkpoint already exists"
    return 0
  fi
  set_stage train
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
    --hydra-override "user.working_dir=$TRB_ROOT/plant2" \
    --log "$LOGDIR/10_train_continue.log" \
    2>&1 | tee -a "$LOGDIR/10_train_continue.log"
  latest_ckpt > "$WORK/final_ckpt.txt" || fail "train produced no checkpoint"
  ok_stage train
}

stage_eval() {
  if eval_ready; then
    skip_stage eval "existing gif outputs detected"
    return 0
  fi
  set_stage eval
  local ckpt
  ckpt="$(latest_ckpt)" || fail "no checkpoint under $PLAN_T/checkpoints_ft/$ADDON"
  log "eval: ckpt=$ckpt"
  local eval_out="$WORK/eval_test"
  mkdir -p "$eval_out"
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
    2>&1 | tee "$LOGDIR/11_eval_full_continue.log"

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
    2>&1 | tee "$LOGDIR/11_eval_gifs_continue.log"

  EVAL_OUT="$eval_out/gifs5" GIF_LIST="$GIF_PATHS" "$PY" - <<'PY'
import os
from pathlib import Path
out = Path(os.environ["EVAL_OUT"])
gifs = sorted(str(p.resolve()) for p in out.rglob("*.gif"))
Path(os.environ["GIF_LIST"]).write_text("\n".join(gifs) + ("\n" if gifs else ""))
print("\n".join(gifs))
PY
  echo "$ckpt" > "$WORK/final_ckpt.txt"
  eval_ready || fail "eval did not produce $N_GIFS gifs"
  log "eval done; gifs listed in $WORK/gif_paths.txt"
  ok_stage eval
}

stage_split
stage_prefill
stage_train
stage_eval

log "=== CONTINUATION COMPLETE ==="
echo "SUCCESS $(ts)" > "$STATUS"
echo "ckpt=$(cat "$WORK/final_ckpt.txt")" >> "$STATUS"
echo "gifs=$WORK/gif_paths.txt" >> "$STATUS"
echo "current_stage=done" > "$CURRENT_STAGE"
