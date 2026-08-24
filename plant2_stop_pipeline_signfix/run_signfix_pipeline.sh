#!/usr/bin/env bash
# STOP / priority_bench end-to-end after the recorded-sign-placement fix:
#   dump_train -> dump_test -> split -> prefill -> train -> eval(test + 5 GIFs)
#
# Fresh work dir so plant2_stop_pipeline_debug400/ stays intact for comparison.
# Designed to run under nohup; every stage logs under $WORK/logs/ and is
# skip-if-already-complete.
set -euo pipefail

TRB_ROOT="/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"
# Nothing here reads the zinkovich tree any more. It was restructured on
# 2026-08-20 (pdd-bench/scripts/per_sign_bench/priority_bench -> pdd-bench/
# sign_bench) and data/stop/scenes/junc_* became dangling symlinks into the
# deleted path, which killed eval. Data now comes from $STOP_DATA (vendored by
# vendor_stop_data.py, provenance in $STOP_DATA/README.md) and code from
# $PRIORITY, the belyaev priority_bench that carries the sign-restore fix.
STOP_DATA="${STOP_DATA:-$TRB_ROOT/stop_data}"
PRIORITY="$TRB_ROOT/pdd-bench/scripts/per_sign_bench/priority_bench"
TRAIN_TRAJ="$STOP_DATA/trajectories/debug_train_400"
TRAIN_EXPERTS="$TRAIN_TRAJ/experts/experts_scene_uid_top1.jsonl"
TEST_MANIFEST="$STOP_DATA/output/ts_test/real_manifest.jsonl"
SCENES="$STOP_DATA/scenes"
WORK="$TRB_ROOT/plant2_stop_pipeline_signfix"
PIPELINE="$TRB_ROOT/scripts/plant2_ft_pipeline"
PY="${PY:-/home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python}"
CKPT0="${CKPT0:-$STOP_DATA/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt}"

export TRB_ROOT
export SHEPELEV="$WORK"
export PIPELINE_DIR="$PIPELINE"
export PLAN_T="$TRB_ROOT/plant2/PlanT"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export PER_SIGN_COMPLIANT_NPC=1
export WANDB_MODE="${WANDB_MODE:-offline}"
# Only 2.5 may reach dump boxes / x_objs (default of bench/plant2_frames.py;
# stated here so the run is self-documenting).
export PLANT2_DUMP_SIGN_CLASSES="${PLANT2_DUMP_SIGN_CLASSES:-2.5}"

# Belyaev tree: carries the sign-restore fix. Every stage uses it, including
# trajectory collection. Collection used to run on the zinkovich tree because
# the zinkovich run_benchmark.py imports traffic_signs.pedestrian_crossing_sign,
# which does not exist here; the belyaev priority_bench needs only
# priority_signs/no_traffic_sign, so expert_replay_priority.py and
# select_experts_coverage.py import cleanly under this PYTHONPATH.
PP_BELYAEV="$TRB_ROOT/metadrive:$TRB_ROOT/pdd-bench:$TRB_ROOT/pdd-bench/scripts/per_sign_bench"
export PYTHONPATH="$PP_BELYAEV"

DUMP_TRAIN="$WORK/plant2_l1_stop_train"
DUMP_TEST="$WORK/plant2_l1_stop_test"
TEST_TRAJ="$WORK/trajectories_test"
TEST_EXPERTS="$TEST_TRAJ/experts/experts_scene_uid_top1.jsonl"
SPLIT="$WORK/plant2_l1_stop_split"
DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_stop_signfix}"
CACHE_SIZE_GB="${CACHE_SIZE_GB:-400}"
ADDON="${CHECKPOINT_ADDON:-stop_signfix_lr3e4_ep20}"
N_DUMP_WORKERS="${N_DUMP_WORKERS:-16}"
N_TEST_DUMP_WORKERS="${N_TEST_DUMP_WORKERS:-8}"
N_VAL="${N_VAL:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-3e-4}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_EVAL="${GPU_EVAL:-1}"
N_GIFS="${N_GIFS:-5}"
JOBS_FULL="${JOBS_FULL:-8}"
CKPT_DIR="$PLAN_T/checkpoints_ft/$ADDON"
# latest_ckpt() picks by mtime and epoch=019/last_ft are written in the same
# second, so set CKPT_EVAL to pin the checkpoint the eval must use.
CKPT_EVAL="${CKPT_EVAL:-}"
GIF_PATHS="$WORK/gif_paths.txt"
GIF_MANIFEST_DIR="$WORK/gif_input"
GIF_MANIFEST="$GIF_MANIFEST_DIR/real_manifest.jsonl"
# Scenes whose route fails RouteValidation ("route loops back to spawn"); they
# never produce a GIF, so they are excluded before the 5-GIF run.
BAD_SCENE_IDS="${BAD_SCENE_IDS:-junc_1465245376,junc_266723098,junc_96929782}"

LOGDIR="$WORK/logs"
MASTER_LOG="$LOGDIR/pipeline_master.log"
STATUS="$WORK/STATUS.txt"
CURRENT_STAGE="$WORK/CURRENT_STAGE.txt"

# `--check-only` runs the preflight and stops, without touching STATUS.txt or any
# stage. Use it to confirm the data and the imports are sound before committing
# to a multi-hour run.
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  # Keep the real run's STATUS.txt intact.
  STATUS="$LOGDIR/preflight_status.txt"
fi

# Inline python heredocs read these via os.environ; a missing export silently
# kills a stage, so keep the list in sync with what the heredocs use.
export WORK DUMP_TRAIN DUMP_TEST TEST_TRAJ TEST_EXPERTS TRAIN_EXPERTS SPLIT
export DS_LOCAL CACHE_SIZE_GB ADDON N_VAL MAX_EPOCHS LR
export GPU_TRAIN GPU_EVAL N_GIFS LOGDIR TEST_MANIFEST SCENES CKPT0 CKPT_DIR
export GIF_PATHS GIF_MANIFEST GIF_MANIFEST_DIR BAD_SCENE_IDS JOBS_FULL CKPT_EVAL
export STOP_DATA

mkdir -p "$LOGDIR" "$WORK" "$PLAN_T/checkpoints_ft"

ts() { date -Is; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }
cur_stage() { cat "$CURRENT_STAGE" 2>/dev/null || echo unknown; }
FAIL_RECORDED=0
fail() {
  FAIL_RECORDED=1
  log "FAIL stage=$(cur_stage): $*"
  echo "FAIL $(cur_stage) $(ts) $*" >> "$STATUS"
  exit 1
}
warn() { log "WARN: $*"; echo "WARN $1 $(ts)" >> "$STATUS"; }
set_stage() { echo "$1" > "$CURRENT_STAGE"; log "stage=$1"; }
ok_stage() { log "OK stage=$1"; echo "OK $1 $(ts)" >> "$STATUS"; }
skip_stage() { log "SKIP stage=$1 reason=$2"; echo "SKIP $1 $(ts) $2" >> "$STATUS"; }

# `set -e` + pipefail abort the driver on any stage command that lacks an
# explicit `|| fail`, and that abort used to leave STATUS.txt showing the last
# OK line as if the run were still healthy. These traps guarantee a FAIL line.
# SIGKILL (OOM killer) still cannot be caught; a missing terminal line in
# STATUS.txt therefore means "killed with -9".
on_exit() {
  local rc=$?
  [[ "$rc" -eq 0 || "$FAIL_RECORDED" -eq 1 ]] && return 0
  log "FAIL stage=$(cur_stage): abnormal exit rc=$rc"
  echo "FAIL $(cur_stage) $(ts) abnormal exit rc=$rc" >> "$STATUS"
}
on_signal() {
  FAIL_RECORDED=1
  log "FAIL stage=$(cur_stage): killed by SIG$1"
  echo "FAIL $(cur_stage) $(ts) killed by SIG$1" >> "$STATUS"
  exit 143
}
trap on_exit EXIT
trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# `set -o pipefail` turns a find over a missing dir into a fatal error, so the
# not-yet-dumped case is handled explicitly.
n_route_dirs() {
  local d="$1/data"
  if [[ ! -d "$d" ]]; then
    echo 0
    return 0
  fi
  find "$d" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
}

have_all_runs() {
  [[ -d "$TEST_TRAJ" ]] || return 1
  local hit
  hit="$(find "$TEST_TRAJ" -name all_runs.jsonl -print -quit 2>/dev/null || true)"
  [[ -n "$hit" ]]
}

echo "RUNNING $(ts)" > "$STATUS"
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "=== STOP signfix preflight (--check-only, no stage will run) ==="
else
  log "=== STOP signfix pipeline start ==="
fi
log "WORK=$WORK PY=$PY CKPT0=$CKPT0"
log "STOP_DATA=$STOP_DATA"
log "PRIORITY=$PRIORITY"
log "TRAIN_EXPERTS=$TRAIN_EXPERTS"
log "TEST_MANIFEST=$TEST_MANIFEST"
log "SCENES=$SCENES"
log "PLANT2_DUMP_SIGN_CLASSES=$PLANT2_DUMP_SIGN_CLASSES"

[[ -x "$PY" ]] || fail "python missing: $PY"
[[ -f "$CKPT0" ]] || fail "ckpt missing: $CKPT0"
[[ -s "$TRAIN_EXPERTS" ]] || fail "train experts missing: $TRAIN_EXPERTS"
[[ -f "$TEST_MANIFEST" ]] || fail "test manifest missing: $TEST_MANIFEST"
[[ -d "$SCENES" ]] || fail "scenes missing: $SCENES"

# The -f/-d tests above pass on a dangling symlink's parent, which is exactly how
# the 2026-08-20 failure got through: the scenes dir existed, every junc_* inside
# it was dead, and 50 eval subprocesses died one by one while the driver reported
# nothing. check_stop_data.py resolves every scene_id, every relative net_path and
# every expert replay, and names the offending path. It costs ~0.5s, so it runs
# unconditionally before any stage.
DATA_CHECK="$WORK/check_stop_data.py"
[[ -f "$DATA_CHECK" ]] || fail "data check missing: $DATA_CHECK"
"$PY" "$DATA_CHECK" \
  --scenes "$SCENES" \
  --test-manifest "$TEST_MANIFEST" \
  --train-experts "$TRAIN_EXPERTS" \
  --ckpt "$CKPT0" \
  2>&1 | tee -a "$MASTER_LOG" \
  || fail "stop-data validation failed (see the named paths above); re-vendor with $WORK/vendor_stop_data.py"

# Every scene is a fresh `python run_benchmark.py` / expert_replay_*.py, and the
# drivers surface an import failure only as a per-scene "exit 1", so verify the
# imports once here instead of discovering it 50 subprocesses in.
code_preflight() {
  local dir="$1" script="$2"
  ( cd "$dir" && PYTHONPATH="$PP_BELYAEV" "$PY" "$script" --help ) >/dev/null 2>&1
}
for spec in \
  "$PRIORITY:run_benchmark.py" \
  "$PRIORITY/collect_trajectories:expert_replay_priority.py" \
  "$PRIORITY/collect_trajectories:select_experts_coverage.py" \
  "$TRB_ROOT/pdd-bench/scripts/per_sign_bench:expert_replay_for_plant2.py"
do
  code_preflight "${spec%:*}" "${spec##*:}" \
    || fail "code preflight: ${spec%:*}/${spec##*:} does not import under PYTHONPATH=$PP_BELYAEV"
done
log "preflight OK: data validated, all entrypoints import"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "=== PREFLIGHT OK (--check-only), no stage run ==="
  echo "PREFLIGHT_OK $(ts)" > "$STATUS"
  exit 0
fi

# Shard expert_replay_for_plant2.py over N workers and wait for all of them.
# $1=experts jsonl  $2=out dir  $3=n workers  $4=log tag
run_sharded_dump() {
  local experts="$1" out="$2" workers="$3" tag="$4"
  local n shard pids=() fails=0 i=0 start=0 count slog
  n=$(wc -l < "$experts")
  [[ "$n" -gt 0 ]] || fail "$tag: empty experts file $experts"
  mkdir -p "$out/logs"
  shard=$(( (n + workers - 1) / workers ))
  [[ "$shard" -lt 1 ]] && shard=1
  log "$tag: n=$n -> $out workers=$workers shard=$shard"
  while [[ "$start" -lt "$n" ]]; do
    count=$shard
    if (( start + count > n )); then count=$(( n - start )); fi
    slog="$out/logs/shard_${i}_s${start}_c${count}.log"
    (
      cd "$TRB_ROOT/pdd-bench/scripts/per_sign_bench"
      PYTHONPATH="$PP_BELYAEV" "$PY" -u expert_replay_for_plant2.py \
        --experts "$experts" \
        --scenes-root "$SCENES" \
        --save-plant2-dir "$out" \
        --start "$start" \
        --count "$count" \
        --max-steps 1500 \
        --backends sumo
    ) >"$slog" 2>&1 &
    pids+=($!)
    log "$tag: spawn shard=$i start=$start count=$count pid=${pids[-1]} log=$slog"
    start=$(( start + count ))
    i=$(( i + 1 ))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fails=$(( fails + 1 ))
      log "$tag: worker pid=$pid FAILED"
    fi
  done
  log "$tag: routes=$(n_route_dirs "$out") fails=$fails of $i shards"
}

# ---------- 1) dump train (344 routes, sharded) ----------
stage_dump_train() {
  local n n_routes
  n=$(wc -l < "$TRAIN_EXPERTS")
  n_routes=$(n_route_dirs "$DUMP_TRAIN")
  if [[ "$n_routes" -ge "$n" ]]; then
    skip_stage dump_train "$n_routes routes already dumped (>= $n experts)"
    return 0
  fi
  set_stage dump_train
  run_sharded_dump "$TRAIN_EXPERTS" "$DUMP_TRAIN" "$N_DUMP_WORKERS" dump_train
  n_routes=$(n_route_dirs "$DUMP_TRAIN")
  [[ "$n_routes" -ge $(( n * 9 / 10 )) ]] \
    || fail "dump_train produced only $n_routes routes for $n experts"
  ok_stage dump_train
}

# ---------- 2) collect + select + dump TEST ----------
stage_collect_test() {
  if have_all_runs; then
    skip_stage collect_test "all_runs.jsonl present under $TEST_TRAJ"
    return 0
  fi
  set_stage collect_test
  mkdir -p "$TEST_TRAJ"
  log "collect_test: manifest=$TEST_MANIFEST -> $TEST_TRAJ"
  ( cd "$PRIORITY/collect_trajectories"
    PYTHONPATH="$PP_BELYAEV" "$PY" -u expert_replay_priority.py \
      --sign stop \
      --manifest "$TEST_MANIFEST" \
      --scenes-root "$SCENES" \
      --policy comprehensive_rule_expert \
      --output-dir "$TEST_TRAJ" \
      --max-steps 1500 \
      --resume
  ) >"$LOGDIR/02a_collect_test.log" 2>&1 \
    || fail "collect_test failed; see $LOGDIR/02a_collect_test.log"
  have_all_runs || fail "collect_test produced no all_runs.jsonl under $TEST_TRAJ"
  ok_stage collect_test
}

stage_select_test() {
  if [[ -s "$TEST_EXPERTS" ]]; then
    skip_stage select_test "$(wc -l < "$TEST_EXPERTS") experts already selected"
    return 0
  fi
  set_stage select_test
  log "select_test: -> $TEST_TRAJ/experts"
  ( cd "$PRIORITY/collect_trajectories"
    PYTHONPATH="$PP_BELYAEV" "$PY" -u select_experts_coverage.py \
      --root "$TEST_TRAJ" \
      --manifest "$TEST_MANIFEST" \
      --signs 2.5 \
      --horizon 1500 \
      --min-join-rate 0.0 \
      --out-dir "$TEST_TRAJ/experts"
  ) >"$LOGDIR/02b_select_test.log" 2>&1 \
    || fail "select_test failed; see $LOGDIR/02b_select_test.log"
  [[ -s "$TEST_EXPERTS" ]] || fail "select_test produced no $TEST_EXPERTS"
  ok_stage select_test
}

stage_dump_test() {
  local n n_routes
  n=$(wc -l < "$TEST_EXPERTS")
  n_routes=$(n_route_dirs "$DUMP_TEST")
  if [[ "$n_routes" -ge "$n" ]]; then
    skip_stage dump_test "$n_routes routes already dumped (>= $n experts)"
    return 0
  fi
  set_stage dump_test
  run_sharded_dump "$TEST_EXPERTS" "$DUMP_TEST" "$N_TEST_DUMP_WORKERS" dump_test
  n_routes=$(n_route_dirs "$DUMP_TEST")
  [[ "$n_routes" -gt 0 ]] || fail "dump_test produced no routes"
  ok_stage dump_test
}

# ---------- 3) split train dump into train/val (exactly 50 val) ----------
split_ready() {
  [[ -f "$SPLIT/split_meta.json" ]] || return 1
  "$PY" - <<'PY'
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

stage_split() {
  if split_ready; then
    skip_stage split "existing split with $N_VAL val"
    return 0
  fi
  set_stage split
  log "split: $DUMP_TRAIN -> $SPLIT (n_val=$N_VAL)"
  "$PY" -u - <<'PY' 2>&1 | tee "$LOGDIR/03_split.log"
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

# ---------- 4) prefill diskcache ----------
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
    --log-dir "$LOGDIR/prefill" \
    2>&1 | tee "$LOGDIR/04_prefill.log"
  prefill_ready || fail "prefill validation failed"
  ok_stage prefill
}

# ---------- 5) train (20 epochs, lr 3e-4) ----------
latest_ckpt() {
  "$PY" - <<'PY'
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
print(max({p.resolve() for p in candidates}, key=lambda p: p.stat().st_mtime))
PY
}

stage_train() {
  if latest_ckpt >/dev/null 2>&1; then
    skip_stage train "checkpoint already exists under $CKPT_DIR"
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
    --log "$LOGDIR/05_train.log" \
    2>&1 | tee -a "$LOGDIR/05_train.log"
  latest_ckpt > "$WORK/final_ckpt.txt" || fail "train produced no checkpoint"
  log "train: ckpt=$(cat "$WORK/final_ckpt.txt")"
  ok_stage train
}

# ---------- 6) eval on test manifest + 5 GIFs ----------
# Eval runs the belyaev priority_bench. On 2026-08-19 the zinkovich
# run_benchmark.py gained a `traffic_signs.pedestrian_crossing_sign` import; that
# module ships only under the zinkovich pdd-bench, which is not on PP_BELYAEV, so
# all 50 scene subprocesses died with ModuleNotFoundError. The belyaev copy needs
# only priority_signs/no_traffic_sign and keeps eval on the same tree as the
# sign-restore fix the run is testing. $PRIORITY is now that copy, so this is an
# alias; the name is kept because the eval logs and reports refer to it.
EVAL_PRIORITY="$PRIORITY"

eval_ready() {
  [[ -f "$GIF_PATHS" ]] || return 1
  "$PY" - <<'PY'
import os
from pathlib import Path
paths = [l.strip() for l in Path(os.environ["GIF_PATHS"]).read_text().splitlines() if l.strip()]
n = int(os.environ["N_GIFS"])
assert len(paths) >= n, f"{len(paths)} gifs < {n}"
for p in paths[:n]:
    assert Path(p).is_file(), p
PY
}

stage_eval() {
  if eval_ready; then
    skip_stage eval "existing $N_GIFS gifs listed in $GIF_PATHS"
    return 0
  fi
  set_stage eval
  local ckpt eval_out
  if [[ -n "$CKPT_EVAL" ]]; then
    ckpt="$CKPT_EVAL"
    [[ -f "$ckpt" ]] || fail "CKPT_EVAL is not a file: $ckpt"
  else
    ckpt="$(latest_ckpt)" || fail "no checkpoint under $CKPT_DIR"
  fi
  log "eval: ckpt=$ckpt"
  eval_out="$WORK/eval_test"
  mkdir -p "$eval_out"

  cd "$EVAL_PRIORITY"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" \
  "$PY" -u eval_pipeline.py \
    --policies plant2 \
    --model-paths "plant2:$ckpt" \
    --manifest "$TEST_MANIFEST" \
    --scenes-root "$SCENES" \
    --out-dir "$eval_out/full" \
    --plant2-action-mode pid \
    --jobs "$JOBS_FULL" \
    --backends sumo \
    2>&1 | tee "$LOGDIR/06a_eval_full.log" \
    || fail "eval full run failed; see $LOGDIR/06a_eval_full.log"

  # load_manifest_config() reads manifest.json / real_manifest_summary.json from
  # the manifest's own directory, so the gif manifest gets copies of the ts_test
  # sidecars; without them spawn_distance_before_end silently falls back to 12.0
  # and the GIFs no longer show the same scenario as the full eval.
  mkdir -p "$GIF_MANIFEST_DIR"
  for sidecar in manifest.json real_manifest_summary.json; do
    if [[ -f "$(dirname "$TEST_MANIFEST")/$sidecar" ]]; then
      cp -f "$(dirname "$TEST_MANIFEST")/$sidecar" "$GIF_MANIFEST_DIR/$sidecar"
    fi
  done

  # Drop loop-back routes (known-bad plus anything RouteValidation rejected in
  # the full run) so --max-scenes actually yields N_GIFS gifs.
  FULL_LOG="$LOGDIR/06a_eval_full.log" "$PY" - <<'PY' || fail "could not build $GIF_MANIFEST with $N_GIFS usable scenes"
import json, os, re
from pathlib import Path
bad = {s for s in os.environ["BAD_SCENE_IDS"].split(",") if s}
log = Path(os.environ["FULL_LOG"])
if log.is_file():
    bad |= set(re.findall(r"\[RouteValidation\] INVALID: (\S+)", log.read_text(errors="replace")))
n = int(os.environ["N_GIFS"])
rows = []
with open(os.environ["TEST_MANIFEST"], encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get("scene_id")) in bad:
            continue
        rows.append(line)
        if len(rows) >= n:
            break
out = Path(os.environ["GIF_MANIFEST"])
out.write_text("\n".join(rows) + "\n")
print(f"gif manifest: {len(rows)} rows -> {out} (excluded scene_ids: {sorted(bad)})")
if len(rows) < n:
    raise SystemExit(f"only {len(rows)} usable scenes for {n} gifs")
PY

  CUDA_VISIBLE_DEVICES="$GPU_EVAL" \
  "$PY" -u eval_pipeline.py \
    --policies plant2 \
    --model-paths "plant2:$ckpt" \
    --manifest "$GIF_MANIFEST" \
    --scenes-root "$SCENES" \
    --out-dir "$eval_out/gifs5" \
    --max-scenes "$N_GIFS" \
    --save-gifs \
    --plant2-action-mode pid \
    --jobs 1 \
    --backends sumo \
    2>&1 | tee "$LOGDIR/06b_eval_gifs.log" \
    || fail "eval gif run failed; see $LOGDIR/06b_eval_gifs.log"

  EVAL_OUT="$eval_out/gifs5" "$PY" - <<'PY' || fail "could not collect gif paths into $GIF_PATHS"
import os
from pathlib import Path
gifs = sorted(str(p.resolve()) for p in Path(os.environ["EVAL_OUT"]).rglob("*.gif"))
Path(os.environ["GIF_PATHS"]).write_text("\n".join(gifs) + ("\n" if gifs else ""))
print("\n".join(gifs))
PY
  echo "$ckpt" > "$WORK/final_ckpt.txt"
  eval_ready || fail "eval did not produce $N_GIFS gifs (see $LOGDIR/06b_eval_gifs.log)"
  log "eval done; gifs listed in $GIF_PATHS"
  ok_stage eval
}

stage_dump_train
stage_collect_test
stage_select_test
stage_dump_test
stage_split
stage_prefill
stage_train
stage_eval

log "=== PIPELINE COMPLETE ==="
echo "SUCCESS $(ts)" > "$STATUS"
echo "ckpt=$(cat "$WORK/final_ckpt.txt")" >> "$STATUS"
echo "gifs=$GIF_PATHS" >> "$STATUS"
echo "done" > "$CURRENT_STAGE"
