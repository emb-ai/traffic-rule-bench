#!/usr/bin/env bash
# Parallel PlanT2 L1 dump for traj_lane_5_15_train80 (sharded across workers).
#
# Usage:
#   bash dump_plant2_l1_lane_parallel.sh
#   COUNT=300 N_SHARDS=8 bash dump_plant2_l1_lane_parallel.sh
#   DRY_RUN=1 bash dump_plant2_l1_lane_parallel.sh
#
# Env:
#   OUT_DIR     default: .../shepelev/plant2_l1_lane300
#   COUNT       default: 300 (top1 has 568)
#   START       global start offset (default: 0)
#   N_SHARDS    parallel workers (default: 8)
#   EXPERTS_RANK top1|top2 (default: top1)

set -u

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

SHEPELEV_ROOT="${SHEPELEV_ROOT:-$SHEPELEV}"
CT="${CT:-$PIPELINE_DIR}"
REPO="${REPO:-$TRB_ROOT}"
BENCH_DIR="$REPO/pdd-bench/scripts/per_sign_bench"
ZINK_BENCH="${ZINK_BENCH:-/mnt/virtual_ai0001053-01202_SR006-nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench/scripts/per_sign_bench}"

OUT_DIR="${OUT_DIR:-$SHEPELEV_ROOT/plant2_l1_lane300}"
EXPERTS_RANK="${EXPERTS_RANK:-top1}"
COUNT="${COUNT:-300}"
START="${START:-0}"
N_SHARDS="${N_SHARDS:-8}"
BACKENDS="${BACKENDS:-sumo}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_GIFS="${SAVE_GIFS:-0}"

ARBELYAEV_PY="$SHEPELEV_ROOT/conda_envs/arbelyaev-sdc/bin/python"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ARBELYAEV_PY" ]]; then
    PYTHON="$ARBELYAEV_PY"
  elif [[ -x /home/user/conda/envs/zinkovich-sdc/bin/python ]]; then
    PYTHON=/home/user/conda/envs/zinkovich-sdc/bin/python
  else
    PYTHON=python
  fi
fi

EXPERTS="$CT/traj-priority-signs/traj_lane_5_15_train80/experts/experts_scene_uid_${EXPERTS_RANK}.jsonl"
SCENES="$ZINK_BENCH/lane_direction_signs/scenes/5_15_1"
SCRIPT="$BENCH_DIR/expert_replay_inenv.py"

if [[ ! -f "$SCRIPT" ]]; then echo "ERROR: missing $SCRIPT" >&2; exit 1; fi
if [[ ! -f "$EXPERTS" ]]; then echo "ERROR: missing $EXPERTS" >&2; exit 1; fi
if [[ ! -d "$SCENES" ]]; then echo "ERROR: missing $SCENES" >&2; exit 1; fi

mkdir -p "$OUT_DIR/logs"
ts="$(date +%Y%m%d_%H%M%S)"
summary_log="$OUT_DIR/logs/lane_parallel_${ts}.log"

{
  echo "================================================================"
  echo "dump_plant2_l1_lane_parallel  [$ts]"
  echo "  OUT_DIR       = $OUT_DIR"
  echo "  EXPERTS       = $EXPERTS"
  echo "  SCENES        = $SCENES"
  echo "  COUNT/START   = $COUNT / $START"
  echo "  N_SHARDS      = $N_SHARDS"
  echo "  PYTHON        = $PYTHON"
  echo "  DRY_RUN       = $DRY_RUN"
  echo "  log           = $summary_log"
  echo "================================================================"
} | tee "$summary_log"

n_experts="$(wc -l <"$EXPERTS" | tr -d ' ')"
avail=$((n_experts - START))
if [[ "$avail" -lt 0 ]]; then avail=0; fi
if [[ "$COUNT" -gt "$avail" ]]; then COUNT="$avail"; fi
if [[ "$COUNT" -le 0 ]]; then
  echo "ERROR: nothing to dump (n_experts=$n_experts START=$START)" | tee -a "$summary_log"
  exit 1
fi
if [[ "$N_SHARDS" -gt "$COUNT" ]]; then N_SHARDS="$COUNT"; fi

run_shard() {
  local shard="$1"
  local shard_start="$2"
  local shard_count="$3"
  local sign_log="$OUT_DIR/logs/lane_shard${shard}_${ts}.log"
  local rc_file="$OUT_DIR/logs/lane_shard${shard}_${ts}.rc"

  local -a cmd=(
    "$PYTHON" "$SCRIPT"
    --experts "$EXPERTS"
    --scenes-root "$SCENES"
    --save-plant2-dir "$OUT_DIR"
    --backends "$BACKENDS"
    --ego-mode recorded
    --npc-mode recorded
    --start "$shard_start"
    --count "$shard_count"
  )
  if [[ "$SAVE_GIFS" == "1" ]]; then
    cmd+=(--save-gifs --gif-dir "$OUT_DIR/gifs/lane_shard${shard}")
  fi

  {
    echo "[run] shard=$shard start=$shard_start count=$shard_count"
    printf '  cmd ='
    printf ' %q' "${cmd[@]}"
    echo
  } | tee -a "$summary_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] shard=$shard skipped" | tee -a "$summary_log"
    echo 0 >"$rc_file"
    return 0
  fi

  ( cd "$BENCH_DIR" && "${cmd[@]}" ) >"$sign_log" 2>&1
  local rc=$?
  echo "$rc" >"$rc_file"
  if [[ "$rc" -eq 0 ]]; then
    echo "[ok]  shard=$shard (log: $sign_log)" | tee -a "$summary_log"
  else
    echo "[FAIL] shard=$shard rc=$rc (log: $sign_log)" | tee -a "$summary_log"
  fi
  return 0
}

pids=()
for ((shard=0; shard<N_SHARDS; shard++)); do
  # Partition COUNT items starting at START across shards.
  local_start=$(( START + shard * COUNT / N_SHARDS ))
  next_start=$(( START + (shard + 1) * COUNT / N_SHARDS ))
  shard_count=$(( next_start - local_start ))
  if [[ "$shard_count" -le 0 ]]; then
    continue
  fi
  run_shard "$shard" "$local_start" "$shard_count" &
  pid=$!
  pids+=("$pid")
  echo "[spawn] shard=$shard pid=$pid start=$local_start count=$shard_count" | tee -a "$summary_log"
done

for pid in "${pids[@]+"${pids[@]}"}"; do
  wait "$pid" || true
done

ok=0
fail=0
for ((shard=0; shard<N_SHARDS; shard++)); do
  rc_file="$OUT_DIR/logs/lane_shard${shard}_${ts}.rc"
  [[ -f "$rc_file" ]] || continue
  rc="$(cat "$rc_file")"
  if [[ "$rc" == "0" ]]; then ok=$((ok + 1)); else fail=$((fail + 1)); fi
done

{
  echo ""
  echo "=== done: ok=$ok fail=$fail out=$OUT_DIR count=$COUNT shards=$N_SHARDS ==="
  if [[ -d "$OUT_DIR/data" ]]; then
    echo "  routes in data/: $(find "$OUT_DIR/data" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  fi
} | tee -a "$summary_log"

exit "$fail"
