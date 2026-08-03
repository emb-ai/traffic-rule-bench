#!/usr/bin/env bash
# Dump PlanT2 L1 frames from remapped expert winners via expert_replay_inenv.py.
#
# Usage:
#   bash dump_plant2_l1_from_experts.sh                  # all sign families
#   SIGNS="yield detour" bash dump_plant2_l1_from_experts.sh
#   COUNT=3 SAVE_GIFS=1 bash dump_plant2_l1_from_experts.sh   # smoke
#   COUNT=100 START=0 bash dump_plant2_l1_from_experts.sh     # chunk
#   DRY_RUN=1 bash dump_plant2_l1_from_experts.sh
#
# Env overrides:
#   OUT_DIR      PlanT2 root (default: .../shepelev/plant2_l1_from_experts)
#   EXPERTS_RANK top1|top2 (default: top1)
#   SIGNS        space-separated subset (default: all)
#   COUNT/START  passed to expert_replay_inenv (--count / --start)
#   SAVE_GIFS=1  write GIFs under OUT_DIR/gifs
#   BACKENDS     default: sumo
#   PYTHON       default: python
#   DRY_RUN=1    print commands only

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

SHEPELEV_ROOT="${SHEPELEV_ROOT:-$SHEPELEV}"
REPO="${REPO:-$TRB_ROOT}"
BENCH_DIR="$REPO/pdd-bench/scripts/per_sign_bench"
CT="${CT:-$PIPELINE_DIR}"

# Scene packs used when trajectories were collected.
ZINK_BENCH="${ZINK_BENCH:-/mnt/virtual_ai0001053-01202_SR006-nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench/scripts/per_sign_bench}"
DETOUR_SCENES="${DETOUR_SCENES:-/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/sdc/pdd-bench/scenes}"

OUT_DIR="${OUT_DIR:-$SHEPELEV_ROOT/plant2_l1_from_experts}"
EXPERTS_RANK="${EXPERTS_RANK:-top1}"
SIGNS="${SIGNS:-yield stop secondary main roundabout detour}"
BACKENDS="${BACKENDS:-sumo}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_GIFS="${SAVE_GIFS:-0}"
COUNT="${COUNT:-}"
START="${START:-0}"

EXPERTS_FILE="experts_scene_uid_${EXPERTS_RANK}.jsonl"

declare -A EXPERTS_PATH=(
  [yield]="$CT/traj-priority-signs/traj_yield_2_4_train80/experts/$EXPERTS_FILE"
  [stop]="$CT/traj-priority-signs/traj_stop_2_5_train80/experts/$EXPERTS_FILE"
  [secondary]="$CT/traj-priority-signs/traj_secondary_2_3_train80/experts/$EXPERTS_FILE"
  [main]="$CT/traj-priority-signs/traj_main_2_1_train80/experts/$EXPERTS_FILE"
  [roundabout]="$CT/traj-priority-signs/traj_roundabout_4_3_train80/experts/$EXPERTS_FILE"
  [detour]="$CT/traffic-rule-bench-traj/experts_detour_train80/$EXPERTS_FILE"
)

declare -A SCENES_ROOT=(
  [yield]="$ZINK_BENCH/yield_sign/scenes/2_4"
  [stop]="$ZINK_BENCH/stop_sign/scenes/2_5"
  [secondary]="$ZINK_BENCH/secondary_sign/scenes/2_3"
  [main]="$ZINK_BENCH/main_sign/scenes/2_1"
  [roundabout]="$ZINK_BENCH/roundabout_sign/scenes/4_3"
  [detour]="$DETOUR_SCENES"
)

SCRIPT="$BENCH_DIR/expert_replay_inenv.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: missing $SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/dump_${ts}.log"

echo "================================================================"
echo "dump_plant2_l1_from_experts  [$ts]"
echo "  OUT_DIR       = $OUT_DIR"
echo "  EXPERTS_RANK  = $EXPERTS_RANK"
echo "  SIGNS         = $SIGNS"
echo "  COUNT/START   = ${COUNT:-(all)} / $START"
echo "  SAVE_GIFS     = $SAVE_GIFS"
echo "  BACKENDS      = $BACKENDS"
echo "  DRY_RUN       = $DRY_RUN"
echo "  log           = $summary_log"
echo "================================================================" | tee "$summary_log"

fail=0
ok=0

for sign in $SIGNS; do
  experts="${EXPERTS_PATH[$sign]:-}"
  scenes="${SCENES_ROOT[$sign]:-}"
  if [[ -z "$experts" || -z "$scenes" ]]; then
    echo "[skip] unknown sign='$sign' (known: ${!EXPERTS_PATH[*]})" | tee -a "$summary_log"
    continue
  fi
  if [[ ! -f "$experts" ]]; then
    echo "[FAIL] $sign: experts missing: $experts" | tee -a "$summary_log"
    fail=$((fail + 1))
    continue
  fi
  if [[ ! -d "$scenes" ]]; then
    echo "[FAIL] $sign: scenes-root missing: $scenes" | tee -a "$summary_log"
    fail=$((fail + 1))
    continue
  fi

  cmd=(
    "$PYTHON" "$SCRIPT"
    --experts "$experts"
    --scenes-root "$scenes"
    --save-plant2-dir "$OUT_DIR"
    --backends "$BACKENDS"
    --ego-mode recorded
    --npc-mode recorded
    --start "$START"
  )
  if [[ -n "$COUNT" ]]; then
    cmd+=(--count "$COUNT")
  fi
  if [[ "$SAVE_GIFS" == "1" ]]; then
    cmd+=(--save-gifs --gif-dir "$OUT_DIR/gifs/$sign")
  fi

  echo "" | tee -a "$summary_log"
  echo "[run] $sign" | tee -a "$summary_log"
  echo "  experts = $experts" | tee -a "$summary_log"
  echo "  scenes  = $scenes" | tee -a "$summary_log"
  printf '  cmd     =' | tee -a "$summary_log"
  printf ' %q' "${cmd[@]}" | tee -a "$summary_log"
  echo | tee -a "$summary_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] skipped" | tee -a "$summary_log"
    ok=$((ok + 1))
    continue
  fi

  sign_log="$LOG_DIR/${sign}_${ts}.log"
  if ( cd "$BENCH_DIR" && "${cmd[@]}" ) 2>&1 | tee "$sign_log" | tee -a "$summary_log"; then
    echo "[ok]  $sign  (log: $sign_log)" | tee -a "$summary_log"
    ok=$((ok + 1))
  else
    echo "[FAIL] $sign  (log: $sign_log)" | tee -a "$summary_log"
    fail=$((fail + 1))
  fi
done

echo "" | tee -a "$summary_log"
echo "=== done: ok=$ok fail=$fail out=$OUT_DIR ===" | tee -a "$summary_log"
exit "$fail"
