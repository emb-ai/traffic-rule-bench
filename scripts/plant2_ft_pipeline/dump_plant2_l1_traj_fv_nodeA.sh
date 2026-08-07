#!/usr/bin/env bash
# Dump PlanT2 L1 from traj_fv_train80_nodeA experts (experts_fv_train80 top1, nodeA only).
#
# Usage:
#   bash dump_plant2_l1_traj_fv_nodeA.sh
#   COUNT=300 bash dump_plant2_l1_traj_fv_nodeA.sh
#   SIGNS="3.24 4.6" MAX_JOBS=2 bash dump_plant2_l1_traj_fv_nodeA.sh
#   DRY_RUN=1 bash dump_plant2_l1_traj_fv_nodeA.sh
#
# Env:
#   OUT_DIR       default: .../shepelev/plant2_l1_traj_fv_nodeA
#   EXPERTS_SRC   default: .../smirnova/experts_fv_train80/experts_scene_uid_top1.jsonl
#   SCENES_ROOT   default: .../smirnova/.../scenes_balanced
#   COUNT/START   optional slice per sign
#   SIGNS         default: 3.24 4.6 5.21 5.31
#   MAX_JOBS      default: all signs at once
#   PYTHON        default: zinkovich-sdc

set -u

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

SHEPELEV_ROOT="${SHEPELEV_ROOT:-$SHEPELEV}"
REPO="${REPO:-$TRB_ROOT}"
BENCH_DIR="$REPO/pdd-bench/scripts/per_sign_bench"

SM_MNT="${SM_MNT:-/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova}"
OUT_DIR="${OUT_DIR:-$SHEPELEV_ROOT/plant2_l1_traj_fv_nodeA}"
EXPERTS_SRC="${EXPERTS_SRC:-$SM_MNT/experts_fv_train80/experts_scene_uid_top1.jsonl}"
SCENES_ROOT="${SCENES_ROOT:-$SM_MNT/traffic-rule-bench/pdd-bench/scenes_balanced}"
NODE_FILTER="${NODE_FILTER:-nodeA}"

SIGNS="${SIGNS:-3.24 4.6 5.21 5.31}"
COUNT="${COUNT:-}"
START="${START:-0}"
BACKENDS="${BACKENDS:-sumo}"
PYTHON="${PYTHON:-/home/user/conda/envs/zinkovich-sdc/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_GIFS="${SAVE_GIFS:-0}"
MAX_JOBS="${MAX_JOBS:-0}"

SCRIPT="$BENCH_DIR/expert_replay_inenv.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: missing $SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$EXPERTS_SRC" ]]; then
  echo "ERROR: missing experts: $EXPERTS_SRC" >&2
  exit 1
fi
if [[ ! -d "$SCENES_ROOT" ]]; then
  echo "ERROR: missing scenes: $SCENES_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/experts"
ts="$(date +%Y%m%d_%H%M%S)"
summary_log="$OUT_DIR/logs/fv_nodeA_${ts}.log"
EXPERTS_DIR="$OUT_DIR/experts"

{
  echo "================================================================"
  echo "dump_plant2_l1_traj_fv_nodeA  [$ts]"
  echo "  OUT_DIR       = $OUT_DIR"
  echo "  EXPERTS_SRC   = $EXPERTS_SRC"
  echo "  NODE_FILTER   = $NODE_FILTER"
  echo "  SCENES_ROOT   = $SCENES_ROOT"
  echo "  SIGNS         = $SIGNS"
  echo "  COUNT/START   = ${COUNT:-(all)} / $START"
  echo "  MAX_JOBS      = $MAX_JOBS"
  echo "  DRY_RUN       = $DRY_RUN"
  echo "  log           = $summary_log"
  echo "================================================================"
} | tee "$summary_log"

# Filter nodeA + remap nfs2 shares paths -> mnt, split by sign.
"$PYTHON" - <<PY
import json
from pathlib import Path
from collections import Counter

src = Path("$EXPERTS_SRC")
out_dir = Path("$EXPERTS_DIR")
node = "$NODE_FILTER"
signs = """$SIGNS""".split()
old = "/home/jovyan/shares/SR006.nfs2/smirnova/"
new = "/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/"

PATH_KEYS = ("pkl_path", "sidecar_path", "gif_path", "winning_pkl", "winning_sidecar")

def remap(v):
    if isinstance(v, str) and v.startswith(old):
        return new + v[len(old):]
    return v

by_sign = {s: [] for s in signs}
other = 0
miss_pkl = 0
n_in = 0
for line in open(src):
    o = json.loads(line)
    pkl = o.get("pkl_path") or ""
    if node not in pkl:
        continue
    n_in += 1
    for k in PATH_KEYS:
        if k in o and o[k]:
            o[k] = remap(o[k])
    sign = str(o.get("sign") or "")
    if sign not in by_sign:
        other += 1
        continue
    p = Path(o.get("pkl_path") or "")
    if not p.exists():
        miss_pkl += 1
        continue
    by_sign[sign].append(o)

counts = {}
for sign, rows in by_sign.items():
    path = out_dir / f"experts_{sign.replace('.', '_')}_top1.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts[sign] = len(rows)
    print(f"wrote {path} n={len(rows)}")
print(f"FILTERED node={node} kept={n_in} other_sign={other} miss_pkl={miss_pkl} counts={counts}")
PY

# shellcheck disable=SC2206
sign_list=($SIGNS)
n_signs="${#sign_list[@]}"
if [[ "$MAX_JOBS" -le 0 || "$MAX_JOBS" -gt "$n_signs" ]]; then
  MAX_JOBS="$n_signs"
fi

run_one() {
  local sign="$1"
  local slug="${sign//./_}"
  local experts="$EXPERTS_DIR/experts_${slug}_top1.jsonl"
  local sign_log="$OUT_DIR/logs/${slug}_${ts}.log"
  local rc_file="$OUT_DIR/logs/${slug}_${ts}.rc"

  if [[ ! -f "$experts" ]]; then
    echo "[FAIL] $sign: experts missing: $experts" | tee -a "$summary_log"
    echo 1 >"$rc_file"
    return 0
  fi
  local n_rows
  n_rows="$(wc -l <"$experts" | tr -d ' ')"
  if [[ "$n_rows" -eq 0 ]]; then
    echo "[FAIL] $sign: empty experts" | tee -a "$summary_log"
    echo 1 >"$rc_file"
    return 0
  fi

  local -a cmd=(
    "$PYTHON" "$SCRIPT"
    --experts "$experts"
    --scenes-root "$SCENES_ROOT"
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
    cmd+=(--save-gifs --gif-dir "$OUT_DIR/gifs/$slug")
  fi

  {
    echo "[run] $sign  (n_experts=$n_rows)"
    echo "  experts = $experts"
    echo "  scenes  = $SCENES_ROOT"
    printf '  cmd     ='
    printf ' %q' "${cmd[@]}"
    echo
  } | tee -a "$summary_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $sign skipped" | tee -a "$summary_log"
    echo 0 >"$rc_file"
    return 0
  fi

  ( cd "$BENCH_DIR" && "${cmd[@]}" ) >"$sign_log" 2>&1
  local rc=$?
  echo "$rc" >"$rc_file"
  if [[ "$rc" -eq 0 ]]; then
    echo "[ok]  $sign  (log: $sign_log)" | tee -a "$summary_log"
  else
    echo "[FAIL] $sign rc=$rc (log: $sign_log)" | tee -a "$summary_log"
  fi
  return 0
}

wait_for_slot() {
  while true; do
    local alive=0
    local pid
    for pid in "${pids[@]+"${pids[@]}"}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    if [[ "$alive" -lt "$MAX_JOBS" ]]; then
      break
    fi
    sleep 2
  done
}

pids=()
for sign in "${sign_list[@]}"; do
  wait_for_slot
  run_one "$sign" &
  pid=$!
  pids+=("$pid")
  echo "[spawn] $sign pid=$pid" | tee -a "$summary_log"
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

ok=0
fail=0
for sign in "${sign_list[@]}"; do
  slug="${sign//./_}"
  rc_file="$OUT_DIR/logs/${slug}_${ts}.rc"
  rc=1
  if [[ -f "$rc_file" ]]; then
    rc="$(cat "$rc_file")"
  fi
  if [[ "$rc" == "0" ]]; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi
done

{
  echo ""
  echo "=== done: ok=$ok fail=$fail out=$OUT_DIR ==="
  if [[ -d "$OUT_DIR/data" ]]; then
    echo "  routes in data/: $(find "$OUT_DIR/data" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  fi
} | tee -a "$summary_log"

exit "$fail"
