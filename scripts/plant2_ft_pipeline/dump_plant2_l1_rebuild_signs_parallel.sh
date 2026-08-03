#!/usr/bin/env bash
# Parallel PlanT2 L1 rebuild with PDD signs in boxes (sharded).
#
# Machine assumed dedicated to this dump. Uses a global worker pool.
#
# Shard sizing (nproc=224, exclusive):
#   Each MetaDrive+SUMO worker often burns ~3 cores under load.
#   MAX_WORKERS = floor(0.80 * nproc / 3) ≈ 59 → clamp to 56 (headroom for
#   OS / NFS / occasional spikes). Override with MAX_WORKERS=.
#
# Usage:
#   bash dump_plant2_l1_rebuild_signs_parallel.sh
#   DRY_RUN=1 bash dump_plant2_l1_rebuild_signs_parallel.sh
#   MAX_WORKERS=32 SIGNS_EXP="detour" bash ...   # subset
#   JOBS="fv:3.24 fv:5.21" bash ...
#
# Outputs (new trees, does not clobber old dumps):
#   plant2_l1_from_experts_signs/
#   plant2_l1_traj_fv_nodeA_signs/
#   plant2_l1_lane_signs/   (all lane experts; no 300 cap)

set -u

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

SHEPELEV_ROOT="${SHEPELEV_ROOT:-$SHEPELEV}"
REPO="${REPO:-$TRB_ROOT}"
BENCH_DIR="$REPO/pdd-bench/scripts/per_sign_bench"
CT="${CT:-$PIPELINE_DIR}"

ZINK_BENCH="${ZINK_BENCH:-/mnt/virtual_ai0001053-01202_SR006-nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench/scripts/per_sign_bench}"
DETOUR_SCENES="${DETOUR_SCENES:-/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/sdc/pdd-bench/scenes}"
SM_MNT="${SM_MNT:-/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova}"
FV_SCENES="${FV_SCENES:-$SM_MNT/traffic-rule-bench/pdd-bench/scenes_balanced}"
FV_EXPERTS_SRC="${FV_EXPERTS_SRC:-$SM_MNT/experts_fv_train80/experts_scene_uid_top1.jsonl}"

OUT_EXP="${OUT_EXP:-$SHEPELEV_ROOT/plant2_l1_from_experts_signs}"
OUT_FV="${OUT_FV:-$SHEPELEV_ROOT/plant2_l1_traj_fv_nodeA_signs}"
OUT_LANE="${OUT_LANE:-$SHEPELEV_ROOT/plant2_l1_lane_signs}"

EXPERTS_RANK="${EXPERTS_RANK:-top1}"
EXPERTS_FILE="experts_scene_uid_${EXPERTS_RANK}.jsonl"
BACKENDS="${BACKENDS:-sumo}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_GIFS="${SAVE_GIFS:-0}"

NPROC="$(nproc)"
# ~3 cores/worker effective; keep ~20% headroom.
DEFAULT_MAX="$(python3 - <<PY
n=$NPROC
print(max(8, min(64, int(0.80 * n / 3))))
PY
)"
MAX_WORKERS="${MAX_WORKERS:-$DEFAULT_MAX}"
# Target scenes per shard (larger shards → fewer processes; smaller → better balance).
TARGET_PER_SHARD="${TARGET_PER_SHARD:-120}"

ARBELYAEV_PY="$SHEPELEV_ROOT/conda_envs/arbelyaev-sdc/bin/python"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ARBELYAEV_PY" ]]; then
    PYTHON="$ARBELYAEV_PY"
  elif [[ -x /home/user/conda/envs/zinkovich-sdc/bin/python ]]; then
    PYTHON=/home/user/conda/envs/zinkovich-sdc/bin/python
  else
    PYTHON=python3
  fi
fi

SCRIPT="$BENCH_DIR/expert_replay_inenv.py"
[[ -f "$SCRIPT" ]] || { echo "ERROR: missing $SCRIPT" >&2; exit 1; }

LOG_ROOT="$SHEPELEV_ROOT/collected_trajectories/logs_dump_signs"
mkdir -p "$LOG_ROOT" "$OUT_EXP/logs" "$OUT_FV/logs" "$OUT_FV/experts" "$OUT_LANE/logs"
ts="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_ROOT/rebuild_signs_${ts}.log"
task_list="$LOG_ROOT/tasks_${ts}.tsv"

{
  echo "================================================================"
  echo "dump_plant2_l1_rebuild_signs_parallel  [$ts]"
  echo "  nproc         = $NPROC"
  echo "  MAX_WORKERS   = $MAX_WORKERS  (default from 0.80*nproc/3, capped 64)"
  echo "  TARGET/SHARD  = $TARGET_PER_SHARD"
  echo "  PYTHON        = $PYTHON"
  echo "  OUT_EXP       = $OUT_EXP"
  echo "  OUT_FV        = $OUT_FV"
  echo "  OUT_LANE      = $OUT_LANE"
  echo "  DRY_RUN       = $DRY_RUN"
  echo "  log           = $summary_log"
  echo "================================================================"
} | tee "$summary_log"

# --- prepare FV experts (nodeA filter), same as dump_plant2_l1_traj_fv_nodeA.sh ---
"$PYTHON" - <<PY
import json
from pathlib import Path

src = Path("$FV_EXPERTS_SRC")
out_dir = Path("$OUT_FV/experts")
out_dir.mkdir(parents=True, exist_ok=True)
signs = ["3.24", "4.6", "5.21", "5.31"]
old = "/home/jovyan/shares/SR006.nfs2/smirnova/"
new = "/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/"
PATH_KEYS = ("pkl_path", "sidecar_path", "gif_path", "winning_pkl", "winning_sidecar")

def remap(v):
    if isinstance(v, str) and v.startswith(old):
        return new + v[len(old):]
    return v

by = {s: [] for s in signs}
if not src.is_file():
    print(f"WARN: FV experts missing: {src}")
else:
    for line in open(src):
        o = json.loads(line)
        pkl = o.get("pkl_path") or ""
        if "nodeA" not in pkl:
            continue
        for k in PATH_KEYS:
            if k in o and o[k]:
                o[k] = remap(o[k])
        sign = str(o.get("sign") or "")
        if sign not in by:
            continue
        p = Path(o.get("pkl_path") or "")
        if not p.exists():
            continue
        by[sign].append(o)
for sign, rows in by.items():
    path = out_dir / f"experts_{sign.replace('.', '_')}_top1.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"FV wrote {path.name} n={len(rows)}")
PY

# job definitions: name | experts | scenes | out_dir | count_override(empty=all)
declare -a JOB_LINES=()

add_job() {
  local name="$1" experts="$2" scenes="$3" out="$4" count_ov="${5:-}"
  if [[ ! -f "$experts" ]]; then
    echo "[skip] $name: missing experts $experts" | tee -a "$summary_log"
    return
  fi
  if [[ ! -d "$scenes" ]]; then
    echo "[skip] $name: missing scenes $scenes" | tee -a "$summary_log"
    return
  fi
  local n
  n="$(wc -l <"$experts" | tr -d ' ')"
  if [[ -n "$count_ov" && "$count_ov" -lt "$n" ]]; then
    n="$count_ov"
  fi
  if [[ "$n" -le 0 ]]; then
    echo "[skip] $name: n=0" | tee -a "$summary_log"
    return
  fi
  JOB_LINES+=("$name|$experts|$scenes|$out|$n")
}

# Priority / detour experts
add_job "exp:yield" \
  "$CT/traj-priority-signs/traj_yield_2_4_train80/experts/$EXPERTS_FILE" \
  "$ZINK_BENCH/yield_sign/scenes/2_4" "$OUT_EXP"
add_job "exp:stop" \
  "$CT/traj-priority-signs/traj_stop_2_5_train80/experts/$EXPERTS_FILE" \
  "$ZINK_BENCH/stop_sign/scenes/2_5" "$OUT_EXP"
add_job "exp:secondary" \
  "$CT/traj-priority-signs/traj_secondary_2_3_train80/experts/$EXPERTS_FILE" \
  "$ZINK_BENCH/secondary_sign/scenes/2_3" "$OUT_EXP"
add_job "exp:main" \
  "$CT/traj-priority-signs/traj_main_2_1_train80/experts/$EXPERTS_FILE" \
  "$ZINK_BENCH/main_sign/scenes/2_1" "$OUT_EXP"
add_job "exp:roundabout" \
  "$CT/traj-priority-signs/traj_roundabout_4_3_train80/experts/$EXPERTS_FILE" \
  "$ZINK_BENCH/roundabout_sign/scenes/4_3" "$OUT_EXP"
add_job "exp:detour" \
  "$CT/traffic-rule-bench-traj/experts_detour_train80/$EXPERTS_FILE" \
  "$DETOUR_SCENES" "$OUT_EXP"

# FV
add_job "fv:3.24" "$OUT_FV/experts/experts_3_24_top1.jsonl" "$FV_SCENES" "$OUT_FV"
add_job "fv:4.6"  "$OUT_FV/experts/experts_4_6_top1.jsonl"  "$FV_SCENES" "$OUT_FV"
add_job "fv:5.21" "$OUT_FV/experts/experts_5_21_top1.jsonl" "$FV_SCENES" "$OUT_FV"
add_job "fv:5.31" "$OUT_FV/experts/experts_5_31_top1.jsonl" "$FV_SCENES" "$OUT_FV"

# Lane: all experts (no count cap)
add_job "lane:5.15.1" \
  "$CT/traj-priority-signs/traj_lane_5_15_train80/experts/$EXPERTS_FILE" \
  "$ZINK_BENCH/lane_direction_signs/scenes/5_15_1" "$OUT_LANE"

# Optional filter: JOBS="exp:detour fv:3.24"
if [[ -n "${JOBS:-}" ]]; then
  declare -a FILTERED=()
  for line in "${JOB_LINES[@]}"; do
    name="${line%%|*}"
    for want in $JOBS; do
      if [[ "$name" == "$want" ]]; then
        FILTERED+=("$line")
        break
      fi
    done
  done
  JOB_LINES=("${FILTERED[@]}")
fi

# Build shard task list: slug start count experts scenes out
: >"$task_list"
total_scenes=0
total_shards=0
for line in "${JOB_LINES[@]}"; do
  IFS='|' read -r name experts scenes out n <<<"$line"
  total_scenes=$((total_scenes + n))
  n_shards=$(( (n + TARGET_PER_SHARD - 1) / TARGET_PER_SHARD ))
  # at least 1, never more than n or MAX_WORKERS*2 (avoid tiny shards explosion)
  if [[ "$n_shards" -lt 1 ]]; then n_shards=1; fi
  if [[ "$n_shards" -gt "$n" ]]; then n_shards="$n"; fi
  if [[ "$n_shards" -gt $((MAX_WORKERS * 2)) ]]; then
    n_shards=$((MAX_WORKERS * 2))
  fi
  slug="${name//:/_}"
  slug="${slug//./_}"
  echo "  job $name n=$n -> shards=$n_shards (~$((n / n_shards)) scenes/shard)" | tee -a "$summary_log"
  for ((s=0; s<n_shards; s++)); do
    st=$(( s * n / n_shards ))
    en=$(( (s + 1) * n / n_shards ))
    cnt=$(( en - st ))
    [[ "$cnt" -gt 0 ]] || continue
    echo -e "${slug}_s${s}\t${st}\t${cnt}\t${experts}\t${scenes}\t${out}\t${name}" >>"$task_list"
    total_shards=$((total_shards + 1))
  done
done

echo "PLAN total_scenes=$total_scenes total_shards=$total_shards MAX_WORKERS=$MAX_WORKERS" | tee -a "$summary_log"
echo "task_list=$task_list" | tee -a "$summary_log"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] first 15 tasks:" | tee -a "$summary_log"
  head -15 "$task_list" | tee -a "$summary_log"
  echo "[dry-run] exit" | tee -a "$summary_log"
  exit 0
fi

run_task() {
  local slug="$1" start="$2" count="$3" experts="$4" scenes="$5" out="$6" name="$7"
  mkdir -p "$out/logs"
  local sign_log="$out/logs/${slug}_${ts}.log"
  local rc_file="$out/logs/${slug}_${ts}.rc"
  local -a cmd=(
    "$PYTHON" "$SCRIPT"
    --experts "$experts"
    --scenes-root "$scenes"
    --save-plant2-dir "$out"
    --backends "$BACKENDS"
    --ego-mode recorded
    --npc-mode recorded
    --start "$start"
    --count "$count"
  )
  if [[ "$SAVE_GIFS" == "1" ]]; then
    cmd+=(--save-gifs --gif-dir "$out/gifs/$slug")
  fi
  echo "[run] $name slug=$slug start=$start count=$count" | tee -a "$summary_log"
  ( cd "$BENCH_DIR" && "${cmd[@]}" ) >"$sign_log" 2>&1
  local rc=$?
  echo "$rc" >"$rc_file"
  if [[ "$rc" -eq 0 ]]; then
    echo "[ok]  $slug (log: $sign_log)" | tee -a "$summary_log"
  else
    echo "[FAIL] $slug rc=$rc (log: $sign_log)" | tee -a "$summary_log"
  fi
  return 0
}

# Global pool
declare -a pids=()
declare -A pid_slug=()

wait_slot() {
  while true; do
    local alive=0
    local pid
    for pid in "${pids[@]+"${pids[@]}"}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    if [[ "$alive" -lt "$MAX_WORKERS" ]]; then
      break
    fi
    sleep 2
  done
}

while IFS=$'\t' read -r slug start count experts scenes out name; do
  [[ -n "$slug" ]] || continue
  wait_slot
  run_task "$slug" "$start" "$count" "$experts" "$scenes" "$out" "$name" &
  pid=$!
  pids+=("$pid")
  pid_slug["$pid"]="$slug"
  echo "[spawn] $slug pid=$pid start=$start count=$count workers_cap=$MAX_WORKERS" | tee -a "$summary_log"
done <"$task_list"

for pid in "${pids[@]+"${pids[@]}"}"; do
  wait "$pid" || true
done

ok=0
fail=0
while IFS=$'\t' read -r slug start count experts scenes out name; do
  rc_file="$out/logs/${slug}_${ts}.rc"
  rc=1
  [[ -f "$rc_file" ]] && rc="$(cat "$rc_file")"
  if [[ "$rc" == "0" ]]; then ok=$((ok + 1)); else fail=$((fail + 1)); fi
done <"$task_list"

{
  echo ""
  echo "=== done: ok=$ok fail=$fail shards=$total_shards ==="
  for d in "$OUT_EXP" "$OUT_FV" "$OUT_LANE"; do
    if [[ -d "$d/data" ]]; then
      echo "  $(basename "$d") routes=$(find "$d/data" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
    fi
  done
} | tee -a "$summary_log"

exit "$fail"
