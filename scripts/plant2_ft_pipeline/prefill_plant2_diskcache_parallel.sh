#!/usr/bin/env bash
# Parallel diskcache prefill with augment=True (base + *_aug keys).
#
# Dedicated machine: nproc=224. Prefill is lighter than MetaDrive dump but
# contends on diskcache locks + /tmp write bandwidth.
#   MAX_WORKERS = floor(0.85 * nproc / 2) ≈ 95 → clamp 96.
# Override: MAX_WORKERS=64 bash ...
#
# RAM: diskcache lives on /tmp disk (~1.5–1.6 TiB for full aug). Worker RSS is
# modest; 2.0 TiB RAM is enough. CACHE_SIZE_GB default 1800.
#
# Usage (after train/val split exists):
#   export DS=.../plant2_l1_fv_experts_split_signs/train
#   export DS_VAL=.../plant2_l1_fv_experts_split_signs/val
#   export DS_LOCAL=/tmp/plant2_ds_cache_spatial_aug
#   bash prefill_plant2_diskcache_parallel.sh
#
#   DRY_RUN=1 bash prefill_plant2_diskcache_parallel.sh
#   MAX_WORKERS=64 bash prefill_plant2_diskcache_parallel.sh

set -u

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

SHEPELEV_ROOT="${SHEPELEV_ROOT:-$SHEPELEV}"
CT="${CT:-$PIPELINE_DIR}"
SPLIT_ROOT="${SPLIT_ROOT:-$SHEPELEV_ROOT/plant2_l1_fv_experts_split_signs}"

export DS="${DS:-$SPLIT_ROOT/train}"
export DS_VAL="${DS_VAL:-$SPLIT_ROOT/val}"
export DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_spatial_aug}"
export CACHE_SIZE_GB="${CACHE_SIZE_GB:-1800}"
export PREFILL_AUGMENT="${PREFILL_AUGMENT:-1}"
export PREFILL_STOP_FRAC="${PREFILL_STOP_FRAC:-0.97}"
export PREFILL_LOG_EVERY="${PREFILL_LOG_EVERY:-1000}"

NPROC="$(nproc)"
DEFAULT_MAX="$(python3 - <<PY
n=$NPROC
# Cap concurrent workers. Prefer 32 over 96: each worker builds full PlanTDataset
# index on NFS; too many simultaneous inits thrash the share.
# Override: MAX_WORKERS=64 bash ...
print(max(8, min(32, int(0.50 * n / 4))))
PY
)"
MAX_WORKERS="${MAX_WORKERS:-$DEFAULT_MAX}"
SPAWN_STAGGER_SEC="${SPAWN_STAGGER_SEC:-3}"
DRY_RUN="${DRY_RUN:-0}"

ARBELYAEV_PY="$SHEPELEV_ROOT/conda_envs/arbelyaev-sdc/bin/python"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ARBELYAEV_PY" ]]; then
    PYTHON="$ARBELYAEV_PY"
  else
    PYTHON=python3
  fi
fi

PREFILL_PY="$CT/prefill_plant2_diskcache.py"
[[ -f "$PREFILL_PY" ]] || { echo "ERROR: missing $PREFILL_PY" >&2; exit 1; }

mkdir -p "$DS_LOCAL" /tmp/plant2_prefill_logs
ts="$(date +%Y%m%d_%H%M%S)"
summary_log="/tmp/plant2_prefill_logs/parallel_${ts}.log"

# Fast sample count (mirrors PlanTDataset index formula without results/slurm I/O).
# samples/route = max(0, n_boxes - wps_len - seq_len - 7) with wps=8, seq=1 → n_boxes-16
_len_ds() {
  local root="$1"
  "$PYTHON" - <<PY
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

root = Path("$root".rstrip("/")) / "data"
# PlanT.yaml: wps_len=8, seq_len=1; range(5, n - wps - seq - 2) → n-16 samples
OVERHEAD = 16
routes = [p for p in root.iterdir() if p.is_dir() and (p / "boxes").is_dir()]

def count_one(p: Path) -> int:
    try:
        n = sum(1 for _ in (p / "boxes").iterdir())
    except OSError:
        return 0
    return max(0, n - OVERHEAD)

total = 0
workers = min(32, max(4, (os.cpu_count() or 8) // 4))
with ProcessPoolExecutor(max_workers=workers) as ex:
    futs = [ex.submit(count_one, r) for r in routes]
    for fut in as_completed(futs):
        total += int(fut.result())
print(total)
PY
}

echo "[$(date -Is)] probing TRAIN_N (fast parallel boxes count) …"
TRAIN_N="$(_len_ds "$DS")"
echo "[$(date -Is)] TRAIN_N=$TRAIN_N"

VAL_N=0
if [[ -d "${DS_VAL}/data" ]]; then
  echo "[$(date -Is)] probing VAL_N …"
  VAL_N="$(_len_ds "$DS_VAL")"
  echo "[$(date -Is)] VAL_N=$VAL_N"
fi

# Persist for reuse / debugging
echo "$TRAIN_N" >"/tmp/plant2_prefill_logs/train_n_${ts}.txt"
echo "$VAL_N" >"/tmp/plant2_prefill_logs/val_n_${ts}.txt"

{
  echo "================================================================"
  echo "prefill_plant2_diskcache_parallel  [$ts]"
  echo "  nproc         = $NPROC"
  echo "  MAX_WORKERS   = $MAX_WORKERS"
  echo "  DS            = $DS  (n=$TRAIN_N)"
  echo "  DS_VAL        = $DS_VAL  (n=$VAL_N)"
  echo "  DS_LOCAL      = $DS_LOCAL"
  echo "  CACHE_SIZE_GB = $CACHE_SIZE_GB"
  echo "  PREFILL_AUGMENT=$PREFILL_AUGMENT"
  echo "  df /tmp:"
  df -h /tmp | tail -1
  echo "  free:"
  free -h | awk '/Mem:/{print}'
  echo "================================================================"
} | tee "$summary_log"

# Train shards + one val job
declare -a TASKS=()
n_shards="$MAX_WORKERS"
if [[ "$TRAIN_N" -lt "$n_shards" ]]; then
  n_shards="$TRAIN_N"
fi
[[ "$n_shards" -ge 1 ]] || n_shards=1

for ((s=0; s<n_shards; s++)); do
  st=$(( s * TRAIN_N / n_shards ))
  en=$(( (s + 1) * TRAIN_N / n_shards ))
  [[ "$en" -gt "$st" ]] || continue
  TASKS+=("train|$st|$en")
done
if [[ "$VAL_N" -gt 0 ]]; then
  TASKS+=("val|0|$VAL_N")
fi

echo "PLAN train_shards=$n_shards val_job=$([ "$VAL_N" -gt 0 ] && echo 1 || echo 0) total_tasks=${#TASKS[@]}" | tee -a "$summary_log"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%s\n' "${TASKS[@]}" | head -20 | tee -a "$summary_log"
  echo "[dry-run] exit" | tee -a "$summary_log"
  exit 0
fi

run_one() {
  local split="$1" start="$2" end="$3" idx="$4"
  local logf="/tmp/plant2_prefill_logs/w${idx}_${split}_${start}_${end}_${ts}.log"
  local rc_file="/tmp/plant2_prefill_logs/w${idx}_${ts}.rc"
  (
    export PREFILL_SPLIT="$split"
    export PREFILL_START="$start"
    export PREFILL_END="$end"
    export PREFILL_LOG="$logf"
    export DS DS_VAL DS_LOCAL CACHE_SIZE_GB PREFILL_AUGMENT PREFILL_STOP_FRAC PREFILL_LOG_EVERY
    "$PYTHON" "$PREFILL_PY"
  ) >"${logf}.outer" 2>&1
  local rc=$?
  echo "$rc" >"$rc_file"
  if [[ "$rc" -eq 0 ]]; then
    echo "[ok]  $split [$start,$end) log=$logf" | tee -a "$summary_log"
  else
    echo "[FAIL] $split [$start,$end) rc=$rc log=$logf" | tee -a "$summary_log"
  fi
}

# Cap concurrent workers at MAX_WORKERS (val is an extra task).
declare -a pids=()
idx=0
for spec in "${TASKS[@]}"; do
  IFS='|' read -r split start end <<<"$spec"
  # wait for slot
  while true; do
    alive=0
    for pid in "${pids[@]+"${pids[@]}"}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    if [[ "$alive" -lt "$MAX_WORKERS" ]]; then break; fi
    sleep 2
  done
  run_one "$split" "$start" "$end" "$idx" &
  pids+=("$!")
  echo "[spawn] #$idx $split [$start,$end) pid=$!" | tee -a "$summary_log"
  idx=$((idx + 1))
  # stagger dataset inits on NFS
  if [[ "$SPAWN_STAGGER_SEC" -gt 0 ]]; then
    sleep "$SPAWN_STAGGER_SEC"
  fi
done

for pid in "${pids[@]+"${pids[@]}"}"; do
  wait "$pid" || true
done

ok=0
fail=0
for ((i=0; i<idx; i++)); do
  rc_file="/tmp/plant2_prefill_logs/w${i}_${ts}.rc"
  rc=1
  [[ -f "$rc_file" ]] && rc="$(cat "$rc_file")"
  if [[ "$rc" == "0" ]]; then ok=$((ok + 1)); else fail=$((fail + 1)); fi
done

{
  echo ""
  echo "=== prefill done: ok=$ok fail=$fail ==="
  if [[ -d "$DS_LOCAL" ]]; then
    du -sh "$DS_LOCAL" 2>/dev/null || true
  fi
} | tee -a "$summary_log"

exit "$fail"
