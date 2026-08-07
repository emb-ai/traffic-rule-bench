#!/usr/bin/env bash
# Plant2-FT-only wrapper around run_eval_fast pattern:
# NN_POLICIES=plant2, PLANT_CKPT from env, per-ckpt OUT dir.
#
# Required env:
#   CKPT  — absolute path to plant2-ft .ckpt
#   OUT   — metrics output directory for this checkpoint
# Optional:
#   TAG, GPUS, NSHARDS, CONCURRENCY, MANIFEST, SCENES, PY, REPO
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

REPO=${REPO:-$SHEPELEV/traffic-rule-bench/pdd-bench}
NFS2=/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/traffic-rule-bench/pdd-bench

# FT eval uses catalog_fv_test20 ONLY (~3.3k rows). Never the full catalog.jsonl (~27k).
MANIFEST=${MANIFEST:-$NFS2/benchmark_output_speed/balanced/run_v61_a6/catalog_fv_test20.jsonl}
SCENES=${SCENES:-$NFS2/scenes_balanced}
OUT=${OUT:?set OUT to per-checkpoint metrics dir}
CKPT=${CKPT:?set CKPT to plant2-ft checkpoint}

GPUS=${GPUS:-"0 1 2 3 4 5 6"}
NSHARDS=${NSHARDS:-28}
CONCURRENCY=${CONCURRENCY:-28}
CONCURRENCY_CPU=${CONCURRENCY_CPU:-0}
MAX_STEPS=${MAX_STEPS:-1500}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-30}
EXCLUDE_CODES=${EXCLUDE_CODES:-"3.25 5.22 5.32"}

PY=${PY:-$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python}
PLANT2_ACTION_MODE=${PLANT2_ACTION_MODE:-pid}

# plant2 base only (FT weights via PLANT_CKPT)
NN_POLICIES=${NN_POLICIES:-plant2}
IDM_FAMILY_POLICIES=${IDM_FAMILY_POLICIES:-}
CPU_SINGLE_POLICIES=${CPU_SINGLE_POLICIES:-}
EGO_VARIANTS=${EGO_VARIANTS:-default}

PLANT_CKPT="$CKPT"
CARL_CKPT=${CARL_CKPT:-}

export PER_SIGN_COMPLIANT_NPC=1
export EGO_SAMPLER="${EGO_SAMPLER:-styles}"
export EGO_CURVE_AWARE="${EGO_CURVE_AWARE:-1}"
export EGO_HOLD_V0="${EGO_HOLD_V0:-1}"
export CARL_LONGITUDINAL="${CARL_LONGITUDINAL:-tracking}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 TORCH_NUM_THREADS=1
export SDL_AUDIODRIVER=dummy
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null; chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

test -f "$CKPT" || { echo "missing CKPT=$CKPT"; exit 1; }
test -f "$MANIFEST" || { echo "missing MANIFEST=$MANIFEST"; exit 1; }
test -d "$SCENES" || { echo "missing SCENES=$SCENES"; exit 1; }
test -x "$PY" || { echo "missing PY=$PY"; exit 1; }
case "$MANIFEST" in
  *catalog_fv_test20.jsonl) ;;
  *)
    echo "ERROR: MANIFEST must be catalog_fv_test20.jsonl, got: $MANIFEST" >&2
    exit 1
    ;;
esac
MANIFEST_ROWS=$(wc -l < "$MANIFEST")
if (( MANIFEST_ROWS > 5000 )); then
  echo "ERROR: MANIFEST has $MANIFEST_ROWS rows (>5000); use catalog_fv_test20, not full catalog" >&2
  exit 1
fi

cd "$REPO"
SHARD_DIR="$OUT/shards"
mkdir -p "$SHARD_DIR" "$OUT/logs" "$OUT/parts"
GPU_ARR=($GPUS)
BENCH=scripts/per_sign_bench

echo "=== run_eval_fast plant2-ft $(date -Is) ==="
echo "  CKPT=$CKPT"
echo "  OUT=$OUT"
echo "  MANIFEST=$MANIFEST ($(wc -l < "$MANIFEST") rows)"
echo "  SCENES=$SCENES"
echo "  NN_POLICIES=$NN_POLICIES  GPUS=$GPUS NSHARDS=$NSHARDS CONCURRENCY=$CONCURRENCY"

SRC_MANIFEST="$MANIFEST"
if [ -n "${EXCLUDE_CODES// /}" ]; then
  EXCL_RE=$(printf '%s' "$EXCLUDE_CODES" | sed -e 's/\./\\./g' -e 's/  */|/g')
  FMANIFEST="$OUT/manifest_filtered.jsonl"
  grep -Ev "\"sign_code\":[[:space:]]*\"(${EXCL_RE})\"" "$MANIFEST" > "$FMANIFEST" || true
  echo "manifest: $(wc -l < "$MANIFEST") -> $(wc -l < "$FMANIFEST") rows (dropped: $EXCLUDE_CODES)"
  SRC_MANIFEST="$FMANIFEST"
fi

rm -f "$SHARD_DIR"/shard_*.jsonl
awk -v n="$NSHARDS" -v dir="$SHARD_DIR" 'NF{ f=sprintf("%s/shard_%02d.jsonl",dir,(NR-1)%n); print >> f }' "$SRC_MANIFEST"
echo "shards: $(ls "$SHARD_DIR"/shard_*.jsonl | wc -l) x ~$(( $(wc -l < "$SRC_MANIFEST") / NSHARDS )) rows"

gpu_jobs=(); cpu_jobs=()
policy_gpus () { echo "$GPUS"; }

for s in $(seq 0 $((NSHARDS-1))); do
  sf=$(printf "%s/shard_%02d.jsonl" "$SHARD_DIR" "$s")
  for p in $NN_POLICIES; do
    case "$p" in carl|carl_rule) ck="$CARL_CKPT";; plant2|plant2_rule) ck="$PLANT_CKPT";; *) ck="";; esac
    pg=($(policy_gpus "$p")); g=${pg[$(( s % ${#pg[@]} ))]}
    rn="${p}_default"; tag=$(printf "%s_s%02d" "$rn" "$s")
    gpu_jobs+=("$g|$p|default|$ck|$sf|$tag|$rn")
  done
done

add_cpu () {
  local policy="$1" variant="$2" model="$3" s sf tag rn
  rn="${policy}_${variant}"
  for s in $(seq 0 $((NSHARDS-1))); do
    sf=$(printf "%s/shard_%02d.jsonl" "$SHARD_DIR" "$s")
    tag=$(printf "%s_s%02d" "$rn" "$s")
    cpu_jobs+=("$policy|$variant|$model|$sf|$tag|$rn")
  done
}
for p in $IDM_FAMILY_POLICIES; do
  for v in ${EGO_VARIANTS//,/ }; do add_cpu "$p" "$v" ""; done
done
for p in $CPU_SINGLE_POLICIES; do add_cpu "$p" default ""; done
echo "gpu jobs: ${#gpu_jobs[@]}; cpu jobs: ${#cpu_jobs[@]}"

DONE_DIR="$OUT/_done"; mkdir -p "$DONE_DIR"

run_job () {
  local gpu="$1" spec="$2" policy variant model sf tag rn cvd rc
  IFS='|' read -r policy variant model sf tag rn <<< "$spec"
  if [ -f "$DONE_DIR/$tag.ok" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $tag (already done)"; return
  fi
  rm -rf "$OUT/parts/$tag" "$DONE_DIR/$tag.fail"
  cvd="$gpu"; [ "$gpu" = cpu ] && cvd=""
  local args=( --policy "$policy" --run-name "$rn" --ego-variant "$variant"
    --manifest "$sf" --scenes-root "$SCENES" --backends sumo
    --max-steps "$MAX_STEPS" --benchmark-output "$OUT/parts/$tag/benchmark" )
  [ -n "$model" ] && args+=( --model-path "$model" )
  case "$policy" in plant2|plant2_rule) args+=( --plant2-action-mode "$PLANT2_ACTION_MODE" );; esac
  echo "[$(date +%H:%M:%S)] gpu=$gpu START $tag ($(wc -l < "$sf") scenes)"
  CUDA_VISIBLE_DEVICES="$cvd" "$PY" "$BENCH/run_benchmark.py" "${args[@]}" \
    >> "$OUT/logs/$tag.log" 2>&1
  rc=$?
  : > "$DONE_DIR/$tag.$([ "$rc" -eq 0 ] && echo ok || echo fail)"
  echo "[$(date +%H:%M:%S)] gpu=$gpu DONE  $tag rc=$rc"
}

progress_watcher () {
  local expected="$1" interval="$2" done_dir="$3" total_jobs="$4" t0; t0=$(date +%s)
  while sleep "$interval"; do
    local ep jobs now el rate pct eta rem es bars
    ep=$(find "$OUT/parts" -name 'episodes_*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
    jobs=$(find "$done_dir" -type f 2>/dev/null | wc -l)
    now=$(date +%s); el=$((now - t0)); rate=0
    [ "$el" -gt 0 ] && rate=$(awk -v n="$ep" -v e="$el" 'BEGIN{printf "%.1f", n*60/e}')
    pct=0; [ "$expected" -gt 0 ] && pct=$(awk -v n="$ep" -v t="$expected" 'BEGIN{p=100*n/t; if(p>100)p=100; printf "%.1f", p}')
    eta="?"
    if [ "$ep" -gt 0 ] && [ "$ep" -lt "$expected" ]; then
      rem=$((expected - ep))
      es=$(awk -v r="$rem" -v rt="$rate" 'BEGIN{if(rt>0)printf "%d", r*60/rt; else print 0}')
      eta="$((es/3600))h$(((es%3600)/60))m"
    fi
    bars=$(awk -v p="$pct" 'BEGIN{n=int(p/2);s="";for(i=0;i<n;i++)s=s"#";for(i=n;i<50;i++)s=s".";print s}')
    echo "[progress $(date +%H:%M:%S)] ep ${ep}/~${expected} (${pct}%) [${bars}] jobs ${jobs}/${total_jobs} | ${rate} ep/min | ETA=${eta}"
    [ "$jobs" -ge "$total_jobs" ] && break
  done
}

gpu_lane () { local li="$1" i spec gpu rest
  for i in "${!gpu_jobs[@]}"; do
    if [ $(( i % CONCURRENCY )) -eq "$li" ]; then
      spec="${gpu_jobs[$i]}"; gpu="${spec%%|*}"; rest="${spec#*|}"
      run_job "$gpu" "$rest"
    fi
  done; }
cpu_lane () { local li="$1" i
  for i in "${!cpu_jobs[@]}"; do
    [ $(( i % CONCURRENCY_CPU )) -eq "$li" ] && run_job cpu "${cpu_jobs[$i]}"
  done; }

TOTAL_JOBS=$(( ${#gpu_jobs[@]} + ${#cpu_jobs[@]} ))
N_BASELINES=$(( $(echo $NN_POLICIES | wc -w) \
              + $(echo $IDM_FAMILY_POLICIES | wc -w) * $(echo "${EGO_VARIANTS//,/ }" | wc -w) \
              + $(echo $CPU_SINGLE_POLICIES | wc -w) ))
EXPECTED=$(( $(wc -l < "$SRC_MANIFEST") * N_BASELINES ))

NDONE=$(find "$DONE_DIR" -name '*.ok' 2>/dev/null | wc -l)
echo "=== launching $CONCURRENCY GPU + $CONCURRENCY_CPU CPU workers ($TOTAL_JOBS jobs, $NDONE done, ~$EXPECTED episodes) ==="
progress_watcher "$EXPECTED" "$PROGRESS_INTERVAL" "$DONE_DIR" "$TOTAL_JOBS" &
WATCHER=$!
trap 'kill "$WATCHER" 2>/dev/null' EXIT
for li in $(seq 0 $((CONCURRENCY-1))); do gpu_lane "$li" & done
if [ "$CONCURRENCY_CPU" -gt 0 ] && [ "${#cpu_jobs[@]}" -gt 0 ]; then
  for li in $(seq 0 $((CONCURRENCY_CPU-1))); do cpu_lane "$li" & done
fi
wait
kill "$WATCHER" 2>/dev/null || true
echo "=== all baselines done ==="

EMPTY="$OUT/_no_manifests"; mkdir -p "$EMPTY"
COMB="$OUT/metrics_per_episode.csv"; : > "$COMB"
while read -r pe; do
  [ -n "$pe" ] || continue
  [ -n "$(find "$pe" -name 'episodes_*.jsonl' -print -quit 2>/dev/null)" ] || continue
  tag=$(basename "$(dirname "$(dirname "$pe")")")
  csv="$OUT/parts/_csv_$tag.csv"
  "$PY" "$BENCH/build_episode_metrics_csv.py" --episodes-root "$pe" \
      --out "$csv" --manifests-root "$EMPTY" >/dev/null 2>&1 || continue
  if [ -s "$COMB" ]; then tail -n +2 "$csv" >> "$COMB"; else cat "$csv" > "$COMB"; fi
done < <(find "$OUT/parts" -type d -name policy_eval | sort)
echo "combined CSV: $COMB ($(($(wc -l < "$COMB")-1)) episodes)"

"$PY" "$BENCH/aggregate_episode_metrics.py" --csv "$COMB" --out-dir "$OUT"
"$PY" "$BENCH/generate_cumulative_markdown_report.py" --run-root "$OUT"
echo "REPORT: $OUT/reports/report_cumulative.md"
echo "=== DONE $(date -Is) ==="
