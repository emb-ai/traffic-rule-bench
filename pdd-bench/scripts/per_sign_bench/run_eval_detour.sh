#!/usr/bin/env bash
# run_eval_detour.sh — detour 4.2.x eval (копия run_eval_fast.sh, каталог-прямой
# режим): все валидные сцены 4.2.1/4.2.2/4.2.3 × 3 вариации, все бейзлайны.
# Каталог строится заранее:
#   python -m sumo_space.sumo_catalog --scenes-root <sdc>/pdd-bench/scenes \
#     --sign-categories 4.2.1,4.2.2,4.2.3 --n-per-category 250 --n-variations 3 \
#     --output $REPO/benchmark_output/detour_v1/catalog.jsonl
# Сплит «конусы+знак / только знак» уже в каталоге (поле detour_cones, 50/50).
#
# vs run_eval_8gpu.sh (which runs eval_pipeline per shard×stream, with each job
# running its stream's baselines SEQUENTIALLY and rebuilding metrics ~96x):
#   - every (policy, ego-variant) is its own job → all baselines run in parallel
#   - calls run_benchmark.py directly → sim only, no per-job build/aggregate/report
#   - GPU baselines pinned one per GPU (round-robin); CPU baselines in a CPU pool
#     running CONCURRENTLY with the GPU pool
#   - metrics built ONCE at the end from episodes_*.jsonl
#
#   nohup bash scripts/per_sign_bench/run_eval_detour.sh > /tmp/eval_detour.log 2>&1 & disown
#
# ---- config (EDIT for the server) -------------------------------------------
REPO=/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench
# каталог-прямой прогон (без материализации) — каталог и есть манифест
MANIFEST=$REPO/benchmark_output/detour_v1/catalog.jsonl
SCENES=/home/jovyan/shares/SR006.nfs2/smirnova/sdc/pdd-bench/scenes
OUT=$REPO/benchmark_output/detour_v1/eval_fast

# 2-GPU схема: симуляция CPU-bound (на 8 GPU утилизация ~5%), поэтому все NN
# консолидированы на GPU 0/1 с ~10 процессами на карту (см. policy_gpus ниже).
GPUS="0 1 2 3 4 5 6 7"      # fallback для policy_gpus(*); NN живут на "0 1"
NSHARDS=16                  # scene shards per baseline (more = finer balance, more ckpt loads)
CONCURRENCY=20              # NN-процессов суммарно на GPU 0+1 (~10/карту, CaRL-1B ~4-5 ГБ GPU
                            # каждый). 32 на 2 картах = OOM (rc=137); поднимать шагами по free -g
CONCURRENCY_CPU=24          # CPU-baseline workers; помнить: NN-джобы тоже жрут CPU (сим),
                            # суммарно CONCURRENCY+CONCURRENCY_CPU процессов симуляции
MAX_STEPS=1500
PROGRESS_INTERVAL=30        # seconds between progress lines in the eval log

# Detour manifest contains only 4.2.x — nothing to exclude.
EXCLUDE_CODES=""

PY=/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/plant2/bin/python
PLANT2_ACTION_MODE=pid

# ego-variants for IDM-family (idm, comprehensive_rule_expert). NN + rule_compliant
# + ppo_lidar always run once (default). Use "default" for a leaner/faster run.
EGO_VARIANTS="default,s1,s2,s3,s4"

# Baselines. Empty string disables that group.
NN_POLICIES="carl carl_rule plant2 plant2_rule"          # GPU, need a checkpoint
IDM_FAMILY_POLICIES="idm comprehensive_rule_expert"      # CPU, run per ego-variant
CPU_SINGLE_POLICIES="rule_compliant ppo_lidar"           # CPU, run once (default)

CARL_CKPT=/home/jovyan/shares/SR006.nfs2/smirnova/sdc/CaRL/nuPlan/checkpoints/nuplan_51479_1B/model_best.pth
PLANT_CKPT=/home/jovyan/shares/SR006.nfs2/smirnova/sdc/pdd-bench/checkpoints/epoch%3D029_final_3.ckpt
# -----------------------------------------------------------------------------

export PER_SIGN_COMPLIANT_NPC=1
# Правки v61 (переопределяемы из окружения): styles-сэмплер ego s1-s4,
# curve-aware база idm, «ego держит v0» для default-пары, carl-трекинг.
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

set -uo pipefail
cd "$REPO"
SHARD_DIR="$OUT/shards"
mkdir -p "$SHARD_DIR" "$OUT/logs" "$OUT/parts"
GPU_ARR=($GPUS)
BENCH=scripts/per_sign_bench

# ---- 0. drop end-of-zone codes (exact sign_code match) ----------------------
SRC_MANIFEST="$MANIFEST"
if [ -n "${EXCLUDE_CODES// /}" ]; then
  EXCL_RE=$(printf '%s' "$EXCLUDE_CODES" | sed -e 's/\./\\./g' -e 's/  */|/g')
  FMANIFEST="$OUT/manifest_filtered.jsonl"
  grep -Ev "\"sign_code\":[[:space:]]*\"(${EXCL_RE})\"" "$MANIFEST" > "$FMANIFEST" || true
  echo "manifest: $(wc -l < "$MANIFEST") -> $(wc -l < "$FMANIFEST") rows (dropped: $EXCLUDE_CODES)"
  SRC_MANIFEST="$FMANIFEST"
fi

# ---- 1. round-robin shards --------------------------------------------------
rm -f "$SHARD_DIR"/shard_*.jsonl
awk -v n="$NSHARDS" -v dir="$SHARD_DIR" 'NF{ f=sprintf("%s/shard_%02d.jsonl",dir,(NR-1)%n); print >> f }' "$SRC_MANIFEST"
echo "shards: $(ls "$SHARD_DIR"/shard_*.jsonl | wc -l) x ~$(( $(wc -l < "$SRC_MANIFEST") / NSHARDS )) rows"

# ---- 2. job lists: one per (policy, ego-variant, shard) ---------------------
# GPU spec = "gpu|policy|variant|model|shardfile|tag|runname"  (GPU pinned in front)
# CPU spec =     "policy|variant|model|shardfile|tag|runname"
gpu_jobs=(); cpu_jobs=()

# 2-GPU схема: ВСЕ NN-политики на GPU 0 и 1 (шарды round-robin между ними).
# Симуляция CPU-bound — карты легко тянут по ~10 процессов; больше карт не
# ускоряет, а размазывает. Старая схема (по паре карт на политику) — в git.
policy_gpus () {  # -> GPU list for a policy
  echo "0 1"
}

# GPU jobs interleaved by shard (outer) so all NN policies start TOGETHER, each on
# its own GPUs — instead of finishing carl, then carl_rule, then plant2, ...
for s in $(seq 0 $((NSHARDS-1))); do
  sf=$(printf "%s/shard_%02d.jsonl" "$SHARD_DIR" "$s")
  for p in $NN_POLICIES; do
    case "$p" in carl|carl_rule) ck="$CARL_CKPT";; plant2|plant2_rule) ck="$PLANT_CKPT";; *) ck="";; esac
    pg=($(policy_gpus "$p")); g=${pg[$(( s % ${#pg[@]} ))]}
    rn="${p}_default"; tag=$(printf "%s_s%02d" "$rn" "$s")
    gpu_jobs+=("$g|$p|default|$ck|$sf|$tag|$rn")
  done
done

add_cpu () {  # policy variant model
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
echo "gpu jobs: ${#gpu_jobs[@]} (NN x $NSHARDS, pinned per-policy); cpu jobs: ${#cpu_jobs[@]}"

# ---- 3. run one baseline-shard (run_benchmark.py directly, no metrics) -------
run_job () {  # gpu spec   (gpu="cpu" -> hide CUDA)
  local gpu="$1" spec="$2" policy variant model sf tag rn cvd rc
  IFS='|' read -r policy variant model sf tag rn <<< "$spec"
  # RESUME: skip a job that already finished cleanly (.ok marker); otherwise drop
  # any partial output from a killed/failed attempt so episodes don't double-count.
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

# ---- progress watcher: episodes_done/~expected + ETA, into the eval log -----
progress_watcher () {  # expected_episodes interval done_dir total_jobs
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

# ---- 4. two pools running CONCURRENTLY: GPU (NN) + CPU (rule-based) ----------
gpu_lane () { local li="$1" i spec gpu rest
  for i in "${!gpu_jobs[@]}"; do
    if [ $(( i % CONCURRENCY )) -eq "$li" ]; then
      spec="${gpu_jobs[$i]}"; gpu="${spec%%|*}"; rest="${spec#*|}"   # unpin GPU baked in front
      run_job "$gpu" "$rest"
    fi
  done; }
cpu_lane () { local li="$1" i
  for i in "${!cpu_jobs[@]}"; do
    [ $(( i % CONCURRENCY_CPU )) -eq "$li" ] && run_job cpu "${cpu_jobs[$i]}"
  done; }
# expected episodes ≈ manifest rows × number of baselines (for the progress bar)
# RESUME: keep .ok markers across runs so a relaunch skips finished jobs. Wipe with
# `rm -rf "$OUT/eval_fast"` (or just "$OUT/_done") only when you want a fresh start.
DONE_DIR="$OUT/_done"; mkdir -p "$DONE_DIR"
TOTAL_JOBS=$(( ${#gpu_jobs[@]} + ${#cpu_jobs[@]} ))
N_BASELINES=$(( $(echo $NN_POLICIES | wc -w) \
              + $(echo $IDM_FAMILY_POLICIES | wc -w) * $(echo "${EGO_VARIANTS//,/ }" | wc -w) \
              + $(echo $CPU_SINGLE_POLICIES | wc -w) ))
EXPECTED=$(( $(wc -l < "$SRC_MANIFEST") * N_BASELINES ))

NDONE=$(find "$DONE_DIR" -name '*.ok' 2>/dev/null | wc -l)
echo "=== launching $CONCURRENCY GPU + $CONCURRENCY_CPU CPU workers ($TOTAL_JOBS jobs, $NDONE already done -> resuming, ~$EXPECTED episodes) ==="
progress_watcher "$EXPECTED" "$PROGRESS_INTERVAL" "$DONE_DIR" "$TOTAL_JOBS" &
WATCHER=$!
trap 'kill "$WATCHER" 2>/dev/null' EXIT
for li in $(seq 0 $((CONCURRENCY-1)));      do gpu_lane "$li" & done
for li in $(seq 0 $((CONCURRENCY_CPU-1)));  do cpu_lane "$li" & done
wait
kill "$WATCHER" 2>/dev/null
echo "=== all baselines done ==="

# ---- 5. metrics ONCE: per-part CSV from episodes -> merge -> aggregate -> MD -
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
