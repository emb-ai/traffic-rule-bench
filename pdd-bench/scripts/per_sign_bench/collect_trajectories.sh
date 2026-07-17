#!/usr/bin/env bash
# collect_trajectories.sh — порт collect_sumo_trajectories.sh на новый репо.
#
# Сбор траекторий по всем знакам: fan-out expert_replay.py по
# (политика × знак × фаза sumo/pgmap/citymap), merge all_runs.jsonl, oracle.
#
# expert_replay.py теперь разделяет цикл эпизода и все метрики с исправленным
# эвалом (run_benchmark.py + bench/), поэтому и имена политик здесь — из
# run_benchmark: comprehensive_rule_expert, rule_compliant, idm, ppo_lidar,
# carl_rule/plant2_rule (бывшие carl/plant2 старого рекордера; голые carl и
# plant2 — plain-политики без знания знаков).
#
# Источник сцен — либо старый layout FULL_DIR/<sign>/real_manifest.jsonl,
# либо НОВЫЙ режим: комбинированный манифест MANIFEST=<path.jsonl>, который
# скрипт сам сплитит по sign_code (только SUMO-фаза).
#
# Запуск (активировать conda env ДО запуска — все политики, включая
# carl/plant2, гоняются из этого env):
#   MANIFEST=$PWD/../../benchmark_output/new/sumo_manifest.jsonl ./collect_trajectories.sh
#
#   N_WORKERS=64 POLICIES_CPU="comprehensive_rule_expert" ./collect_trajectories.sh
#   SKIP_CARL=0 CARL_CKPT=/path/to/ckpt GPU_IDS=0,1 ./collect_trajectories.sh
#
# ВНИМАНИЕ (несовместимость со старыми прогонами):
#   * total_violations/violations_by_class в all_runs.jsonl теперь per-step;
#     event-счётчики — в violations_event_count/violations_by_class_event;
#   * scene_uid всегда с суффиксом _vN (формула run_benchmark) и вариантные
#     папки называются comprehensive_rule_expert_* — RESUME=1 против деревьев
#     старого рекордера не сматчится, используйте свежий OUT_BASE;
#   * s1..s4-варианты сэмплируются per-scene (seed_base + seed + k*1000003),
#     а не одним глобальным сэмплом на k.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${SCRIPT_DIR}/expert_replay.py"
PDD_BENCH="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------
# Комбинированный манифест (новый режим). Если задан — сплитится по sign_code
# в $OUT_BASE/_manifests/<slug>/real_manifest.jsonl и FULL_DIR указывает туда.
: "${MANIFEST:=}"
: "${FULL_DIR:=$SCRIPT_DIR/benchmark_output/full}"
: "${SCENES_ROOT:=$PDD_BENCH/scenes}"
: "${N_WORKERS:=32}"
: "${ROWS_LIMIT:=2000}"    # макс строк манифеста на знак
: "${MAX_STEPS:=600}"

# Наборы политик по пулам. CPU-политики не требуют чекпоинтов.
# Полный набор эвала: idm comprehensive_rule_expert rule_compliant ppo_lidar
#                     carl carl_rule plant2 plant2_rule
: "${POLICIES_CPU:=comprehensive_rule_expert rule_compliant}"
: "${POLICIES_CARL:=carl_rule}"      # пул CARL_CKPT (можно "carl_rule carl")
: "${POLICIES_PLANT2:=plant2_rule}"  # пул PLANT2_CKPT (можно "plant2_rule plant2")

# GPU-параллелизм для carl/plant2 пулов: round-robin по GPU_IDS,
# JOBS_PER_GPU процессов на каждый GPU.
: "${GPU_IDS:=0,1,2,3}"
: "${JOBS_PER_GPU:=3}"

# Раздельные пулы по моделям (carl и plant2). Если не заданы явно —
# GPU_IDS делится пополам (первая половина → carl, вторая → plant2).
: "${GPUS_CARL:=}"
: "${GPUS_PLANT2:=}"
: "${JOBS_PER_GPU_CARL:=$JOBS_PER_GPU}"
: "${JOBS_PER_GPU_PLANT2:=$JOBS_PER_GPU}"

# CUDA_DEVICE_ORDER=PCI_BUS_ID синхронизирует индексы CUDA с nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Чекпоинты (обязательны для соответствующих пулов). Девайс теперь
# выбирается автоматически внутри bench/policy_factory (cuda если доступна),
# принадлежность GPU задаётся через CUDA_VISIBLE_DEVICES per-task.
: "${CARL_CKPT:=}"
: "${PLANT2_CKPT:=}"
: "${PLANT2_ACTION_MODE:=pid}"   # pid | wps_pure_pursuit

: "${SKIP_CPU:=0}"
: "${SKIP_CARL:=0}"
: "${SKIP_PLANT2:=0}"
: "${SKIP_MERGE:=0}"

# Resume: пропустить эпизоды, у которых уже есть валидный replay.pkl + replay.json.
# Реализация — env var PDD_BENCH_RESUME=1 в expert_replay.py.
: "${RESUME:=0}"

# COVERAGE_CSV: путь к scene_coverage.csv. Если задан — манифест фильтруется
# на уровне scene_id ПЕРЕД запуском expert_replay.py (полностью покрытые
# scene_id выкидываются; частично покрытые оставляются под resume).
: "${COVERAGE_CSV:=}"

# MAX_VAR_PER_SID: ограничить число var_idx-строк на каждый scene_id
# ("" = без ограничения).
: "${MAX_VAR_PER_SID:=}"

# COMP_CHUNKS: разбить IDM-family subprocess на N параллельных chunks
# (--start/--count по одному filtered manifest; resume исключает дубли).
: "${COMP_CHUNKS:=1}"

TS="$(date +%Y%m%d_%H%M%S)"
# OUT_BASE можно подставить в существующий каталог чтобы дописать (resume).
: "${OUT_BASE:=$SCRIPT_DIR/benchmark_output/trajectories_$TS}"

: "${NODE_ID:=$(hostname)}"
: "${LOG_DIR:=$OUT_BASE/_logs/run_node${NODE_ID}_${TS}}"
PROGRESS_LOG="$LOG_DIR/progress.log"
MERGED_DIR="$OUT_BASE/_merged"

mkdir -p "$OUT_BASE" "$LOG_DIR" "$MERGED_DIR"

# Дублируем весь stdout/stderr скрипта в общий progress.log.
exec > >(tee -a "$PROGRESS_LOG") 2>&1
echo "=== node=$NODE_ID  ts=$TS  log_dir=$LOG_DIR ==="

if [ "$RESUME" = "1" ]; then
    export PDD_BENCH_RESUME=1
fi

# ---------------------------------------------------------------------------
# Новый режим: сплит комбинированного манифеста по sign_code → per-sign layout.
# ---------------------------------------------------------------------------
if [ -n "$MANIFEST" ]; then
    if [ ! -s "$MANIFEST" ]; then
        echo "[FAIL] MANIFEST=$MANIFEST не найден или пуст"
        exit 1
    fi
    SPLIT_DIR="$OUT_BASE/_manifests"
    echo "=== Split $MANIFEST → $SPLIT_DIR/<slug>/real_manifest.jsonl ==="
    "$PYTHON_BIN" - "$MANIFEST" "$SPLIT_DIR" <<'PY'
import json, pathlib, sys
manifest, out_root = sys.argv[1], pathlib.Path(sys.argv[2])
handles, counts = {}, {}
n_invalid = 0
for line in open(manifest):
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("valid") is False:
        n_invalid += 1
        continue
    code = row.get("sign_code") or row.get("pdd_code") or row.get("sign_type")
    if not code:
        continue
    slug = str(code).replace(".", "_")
    if slug not in handles:
        d = out_root / slug
        d.mkdir(parents=True, exist_ok=True)
        handles[slug] = open(d / "real_manifest.jsonl", "w")
        counts[slug] = 0
    handles[slug].write(line + "\n")
    counts[slug] += 1
for h in handles.values():
    h.close()
for slug in sorted(counts):
    print(f"  {slug}: {counts[slug]} rows")
if n_invalid:
    print(f"  (пропущено valid:false строк: {n_invalid})")
PY
    FULL_DIR="$SPLIT_DIR"
fi

# Parse GPU_IDS into array
IFS=',' read -ra _GPU_LIST <<< "$GPU_IDS"
_NUM_GPUS=${#_GPU_LIST[@]}

# Auto-split GPU_IDS между carl и plant2, если GPUS_CARL/GPUS_PLANT2 не заданы.
if [ -z "$GPUS_CARL" ] && [ -z "$GPUS_PLANT2" ]; then
    if [ "$_NUM_GPUS" -le 1 ]; then
        GPUS_CARL="$GPU_IDS"
        GPUS_PLANT2="$GPU_IDS"
    else
        _half=$((_NUM_GPUS / 2))
        [ "$_half" -lt 1 ] && _half=1
        GPUS_CARL=$(IFS=, ; echo "${_GPU_LIST[*]:0:$_half}")
        GPUS_PLANT2=$(IFS=, ; echo "${_GPU_LIST[*]:$_half}")
    fi
elif [ -z "$GPUS_CARL" ]; then
    GPUS_CARL="$GPU_IDS"
elif [ -z "$GPUS_PLANT2" ]; then
    GPUS_PLANT2="$GPU_IDS"
fi

IFS=',' read -ra _GPU_CARL_LIST   <<< "$GPUS_CARL"
IFS=',' read -ra _GPU_PLANT2_LIST <<< "$GPUS_PLANT2"
GPU_SLOTS_CARL=$(( ${#_GPU_CARL_LIST[@]} * JOBS_PER_GPU_CARL ))
GPU_SLOTS_PLANT2=$(( ${#_GPU_PLANT2_LIST[@]} * JOBS_PER_GPU_PLANT2 ))

echo "================================================================"
echo "collect_trajectories.sh  [$TS]"
echo "  NODE_ID              = $NODE_ID"
echo "  LOG_DIR              = $LOG_DIR"
echo "  MANIFEST             = ${MANIFEST:-<not set — per-sign FULL_DIR mode>}"
echo "  FULL_DIR             = $FULL_DIR"
echo "  SCENES_ROOT          = $SCENES_ROOT"
echo "  COVERAGE_CSV         = ${COVERAGE_CSV:-<not set>}"
echo "  MAX_VAR_PER_SID      = ${MAX_VAR_PER_SID:-<no cap>}"
echo "  COMP_CHUNKS          = ${COMP_CHUNKS} (1 = no chunking)"
echo "  PYTHON_BIN           = $PYTHON_BIN"
echo "  N_WORKERS            = $N_WORKERS  (CPU policies)"
echo "  POLICIES_CPU         = $POLICIES_CPU"
echo "  POLICIES_CARL        = $POLICIES_CARL"
echo "  POLICIES_PLANT2      = $POLICIES_PLANT2"
echo "  GPU_IDS              = $GPU_IDS"
echo "  GPUS_CARL            = $GPUS_CARL    (× $JOBS_PER_GPU_CARL jobs/GPU = $GPU_SLOTS_CARL слотов)"
echo "  GPUS_PLANT2          = $GPUS_PLANT2    (× $JOBS_PER_GPU_PLANT2 jobs/GPU = $GPU_SLOTS_PLANT2 слотов)"
echo "  ROWS_LIMIT           = $ROWS_LIMIT"
echo "  MAX_STEPS            = $MAX_STEPS"
echo "  CARL_CKPT            = ${CARL_CKPT:-<unset>}"
echo "  PLANT2_CKPT          = ${PLANT2_CKPT:-<unset>}"
echo "  PLANT2_ACTION_MODE   = $PLANT2_ACTION_MODE"
echo "  SKIP_CPU/CARL/PLANT2 = $SKIP_CPU/$SKIP_CARL/$SKIP_PLANT2"
echo "  RESUME               = $RESUME"
echo "  OUT_BASE             = $OUT_BASE"
echo "================================================================"
echo

# ---------------------------------------------------------------------------
# Источники манифестов:
#   SUMO real (priority 1)  — real_manifest.jsonl
#   PG-maps   (priority 2)  — pgmap_materialized.jsonl / synthetic_manifest.jsonl
#   CityMap   (priority 3)  — citymap_materialized.jsonl
#     (bench/env_builders гонит citymap через pgmap-билдер — фаза оставлена,
#      но в новом репо citymap-манифестов пока нет и она не проверена)
# ---------------------------------------------------------------------------
: "${SUMO_MANIFEST_NAME:=real_manifest.jsonl}"
: "${PGMAP_MANIFEST_NAME:=pgmap_materialized.jsonl}"
: "${PGMAP_FALLBACK_NAME:=synthetic_manifest.jsonl}"
: "${CITYMAP_MANIFEST_NAME:=citymap_materialized.jsonl}"

: "${SKIP_SUMO:=0}"
: "${SKIP_PGMAP:=0}"
: "${SKIP_CITYMAP:=0}"
: "${PGMAP_ROWS_LIMIT:=$ROWS_LIMIT}"
: "${CITYMAP_ROWS_LIMIT:=$ROWS_LIMIT}"

# while-read вместо mapfile — на macOS системный bash 3.2 без mapfile.
SIGNS_SUMO=()
while IFS= read -r _s; do
    [ -n "$_s" ] && SIGNS_SUMO+=("$_s")
done < <(
    find "$FULL_DIR" -maxdepth 2 -name "$SUMO_MANIFEST_NAME" 2>/dev/null \
        | sed "s|/$SUMO_MANIFEST_NAME||" | xargs -I{} basename {} | sort
)
SIGNS_PGMAP=()
while IFS= read -r _s; do
    [ -n "$_s" ] && SIGNS_PGMAP+=("$_s")
done < <(
    find "$FULL_DIR" -maxdepth 2 \( -name "$PGMAP_MANIFEST_NAME" -o -name "$PGMAP_FALLBACK_NAME" \) 2>/dev/null \
        | xargs -I{} dirname {} \
        | xargs -I{} basename {} \
        | sort -u
)
SIGNS_CITYMAP=()
while IFS= read -r _s; do
    [ -n "$_s" ] && SIGNS_CITYMAP+=("$_s")
done < <(
    find "$FULL_DIR" -maxdepth 2 -name "$CITYMAP_MANIFEST_NAME" 2>/dev/null \
        | xargs -I{} dirname {} | xargs -I{} basename {} | sort -u
)

echo "Знаков с SUMO-манифестами:    ${#SIGNS_SUMO[@]}  (${SIGNS_SUMO[*]:-})"
echo "Знаков с PG-map-манифестами:  ${#SIGNS_PGMAP[@]}  (${SIGNS_PGMAP[*]:-})"
echo "Знаков с CityMap-манифестами: ${#SIGNS_CITYMAP[@]}  (${SIGNS_CITYMAP[*]:-})"
echo

# SIGNS_FILTER — comma-separated whitelist знаков для этого сервера.
# Пример: SIGNS_FILTER=2_1,2_4,3_1 → только эти 3 знака.
: "${SIGNS_FILTER:=}"
if [ -n "$SIGNS_FILTER" ]; then
    IFS=',' read -ra _FILTER_LIST <<< "$SIGNS_FILTER"
    _filter_signs() {
        local arr_name="$1"
        eval "local src=( \${${arr_name}[@]+\"\${${arr_name}[@]}\"} )"
        local out=()
        for s in ${src[@]+"${src[@]}"}; do
            [ -z "$s" ] && continue
            for f in "${_FILTER_LIST[@]}"; do
                if [ "$s" = "$f" ]; then
                    out+=("$s")
                    break
                fi
            done
        done
        if [ "${#out[@]}" -eq 0 ]; then
            eval "${arr_name}=()"
        else
            eval "${arr_name}=( \"\${out[@]}\" )"
        fi
    }
    _filter_signs SIGNS_SUMO
    _filter_signs SIGNS_PGMAP
    _filter_signs SIGNS_CITYMAP
    echo "SIGNS_FILTER='$SIGNS_FILTER' → SUMO=${#SIGNS_SUMO[@]}: ${SIGNS_SUMO[*]:-}"
    echo "SIGNS_FILTER='$SIGNS_FILTER' → PGMAP=${#SIGNS_PGMAP[@]}: ${SIGNS_PGMAP[*]:-}"
    echo "SIGNS_FILTER='$SIGNS_FILTER' → CITYMAP=${#SIGNS_CITYMAP[@]}: ${SIGNS_CITYMAP[*]:-}"
    echo
fi

if [ "${#SIGNS_SUMO[@]}" -eq 0 ] && [ "${#SIGNS_PGMAP[@]}" -eq 0 ] && [ "${#SIGNS_CITYMAP[@]}" -eq 0 ]; then
    echo "[FAIL] Нет знаков ни с SUMO, ни с PG-map, ни с CityMap-манифестами в $FULL_DIR (после SIGNS_FILTER='$SIGNS_FILTER')"
    exit 1
fi

# Сэмпл IDM-вариантов для idm/comprehensive_rule_expert: default +
# EXTRA_SAMPLES_COMPREHENSIVE штук на сцену (сэмплируются per-scene внутри
# bench/policy_factory: seed = IDM_SEED_BASE + scene_seed + k*1000003).
: "${EXTRA_SAMPLES_COMPREHENSIVE:=4}"
: "${IDM_SEED_BASE:=42}"

# Политики с IDM-вариантами (s1..sN) и с COMP_CHUNKS-разбивкой.
_is_idm_variant_policy() {
    case "$1" in
        idm|comprehensive_rule_expert) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# CSV-фильтрация манифеста на уровне scene_id, по конкретной policy.
# Печатает (на stdout) путь к манифесту, который нужно использовать.
# ---------------------------------------------------------------------------
_filter_manifest_for_policy() {
    local manifest="$1" policy="$2" tmp_dir="$3"
    if [ -z "$COVERAGE_CSV" ] && [ -z "$MAX_VAR_PER_SID" ]; then
        echo "$manifest"
        return 0
    fi
    if [ -n "$COVERAGE_CSV" ] && [ ! -f "$COVERAGE_CSV" ]; then
        echo "$manifest"
        return 0
    fi
    mkdir -p "$tmp_dir"
    local tmp="$tmp_dir/_filt_${policy}_$(basename "$manifest")"
    "$PYTHON_BIN" - "$manifest" "${COVERAGE_CSV:-}" "$policy" "$tmp" "${MAX_VAR_PER_SID:-}" <<'PY' 2>/dev/null
import csv, json, sys
manifest, csv_path, policy, out_path, max_var = sys.argv[1:6]
max_var_int = int(max_var) if max_var.strip() else None
# Имена бейзлайнов = variant_full из expert_replay.run_batch_multi_variant.
POLICY_EXPERTS = {
    "comprehensive_rule_expert": (
        ["comprehensive_rule_expert_default"]
        + [f"comprehensive_rule_expert_s{k}" for k in range(1, 5)]),
    "idm": ["idm_default"] + [f"idm_s{k}" for k in range(1, 5)],
    "rule_compliant": ["rule_compliant"],
    "ppo_lidar":      ["ppo_lidar"],
    "carl_rule":      ["carl_rule"],
    "carl":           ["carl"],
    "plant2_rule":    ["plant2_rule"],
    "plant2":         ["plant2"],
}
experts = POLICY_EXPERTS.get(policy)
if not experts:
    import shutil
    shutil.copyfile(manifest, out_path)
    sys.exit(0)

# scene_id, для которых эта policy уже полностью покрыта.
done = set()
if csv_path:
    try:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if all(r.get(e, "0").strip() == "1" for e in experts):
                    done.add(r["scene_id"])
    except FileNotFoundError:
        pass

seen_per_sid = {}
n_in = n_out = n_skip_csv = n_skip_cap = 0
with open(out_path, "w") as fo, open(manifest) as fi:
    for line in fi:
        n_in += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("scene_id")
        if sid in done:
            n_skip_csv += 1
            continue
        if max_var_int is not None and sid is not None:
            cnt = seen_per_sid.get(sid, 0)
            if cnt >= max_var_int:
                n_skip_cap += 1
                continue
            seen_per_sid[sid] = cnt + 1
        fo.write(line)
        n_out += 1
sys.stderr.write(f"[csvfilter] {policy}: in={n_in} out={n_out} "
                  f"skipped_csv={n_skip_csv} skipped_cap={n_skip_cap}\n")
PY
    echo "$tmp"
}

# ---------------------------------------------------------------------------
# Поиск манифеста для (sign, phase). phase: "sumo" | "pgmap" | "citymap"
# ---------------------------------------------------------------------------
_pick_manifest_for() {
    local sign="$1" phase="$2"
    if [ "$phase" = "sumo" ]; then
        local f="$FULL_DIR/$sign/$SUMO_MANIFEST_NAME"
        [ -f "$f" ] && [ -s "$f" ] && { echo "$f"; return 0; }
    elif [ "$phase" = "pgmap" ]; then
        local f1="$FULL_DIR/$sign/$PGMAP_MANIFEST_NAME"
        local f2="$FULL_DIR/$sign/$PGMAP_FALLBACK_NAME"
        [ -f "$f1" ] && [ -s "$f1" ] && { echo "$f1"; return 0; }
        [ -f "$f2" ] && [ -s "$f2" ] && { echo "$f2"; return 0; }
    elif [ "$phase" = "citymap" ]; then
        local f="$FULL_DIR/$sign/$CITYMAP_MANIFEST_NAME"
        [ -f "$f" ] && [ -s "$f" ] && { echo "$f"; return 0; }
    fi
    echo ""
    return 1
}

# ---------------------------------------------------------------------------
# Одна задача: один знак + одна политика + одна фаза.
# extra_flags (для GPU-политик: --model-path и т.п.) — аргументы после фазы.
# ---------------------------------------------------------------------------
run_one() {
    local policy="$1"
    local sign="$2"
    local phase="${3:-sumo}"
    shift 3 || shift $#
    local extra_flags=("$@")

    local backend="$phase"
    local rows_limit_phase="$ROWS_LIMIT"
    [ "$phase" = "pgmap" ] && rows_limit_phase="$PGMAP_ROWS_LIMIT"
    [ "$phase" = "citymap" ] && rows_limit_phase="$CITYMAP_ROWS_LIMIT"

    local manifest_orig
    manifest_orig="$(_pick_manifest_for "$sign" "$phase")"
    if [ -z "$manifest_orig" ]; then
        echo "[skip] $policy/$sign/$phase — манифест не найден"
        return 0
    fi

    local out_dir="$OUT_BASE/$policy/$sign"
    local logf="$LOG_DIR/${policy}_${sign}_${phase}.log"
    mkdir -p "$out_dir"

    local manifest
    manifest="$(_filter_manifest_for_policy "$manifest_orig" "$policy" "$out_dir/_filt")"
    [ -z "$manifest" ] && manifest="$manifest_orig"
    if [ ! -s "$manifest" ]; then
        echo "[skip-csv] $policy/$sign/$phase — все scene_id уже покрыты этой policy"
        return 0
    fi

    local total
    total=$(wc -l < "$manifest")
    local rows=$(( total < rows_limit_phase ? total : rows_limit_phase ))

    # Доп.варианты IDM применяются только к idm/comprehensive_rule_expert
    # (для остальных policies expert_replay.py выставляет n_extra=0 сам).
    local idm_args=()
    if _is_idm_variant_policy "$policy"; then
        idm_args+=( --ego-extra-samples "$EXTRA_SAMPLES_COMPREHENSIVE"
                    --ego-sample-seed-base "$IDM_SEED_BASE" )
    fi

    # COMP_CHUNKS: для IDM-family — разбить на параллельные chunks по одному
    # filtered manifest (разные --start/--count; resume исключает дубли).
    local chunks="${COMP_CHUNKS:-1}"
    if _is_idm_variant_policy "$policy" && [ "$chunks" -gt 1 ] && [ "$rows" -gt "$chunks" ]; then
        local chunk_size=$(( (rows + chunks - 1) / chunks ))
        echo "[chunked] $policy/$sign/$phase  rows=$rows split into $chunks chunks × $chunk_size"
        local cpids=()
        for i in $(seq 0 $((chunks-1))); do
            local cstart=$((i * chunk_size))
            [ "$cstart" -ge "$rows" ] && break
            local clog="$LOG_DIR/${policy}_${sign}_${phase}_chunk${i}.log"
            "$PYTHON_BIN" "$RUNNER" \
                --manifest "$manifest" \
                --code "$sign" \
                --backend "$backend" \
                --policy "$policy" \
                --start "$cstart" \
                --count "$chunk_size" \
                --max-steps "$MAX_STEPS" \
                --scenes-root "$SCENES_ROOT" \
                --sample-ego-spawn-velocity \
                --output-dir "$out_dir" \
                ${idm_args[@]+"${idm_args[@]}"} \
                ${extra_flags[@]+"${extra_flags[@]}"} \
                > "$clog" 2>&1 &
            cpids+=("$!")
        done
        local cfail=0
        for pid in ${cpids[@]+"${cpids[@]}"}; do
            wait "$pid" || cfail=$((cfail+1))
        done
        if [ "$cfail" -eq 0 ]; then
            echo "[ok-chunked] $policy/$sign/$phase  (chunks=${#cpids[@]} rows=$rows)"
            return 0
        else
            echo "[FAIL-chunked] $policy/$sign/$phase  failures=$cfail/${#cpids[@]}  logs=$LOG_DIR"
            return 1
        fi
    fi

    "$PYTHON_BIN" "$RUNNER" \
        --manifest "$manifest" \
        --code "$sign" \
        --backend "$backend" \
        --policy "$policy" \
        --count "$rows" \
        --max-steps "$MAX_STEPS" \
        --scenes-root "$SCENES_ROOT" \
        --sample-ego-spawn-velocity \
        --output-dir "$out_dir" \
        ${idm_args[@]+"${idm_args[@]}"} \
        ${extra_flags[@]+"${extra_flags[@]}"} \
        > "$logf" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[ok]   $policy/$sign/$phase  (rows=$rows)"
    else
        echo "[FAIL] $policy/$sign/$phase  exit=$rc  log=$logf"
    fi
    return $rc
}
export -f run_one _pick_manifest_for _filter_manifest_for_policy _is_idm_variant_policy
export FULL_DIR PYTHON_BIN RUNNER ROWS_LIMIT PGMAP_ROWS_LIMIT CITYMAP_ROWS_LIMIT MAX_STEPS OUT_BASE LOG_DIR
export SUMO_MANIFEST_NAME PGMAP_MANIFEST_NAME PGMAP_FALLBACK_NAME CITYMAP_MANIFEST_NAME
export EXTRA_SAMPLES_COMPREHENSIVE IDM_SEED_BASE SCENES_ROOT
export COVERAGE_CSV MAX_VAR_PER_SID COMP_CHUNKS

# ---------------------------------------------------------------------------
# CPU-политики: пул с throttle по N_WORKERS.
# ---------------------------------------------------------------------------
fail=0

throttle() {
    while [ "$(jobs -rp | wc -l)" -ge "$N_WORKERS" ]; do
        sleep 0.5
    done
}

launch_cpu_pool() {
    # Запускается в subshell'е (& снаружи), поэтому jobs -rp здесь видит
    # ТОЛЬКО CPU-задачи, не GPU.
    local phase="$1"
    local signs_var_name="$2"
    eval "local signs_arr=( \"\${${signs_var_name}[@]:-}\" )"

    if [ "$SKIP_CPU" = "1" ] || [ -z "$POLICIES_CPU" ]; then
        echo "--- CPU pool [$phase]: SKIP (SKIP_CPU=$SKIP_CPU, POLICIES_CPU='$POLICIES_CPU') ---"
        return 0
    fi
    if [ "${#signs_arr[@]}" -eq 0 ] || [ -z "${signs_arr[0]:-}" ]; then
        echo "--- CPU pool [$phase]: SKIP (нет знаков с манифестом) ---"
        return 0
    fi

    echo "--- CPU pool [$phase]: знаков=${#signs_arr[@]}, политики='$POLICIES_CPU', N_WORKERS=$N_WORKERS ---"

    local pids=()
    local local_fail=0
    for policy in $POLICIES_CPU; do
        echo "    политика: $policy"
        for sign in "${signs_arr[@]}"; do
            throttle
            run_one "$policy" "$sign" "$phase" &
            pids+=("$!")
        done
    done

    for pid in ${pids[@]+"${pids[@]}"}; do
        wait "$pid" || local_fail=$((local_fail + 1))
    done
    return "$local_fail"
}

# ---------------------------------------------------------------------------
# GPU-политики: round-robin по своему списку GPU, свой JOBS_PER_GPU.
# Каждый пул (carl, plant2) идёт в отдельном subshell'е.
#
# Аргументы: $1=pool (carl|plant2)  $2=policies (space-sep)  $3=gpu_csv
#            $4=jobs_per_gpu  $5=phase  $6=signs_var_name
# ---------------------------------------------------------------------------

run_gpu_one() {
    local policy="$1" sign="$2" gpu="$3" phase="$4"
    shift 4
    CUDA_VISIBLE_DEVICES="$gpu" run_one "$policy" "$sign" "$phase" "$@"
}
export -f run_gpu_one

launch_gpu_pool() {
    local pool="$1"
    local policies="$2"
    local gpu_csv="$3"
    local jobs_per="$4"
    local phase="$5"
    local signs_var_name="$6"
    eval "local signs_arr=( \"\${${signs_var_name}[@]:-}\" )"

    # Гейтинг пула: skip-флаг + наличие чекпоинта.
    local ckpt=""
    local extra_flags=()
    if [ "$pool" = "carl" ]; then
        [ "$SKIP_CARL" = "1" ] && { echo "--- GPU pool [$phase/$pool]: SKIP (SKIP_CARL=1) ---"; return 0; }
        ckpt="$CARL_CKPT"
        extra_flags=( "--model-path" "$ckpt" )
    elif [ "$pool" = "plant2" ]; then
        [ "$SKIP_PLANT2" = "1" ] && { echo "--- GPU pool [$phase/$pool]: SKIP (SKIP_PLANT2=1) ---"; return 0; }
        ckpt="$PLANT2_CKPT"
        extra_flags=( "--model-path" "$ckpt" "--plant2-action-mode" "$PLANT2_ACTION_MODE" )
    fi
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo "--- GPU pool [$phase/$pool]: SKIP (нет чекпоинта: '${ckpt:-<unset>}') ---"
        return 0
    fi
    if [ -z "$policies" ]; then
        echo "--- GPU pool [$phase/$pool]: SKIP (пустой список политик) ---"
        return 0
    fi
    if [ "${#signs_arr[@]}" -eq 0 ] || [ -z "${signs_arr[0]:-}" ]; then
        echo "--- GPU pool [$phase/$pool]: SKIP (нет знаков с манифестом) ---"
        return 0
    fi
    if [ -z "$gpu_csv" ]; then
        echo "--- GPU pool [$phase/$pool]: SKIP (пустой список GPU) ---"
        return 0
    fi

    local gpu_list
    IFS=',' read -ra gpu_list <<< "$gpu_csv"
    local num_gpus=${#gpu_list[@]}
    local total_slots=$((num_gpus * jobs_per))
    local rr_idx=0

    echo "--- GPU pool [$phase/$pool]: знаков=${#signs_arr[@]}  политики='$policies'  GPUs=[$gpu_csv]  jobs/gpu=$jobs_per  slots=$total_slots ---"

    local pids=()
    local local_fail=0
    for policy in $policies; do
        for sign in "${signs_arr[@]}"; do
            while [ "$(jobs -rp | wc -l)" -ge "$total_slots" ]; do
                sleep 0.5
            done
            local gpu="${gpu_list[$((rr_idx % num_gpus))]}"
            rr_idx=$((rr_idx + 1))

            run_gpu_one "$policy" "$sign" "$gpu" "$phase" "${extra_flags[@]}" &
            pids+=("$!")
        done
    done

    for pid in ${pids[@]+"${pids[@]}"}; do
        wait "$pid" || local_fail=$((local_fail + 1))
    done
    return "$local_fail"
}

# ---------------------------------------------------------------------------
# MAIN FLOW: SUMO (priority 1) → wait → PGMAP (priority 2) → wait → CITYMAP
# ---------------------------------------------------------------------------

run_phase() {
    local phase="$1" signs_var_name="$2"
    launch_cpu_pool "$phase" "$signs_var_name" &
    local cpu_pid=$!
    launch_gpu_pool "carl"   "$POLICIES_CARL"   "$GPUS_CARL"   "$JOBS_PER_GPU_CARL"   "$phase" "$signs_var_name" &
    local carl_pid=$!
    launch_gpu_pool "plant2" "$POLICIES_PLANT2" "$GPUS_PLANT2" "$JOBS_PER_GPU_PLANT2" "$phase" "$signs_var_name" &
    local plant2_pid=$!

    echo "Phase [$phase]: cpu=$cpu_pid  carl=$carl_pid  plant2=$plant2_pid"
    echo "Логи: $LOG_DIR/"

    local cpu_fail carl_fail plant2_fail
    wait "$cpu_pid";    cpu_fail=$?
    wait "$carl_pid";   carl_fail=$?
    wait "$plant2_pid"; plant2_fail=$?
    echo "Phase [$phase] завершилась. cpu=$cpu_fail  carl=$carl_fail  plant2=$plant2_fail"
    return $((cpu_fail + carl_fail + plant2_fail))
}

if [ "$SKIP_SUMO" != "1" ] && [ "${#SIGNS_SUMO[@]}" -gt 0 ] && [ -n "${SIGNS_SUMO[0]:-}" ]; then
    echo
    echo "============================================================"
    echo "PHASE 1: SUMO (priority 1) — 3 lane параллельно (CPU + carl + plant2)"
    echo "============================================================"
    run_phase "sumo" "SIGNS_SUMO"
    fail=$((fail + $?))
else
    echo "[skip] Phase 1 (SUMO) пропущена (SKIP_SUMO=$SKIP_SUMO, signs=${#SIGNS_SUMO[@]})"
fi

if [ "$SKIP_PGMAP" != "1" ] && [ "${#SIGNS_PGMAP[@]}" -gt 0 ] && [ -n "${SIGNS_PGMAP[0]:-}" ]; then
    echo
    echo "============================================================"
    echo "PHASE 2: PG-maps (priority 2) — 3 lane параллельно (CPU + carl + plant2)"
    echo "============================================================"
    run_phase "pgmap" "SIGNS_PGMAP"
    fail=$((fail + $?))
else
    echo "[skip] Phase 2 (PGMAP) пропущена (SKIP_PGMAP=$SKIP_PGMAP, signs=${#SIGNS_PGMAP[@]})"
fi

if [ "$SKIP_CITYMAP" != "1" ] && [ "${#SIGNS_CITYMAP[@]}" -gt 0 ] && [ -n "${SIGNS_CITYMAP[0]:-}" ]; then
    echo
    echo "============================================================"
    echo "PHASE 3: CityMap (priority 3) — 3 lane параллельно (CPU + carl + plant2)"
    echo "============================================================"
    run_phase "citymap" "SIGNS_CITYMAP"
    fail=$((fail + $?))
else
    echo "[skip] Phase 3 (CITYMAP) пропущена (SKIP_CITYMAP=$SKIP_CITYMAP, signs=${#SIGNS_CITYMAP[@]})"
fi

echo
echo "=== Все фазы завершились. Total failures: $fail ==="

# ---------------------------------------------------------------------------
# Merge all_runs.jsonl + oracle
# ---------------------------------------------------------------------------
if [ "$SKIP_MERGE" = "1" ]; then
    echo "[skip] SKIP_MERGE=1"
    exit "$fail"
fi

echo
echo "=== Merge → $MERGED_DIR/all_runs.jsonl ==="
: > "$MERGED_DIR/all_runs.jsonl"
find "$OUT_BASE" -name "all_runs.jsonl" ! -path "$MERGED_DIR/*" | sort | while read -r f; do
    lines=$(wc -l < "$f")
    cat "$f" >> "$MERGED_DIR/all_runs.jsonl"
    echo "  + $f  ($lines рядов)"
done
total_lines=$(wc -l < "$MERGED_DIR/all_runs.jsonl")
echo "Итого: $total_lines рядов"

echo
echo "=== Build oracle_manifest ==="
"$PYTHON_BIN" "$RUNNER" \
    --build-oracle-manifest --run-dir "$MERGED_DIR" \
    2>&1 | tee "$LOG_DIR/_oracle.log"

# ---------------------------------------------------------------------------
# Metrics CSV + oracle_rule baseline поверх записанных сцен.
# Консолидируем сайдкары по variant_full (имя листовой папки =
# comprehensive_rule_expert_default/_s1/... /carl_rule/...) в layout
# var_0/<baseline>_replays.jsonl, который понимает
# build_episode_metrics_csv.py --runs-root; затем добавляем oracle_rule
# бейзлайн (лучший комплаентный rule-эксперт на каждую сцену).
# ---------------------------------------------------------------------------
: "${SKIP_ORACLE_CSV:=0}"
if [ "$SKIP_ORACLE_CSV" != "1" ]; then
    echo
    echo "=== Consolidate sidecars by variant → $MERGED_DIR/var_0 ==="
    "$PYTHON_BIN" - "$OUT_BASE" "$MERGED_DIR" <<'PY' 2>&1 | tee "$LOG_DIR/_consolidate_variants.log"
import json, pathlib, sys
out_base, merged = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
var0 = merged / "var_0"
var0.mkdir(parents=True, exist_ok=True)
by_baseline = {}
for sidecar in out_base.glob("*/*/by_sign/*/by_scene/*/*/replay.json"):
    parts = sidecar.parts
    if any(p in ("_merged", "_logs", "_manifests") for p in parts):
        continue
    baseline = sidecar.parent.name  # variant_full: leaf dir written by expert_replay
    try:
        replay = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] {sidecar}: {e}", file=sys.stderr)
        continue
    by_baseline.setdefault(baseline, []).append(replay)
for baseline in sorted(by_baseline):
    fp = var0 / f"{baseline}_replays.jsonl"
    with open(fp, "w", encoding="utf-8") as fh:
        for r in by_baseline[baseline]:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"  {baseline}: {len(by_baseline[baseline])} scenes -> {fp.name}")
if not by_baseline:
    print("  (no sidecars found)")
PY

    echo
    echo "=== metrics_per_episode.csv + oracle_rule baseline ==="
    "$PYTHON_BIN" "$SCRIPT_DIR/build_episode_metrics_csv.py" \
        --runs-root "$MERGED_DIR" \
        --out "$MERGED_DIR/metrics_per_episode.csv" \
        2>&1 | tee "$LOG_DIR/_metrics_csv.log"
    "$PYTHON_BIN" "$SCRIPT_DIR/build_oracle_baseline.py" \
        --csv "$MERGED_DIR/metrics_per_episode.csv" \
        2>&1 | tee "$LOG_DIR/_oracle_baseline.log"
else
    echo "[skip] SKIP_ORACLE_CSV=1"
fi

echo
echo "=== Сводка ==="
"$PYTHON_BIN" - "$MERGED_DIR/all_runs.jsonl" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
stats = collections.defaultdict(lambda: collections.Counter())
speeds = collections.defaultdict(list)
for r in rows:
    if not r.get("valid"):
        continue
    sign = r.get("sign_code") or r.get("sign_slug") or "?"
    pol  = r.get("policy", "?")
    # Нарушения — event-счётчик (новые ряды: violations_event_count;
    # legacy: total_violations и был event-счётчиком).
    viol = r.get("violations_event_count", r.get("total_violations", 0)) or 0
    ok = (r.get("arrived_dest") and not r.get("crashed")
          and not r.get("out_of_road") and not viol)
    stats[(sign, pol)]["ok" if ok else "fail"] += 1
    v = r.get("initial_speed_mps")
    if v is not None:
        speeds[(sign, pol)].append(float(v))
print(f"\n{'sign':<12}{'policy':<28}{'ok':>5}{'fail':>6}{'rate':>8}{'avg_v':>10}")
print("-" * 70)
for (sign, pol), c in sorted(stats.items()):
    tot = c["ok"] + c["fail"]
    rate = c["ok"] / max(1, tot)
    avg_v = sum(speeds[(sign,pol)]) / len(speeds[(sign,pol)]) if speeds[(sign,pol)] else 0.0
    print(f"{sign:<12}{pol:<28}{c['ok']:>5}{c['fail']:>6}{rate:>8.1%}{avg_v:>9.2f}m/s")
PY

echo
echo "================================================================"
echo "Готово."
echo "  Output:  $OUT_BASE"
echo "  Merged:  $MERGED_DIR/all_runs.jsonl  ($total_lines рядов)"
echo "  Oracle:  $MERGED_DIR/oracle_manifest.jsonl"
echo "  CSV:     $MERGED_DIR/metrics_per_episode.csv  (+oracle_rule baseline)"
echo "  Logs:    $LOG_DIR/"
echo "================================================================"

exit "$fail"
