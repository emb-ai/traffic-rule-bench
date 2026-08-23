#!/usr/bin/env bash
# collect.sh — trajectory collector (2.1 / 2.3 / 2.4 / 2.5 / 4.3)
#
# Profile-driven collector. Scenes go through eval run with aux agents.
#
#   SIGN=yield|main|stop|secondary|roundabout
#   (aliases: 2.4|2_4|2.1|2_1|main_road|2.5|2_5|stop_sign|2.3|2_3|secondary_road|4.3|4_3)
#
# Colleague-equivalent (yield):
#   SIGN=yield \
#   PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles EGO_CURVE_AWARE=1 \
#   EGO_HOLD_V0=1 CARL_LONGITUDINAL=tracking \
#   MANIFEST=data/runs/yield/train/real_manifest.jsonl \
#   POLICIES_CPU="comprehensive_rule_expert rule_compliant" \
#   POLICIES_CARL="carl_rule" POLICIES_PLANT2="plant2_rule" \
#   CARL_CKPT=../../../../checkpoints/carl/nuplan_51479_1B/model_best.pth \
#   PLANT2_CKPT=../../../../checkpoints/plant2_pretrain/epoch=029_final_3.ckpt \
#   PLANT2_ACTION_MODE=pid \
#   GPU_IDS=0,1,2,3 GPUS_CARL=0,1 GPUS_PLANT2=2,3 \
#   JOBS_PER_GPU=2 N_WORKERS=16 IDM_CHUNKS=8 \
#   EXTRA_SAMPLES_COMPREHENSIVE=4 IDM_SEED_BASE=42 \
#   MAX_STEPS=1500 RESUME=1 \
#   OUT_BASE=/path/to/traj_yield_2_4 \
#   bash collect.sh
#
# Live progress every PROGRESS_EVERY_S seconds (default 30).
# Smoke / visual QA (3 scenes + GIFs):
#   SIGN=yield SMOKE=1 MANIFEST=data/runs/yield/debug \
#     bash collect.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${SCRIPT_DIR}/run.py"

# ---------------------------------------------------------------------------
# Sign profile → data paths + PDD slug
# ---------------------------------------------------------------------------
: "${SIGN:=yield}"
case "$SIGN" in
    yield|2.4|2_4)
        SIGN=yield
        SIGN_SLUG=2_4
        PDD_CODE=2.4
        DATA_SUBDIR=yield
        ;;
    main|main_road|2.1|2_1)
        SIGN=main
        SIGN_SLUG=2_1
        PDD_CODE=2.1
        DATA_SUBDIR=main_road
        ;;
    stop|stop_sign|2.5|2_5)
        SIGN=stop
        SIGN_SLUG=2_5
        PDD_CODE=2.5
        DATA_SUBDIR=stop
        ;;
    secondary|secondary_road|2.3|2_3|2.3.1|2.3.2|2.3.3)
        SIGN=secondary
        SIGN_SLUG=2_3
        PDD_CODE=2.3
        DATA_SUBDIR=secondary_road
        ;;
    roundabout|4.3|4_3)
        SIGN=roundabout
        SIGN_SLUG=4_3
        PDD_CODE=4.3
        DATA_SUBDIR=roundabout
        ;;
    *)
        echo "[FAIL] unknown SIGN='$SIGN' (expected yield|main|stop|secondary|roundabout or 2.4|2.1|2.5|2.3|4.3)"
        exit 1
        ;;
esac

DATA_SCENES="$REPO_ROOT/data/scenes/$DATA_SUBDIR"
DATA_RUNS="$REPO_ROOT/data/runs/$DATA_SUBDIR"
DATA_TRAJ="$REPO_ROOT/data/trajectories/$DATA_SUBDIR"
: "${SCENES_ROOT:=$DATA_SCENES}"
: "${SIGNS_FILTER:=$SIGN_SLUG}"

# ---------------------------------------------------------------------------
# Policy / ego behaviour knobs (same env vars as the general collector).
# ---------------------------------------------------------------------------
: "${PER_SIGN_COMPLIANT_NPC:=1}"
: "${EGO_SAMPLER:=styles}"
: "${EGO_CURVE_AWARE:=1}"
: "${EGO_HOLD_V0:=1}"
: "${CARL_LONGITUDINAL:=tracking}"
export PER_SIGN_COMPLIANT_NPC EGO_SAMPLER EGO_CURVE_AWARE EGO_HOLD_V0 CARL_LONGITUDINAL

: "${MANIFEST:=}"

: "${N_WORKERS:=8}"
: "${IDM_CHUNKS:=8}"   # shard IDM-family policies across this many processes
: "${PROGRESS_EVERY_S:=30}"
: "${MAX_STEPS:=1500}"
: "${ROWS_LIMIT:=}"          # empty = all rows (or COUNT)
: "${EXTRA_SAMPLES_COMPREHENSIVE:=4}"  # default + s1..s4
: "${IDM_SEED_BASE:=42}"

: "${POLICIES_CPU:=comprehensive_rule_expert rule_compliant}"
: "${POLICIES_CARL:=carl_rule}"
: "${POLICIES_PLANT2:=plant2_rule}"

: "${GPU_IDS:=0}"
: "${GPUS_CARL:=}"
: "${GPUS_PLANT2:=}"
: "${JOBS_PER_GPU:=1}"
: "${NN_CHUNKS:=1}"

# Default checkpoints under the repo root.
: "${CARL_CKPT:=$REPO_ROOT/checkpoints/carl/nuplan_51479_1B/model_best.pth}"
: "${PLANT2_CKPT:=$REPO_ROOT/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt}"
: "${PLANT2_ACTION_MODE:=pid}"

_resolve_path() {
    # Absolute paths unchanged; relative paths resolved from SCRIPT_DIR.
    local p="$1"
    if [ -z "$p" ]; then
        echo ""
        return
    fi
    if [[ "$p" = /* ]]; then
        echo "$p"
        return
    fi
    (cd -- "$SCRIPT_DIR" && realpath -m -- "$p")
}

CARL_CKPT="$(_resolve_path "$CARL_CKPT")"
PLANT2_CKPT="$(_resolve_path "$PLANT2_CKPT")"

if [ -n "$CARL_CKPT" ] && [ ! -f "$CARL_CKPT" ]; then
    echo "[warn] CARL_CKPT not found: $CARL_CKPT — disabling CARL pool"
    CARL_CKPT=""
fi
if [ -n "$PLANT2_CKPT" ] && [ ! -f "$PLANT2_CKPT" ]; then
    # URL-encoded filename fallback (epoch%3D029_...)
    _alt="$REPO_ROOT/checkpoints/plant2_pretrain/epoch%3D029_final_3.ckpt"
    if [ -f "$_alt" ]; then
        PLANT2_CKPT="$_alt"
    else
        echo "[warn] PLANT2_CKPT not found: $PLANT2_CKPT — disabling PLANT2 pool"
        PLANT2_CKPT=""
    fi
fi

: "${SKIP_CPU:=0}"
if [ -z "${SKIP_CARL+x}" ]; then
    if [ -n "$CARL_CKPT" ]; then SKIP_CARL=0; else SKIP_CARL=1; fi
fi
: "${SKIP_CARL:=1}"
if [ -z "${SKIP_PLANT2+x}" ]; then
    if [ -n "$PLANT2_CKPT" ]; then SKIP_PLANT2=0; else SKIP_PLANT2=1; fi
fi
: "${SKIP_PLANT2:=1}"

: "${SKIP_MERGE:=0}"
: "${RESUME:=0}"

: "${SMOKE:=0}"
: "${SAVE_GIFS:=0}"
: "${COUNT:=}"

TS="$(date +%Y%m%d_%H%M%S)"
: "${NODE_ID:=$(hostname -s 2>/dev/null || echo local)}"
# Per-sign storage: data/trajectories/<sign>/...
: "${OUT_BASE:=$DATA_TRAJ/trajectories_$TS}"
: "${LOG_DIR:=$OUT_BASE/_logs/run_node${NODE_ID}_${TS}}"
MERGED_DIR="$OUT_BASE/_merged"
MANIFESTS_DIR="$OUT_BASE/_manifests/$SIGN_SLUG"

# Auto-split GPUs between carl / plant2 if not set.
IFS=',' read -ra _GPU_LIST <<< "$GPU_IDS"
_NUM_GPUS=${#_GPU_LIST[@]}
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

if [ "$SMOKE" = "1" ]; then
    : "${COUNT:=3}"
    SAVE_GIFS=1
    SKIP_CARL=1
    SKIP_PLANT2=1
    if [ -z "${SMOKE_POLICIES:-}" ]; then
        POLICIES_CPU="comprehensive_rule_expert"
    else
        POLICIES_CPU="$SMOKE_POLICIES"
    fi
    : "${SMOKE_EXTRA_SAMPLES:=4}"
    EXTRA_SAMPLES_COMPREHENSIVE="$SMOKE_EXTRA_SAMPLES"
    echo "=== SMOKE mode: SIGN=$SIGN COUNT=$COUNT SAVE_GIFS=1 policies='$POLICIES_CPU' EXTRA_SAMPLES=$EXTRA_SAMPLES_COMPREHENSIVE ==="
fi

mkdir -p "$OUT_BASE" "$LOG_DIR" "$MERGED_DIR" "$MANIFESTS_DIR"
exec > >(tee -a "$LOG_DIR/progress.log") 2>&1

if [ -z "$MANIFEST" ]; then
    echo "[FAIL] MANIFEST= is required (example: MANIFEST=data/runs/${DATA_SUBDIR}/train/real_manifest.jsonl)"
    exit 1
fi
if [[ "$MANIFEST" != /* ]]; then
    if [ -e "$REPO_ROOT/$MANIFEST" ]; then
        MANIFEST="$REPO_ROOT/$MANIFEST"
    elif [ -e "$MANIFEST" ]; then
        MANIFEST="$(cd -- "$(dirname -- "$MANIFEST")" && pwd)/$(basename -- "$MANIFEST")"
    fi
fi
if [ -d "$MANIFEST" ]; then
    if [ -s "$MANIFEST/real_manifest.jsonl" ]; then
        MANIFEST="$MANIFEST/real_manifest.jsonl"
    elif [ -e "$MANIFEST/latest/real_manifest.jsonl" ]; then
        MANIFEST="$(cd -- "$MANIFEST/latest" && pwd)/real_manifest.jsonl"
        echo "[manifests] using debug/latest → $MANIFEST"
    else
        _last=$(ls -1d "$MANIFEST"/[0-9][0-9][0-9][0-9]-* 2>/dev/null | sort | tail -1 || true)
        if [ -n "$_last" ] && [ -s "$_last/real_manifest.jsonl" ]; then
            MANIFEST="$_last/real_manifest.jsonl"
            echo "[manifests] using latest timestamp → $MANIFEST"
        else
            echo "[FAIL] MANIFEST dir has no real_manifest.jsonl: $MANIFEST"
            exit 1
        fi
    fi
fi
if [ ! -s "$MANIFEST" ]; then
    echo "[FAIL] MANIFEST missing/empty: $MANIFEST"
    exit 1
fi

cp -f "$MANIFEST" "$MANIFESTS_DIR/real_manifest.jsonl"
echo "[manifests] $MANIFESTS_DIR/real_manifest.jsonl"

if [ -n "$SIGNS_FILTER" ] && [ "$SIGNS_FILTER" != "$SIGN_SLUG" ] && [ "$SIGNS_FILTER" != "$PDD_CODE" ]; then
    echo "[warn] SIGNS_FILTER='$SIGNS_FILTER' ignored — this run is $SIGN ($PDD_CODE / $SIGN_SLUG)"
fi

CATALOG="$OUT_BASE/catalog.jsonl"

echo "================================================================"
echo "oracle collect  SIGN=$SIGN ($PDD_CODE)  [$TS]"
echo "  MANIFEST        = $MANIFEST"
echo "  SCENES_ROOT     = $SCENES_ROOT"
echo "  OUT_BASE        = $OUT_BASE"
echo "  SIGNS_FILTER    = $SIGNS_FILTER → $SIGN_SLUG"
echo "  COUNT/ROWS_LIMIT= ${COUNT:-—} / ${ROWS_LIMIT:-—}"
echo "  SAVE_GIFS/SMOKE = $SAVE_GIFS / $SMOKE"
echo "  EGO_SAMPLER     = $EGO_SAMPLER  CURVE_AWARE=$EGO_CURVE_AWARE  HOLD_V0=$EGO_HOLD_V0"
echo "  CARL_LONGITUDINAL=$CARL_LONGITUDINAL  COMPLIANT_NPC=$PER_SIGN_COMPLIANT_NPC"
echo "  CPU             = $POLICIES_CPU  (SKIP_CPU=$SKIP_CPU, N_WORKERS=$N_WORKERS, IDM_CHUNKS=$IDM_CHUNKS)"
echo "  CARL            = $POLICIES_CARL (SKIP_CARL=$SKIP_CARL, GPUS=$GPUS_CARL)"
echo "  PLANT2          = $POLICIES_PLANT2 (SKIP_PLANT2=$SKIP_PLANT2, GPUS=$GPUS_PLANT2)"
echo "  EXTRA_SAMPLES   = $EXTRA_SAMPLES_COMPREHENSIVE  IDM_SEED_BASE=$IDM_SEED_BASE"
echo "  MAX_STEPS       = $MAX_STEPS  RESUME=$RESUME  PROGRESS_EVERY_S=$PROGRESS_EVERY_S"
echo "  CARL_CKPT       = ${CARL_CKPT:-<unset>}"
echo "  PLANT2_CKPT     = ${PLANT2_CKPT:-<unset>}"
echo "================================================================"

_is_idm_family() {
    case "$1" in
        idm|comprehensive_rule_expert) return 0 ;;
        *) return 1 ;;
    esac
}

_manifest_nrows() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
n = 0
for ln in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    ln = ln.strip()
    if not ln:
        continue
    try:
        row = json.loads(ln)
    except json.JSONDecodeError:
        continue
    if row.get("valid") is False:
        continue
    n += 1
print(n)
PY
}

N_ROWS="$(_manifest_nrows "$MANIFEST")"
N_USE="$N_ROWS"
if [ -n "$COUNT" ]; then
    N_USE="$COUNT"
fi
if [ -n "$ROWS_LIMIT" ] && [ "$ROWS_LIMIT" -lt "$N_USE" ]; then
    N_USE="$ROWS_LIMIT"
fi
N_BASE=0
N_POL=0
_add_pols() {
    local skip="$1"
    shift
    if [ "$skip" = "1" ]; then
        return
    fi
    local p
    for p in "$@"; do
        [ -z "$p" ] && continue
        N_POL=$((N_POL + 1))
        if _is_idm_family "$p"; then
            N_BASE=$((N_BASE + 1 + EXTRA_SAMPLES_COMPREHENSIVE))
        else
            N_BASE=$((N_BASE + 1))
        fi
    done
}
_add_pols "$SKIP_CPU" $POLICIES_CPU
_add_pols "$SKIP_CARL" $POLICIES_CARL
_add_pols "$SKIP_PLANT2" $POLICIES_PLANT2
N_TRAJ=$((N_USE * N_BASE))
echo "======== collect plan ========"
echo "  rows:         $N_ROWS (will run $N_USE)"
echo "  policies:     $N_POL names → $N_BASE variants (IDM-family × 1+EXTRA_SAMPLES)"
echo "  trajectories: $N_USE × $N_BASE = $N_TRAJ"
echo "  hint:         train manifest is data/runs/${DATA_SUBDIR}/train/real_manifest.jsonl"
echo "=============================="

run_one() {
    local policy="$1"
    shift
    local extra=("$@")
    local out_dir="$OUT_BASE/$policy/$SIGN_SLUG"
    mkdir -p "$out_dir"

    # Log name: policy.log or policy.w03.log for shards.
    local log_tag="$policy"
    local i=0
    while [ "$i" -lt "${#extra[@]}" ]; do
        if [ "${extra[$i]}" = "--worker-id" ] && [ $((i + 1)) -lt "${#extra[@]}" ]; then
            log_tag=$(printf '%s.w%02d' "$policy" "${extra[$((i + 1))]}")
            break
        fi
        i=$((i + 1))
    done
    local logf="$LOG_DIR/${log_tag}.log"

    local has_count=0
    for a in ${extra[@]+"${extra[@]}"}; do
        if [ "$a" = "--count" ]; then
            has_count=1
            break
        fi
    done

    local count_args=()
    if [ "$has_count" -eq 0 ]; then
        if [ -n "$COUNT" ]; then
            count_args+=( --count "$COUNT" )
        elif [ -n "$ROWS_LIMIT" ]; then
            count_args+=( --count "$ROWS_LIMIT" )
        fi
    fi

    local idm_args=()
    if _is_idm_family "$policy"; then
        idm_args+=( --ego-extra-samples "$EXTRA_SAMPLES_COMPREHENSIVE"
                    --ego-sample-seed-base "$IDM_SEED_BASE" )
    fi

    local gif_args=()
    [ "$SAVE_GIFS" = "1" ] && gif_args+=( --save-gifs )

    local resume_args=()
    [ "$RESUME" = "1" ] && resume_args+=( --resume )

    echo "[run] $policy → $out_dir  log=$log_tag"
    if "$PYTHON_BIN" "$RUNNER" \
        --sign "$SIGN" \
        --manifest "$MANIFEST" \
        --scenes-root "$SCENES_ROOT" \
        --policy "$policy" \
        --output-dir "$out_dir" \
        --max-steps "$MAX_STEPS" \
        ${count_args[@]+"${count_args[@]}"} \
        ${idm_args[@]+"${idm_args[@]}"} \
        ${gif_args[@]+"${gif_args[@]}"} \
        ${resume_args[@]+"${resume_args[@]}"} \
        ${extra[@]+"${extra[@]}"} \
        > "$logf" 2>&1
    then
        echo "[ok]  $log_tag"
        if [ ! -s "$CATALOG" ] && [ -s "$out_dir/catalog.jsonl" ]; then
            cp "$out_dir/catalog.jsonl" "$CATALOG"
            echo "[catalog] $CATALOG"
        fi
        return 0
    else
        echo "[FAIL] $log_tag  log=$logf"
        return 1
    fi
}

# Launch one CPU policy, optionally sharded across IDM_CHUNKS processes.
_run_cpu_policy() {
    local policy="$1"
    # Smoke / tiny COUNT: keep single process.
    if [ -n "$COUNT" ] || [ "${IDM_CHUNKS:-1}" -le 1 ]; then
        while [ "$(jobs -rp | wc -l)" -ge "$N_WORKERS" ]; do
            sleep 1
        done
        run_one "$policy" &
        pids+=("$!")
        return 0
    fi

    local n_rows
    n_rows=$(_manifest_nrows "$MANIFEST")
    if [ -z "$n_rows" ] || [ "$n_rows" -le 0 ]; then
        echo "[FAIL] empty manifest for sharding: $MANIFEST"
        fail=$((fail + 1))
        return 1
    fi
    local n_chunks="$IDM_CHUNKS"
    if [ "$n_chunks" -gt "$n_rows" ]; then
        n_chunks="$n_rows"
    fi
    local chunk_size=$(( (n_rows + n_chunks - 1) / n_chunks ))
    echo "[shard] $policy  rows=$n_rows  chunks=$n_chunks  chunk_size=$chunk_size"
    local i start count
    for ((i = 0; i < n_chunks; i++)); do
        start=$((i * chunk_size))
        if [ "$start" -ge "$n_rows" ]; then
            break
        fi
        count=$chunk_size
        if [ $((start + count)) -gt "$n_rows" ]; then
            count=$((n_rows - start))
        fi
        while [ "$(jobs -rp | wc -l)" -ge "$N_WORKERS" ]; do
            sleep 1
        done
        run_one "$policy" --start "$start" --count "$count" --worker-id "$i" &
        pids+=("$!")
    done
}

fail=0
pids=()

if [ "$SKIP_CPU" != "1" ]; then
    for policy in $POLICIES_CPU; do
        _run_cpu_policy "$policy"
    done
fi

_run_gpu_pool() {
    local policies="$1" gpus="$2" ckpt="$3"
    shift 3
    local extra=("$@")
    IFS=',' read -ra gpu_list <<< "$gpus"
    local gi=0
    local running=0
    local max_parallel=$(( ${#gpu_list[@]} * JOBS_PER_GPU ))
    [ "$max_parallel" -lt 1 ] && max_parallel=1
    for policy in $policies; do
        while [ "$running" -ge "$max_parallel" ]; do
            wait -n 2>/dev/null || true
            running=$(jobs -rp | wc -l)
        done
        local gpu="${gpu_list[$((gi % ${#gpu_list[@]}))]}"
        gi=$((gi + 1))
        CUDA_VISIBLE_DEVICES="$gpu" run_one "$policy" --model-path "$ckpt" "${extra[@]}" &
        pids+=("$!")
        running=$((running + 1))
    done
}

if [ "$SKIP_CARL" != "1" ]; then
    if [ -z "$CARL_CKPT" ]; then
        echo "[FAIL] CARL pool enabled but CARL_CKPT unset"
        fail=$((fail + 1))
    else
        _run_gpu_pool "$POLICIES_CARL" "$GPUS_CARL" "$CARL_CKPT"
    fi
fi

if [ "$SKIP_PLANT2" != "1" ]; then
    if [ -z "$PLANT2_CKPT" ]; then
        echo "[FAIL] PLANT2 pool enabled but PLANT2_CKPT unset"
        fail=$((fail + 1))
    else
        _run_gpu_pool "$POLICIES_PLANT2" "$GPUS_PLANT2" "$PLANT2_CKPT" \
            --plant2-action-mode "$PLANT2_ACTION_MODE"
    fi
fi

# Live dashboard while policies run in parallel (progress is otherwise only in
# $LOG_DIR/<policy>.log — easy to miss from the main terminal).
_print_collect_progress() {
    "$PYTHON_BIN" - "$OUT_BASE" "$MANIFEST" "$EXTRA_SAMPLES_COMPREHENSIVE" "$LOG_DIR" "$SIGN_SLUG" <<'PY'
import os, re, sys, json
from pathlib import Path

out_base = Path(sys.argv[1])
manifest = Path(sys.argv[2])
extra = int(sys.argv[3] or 0)
log_dir = Path(sys.argv[4])
sign_slug = sys.argv[5]

n_rows = 0
if manifest.is_file():
    with open(manifest, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except Exception:
                n_rows += 1
                continue
            if row.get("valid") is False:
                continue
            n_rows += 1

idm = {"idm", "comprehensive_rule_expert"}
print("----- progress -----", flush=True)
any_pol = False
for pol_dir in sorted(p for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("_")):
    any_pol = True
    pol = pol_dir.name
    n_var = (1 + max(0, extra)) if pol in idm else 1
    target = n_rows * n_var if n_rows else 0
    done = 0
    seen = set()
    ledgers = list(pol_dir.glob(f"{sign_slug}/all_runs.jsonl"))
    ledgers += list(pol_dir.glob(f"{sign_slug}/all_runs.w*.jsonl"))
    # fallback if layout differs
    if not ledgers:
        ledgers = list(pol_dir.glob("*/all_runs.jsonl")) + list(pol_dir.glob("*/all_runs.w*.jsonl"))
    for ar in ledgers:
        try:
            with open(ar, encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    try:
                        r = json.loads(ln)
                    except Exception:
                        continue
                    key = (r.get("scene_uid"), r.get("policy"), r.get("variant"))
                    if key in seen:
                        continue
                    seen.add(key)
                    done += 1
        except OSError:
            pass
    pct = (100.0 * done / target) if target else 0.0
    width = 24
    filled = int(width * done / target) if target else 0
    bar = "#" * filled + "-" * (width - filled)
    last = ""
    log_cands = sorted(log_dir.glob(f"{pol}.log")) + sorted(log_dir.glob(f"{pol}.w*.log"))
    for logf in reversed(log_cands):
        try:
            lines = logf.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in reversed(lines[-80:]):
                if re.match(r"^\[\d+/\d+\]", ln) or "it/s" in ln or "%|" in ln:
                    last = ln.strip()[:100]
                    break
        except OSError:
            pass
        if last:
            break
    print(f"  {pol:<28} [{bar}] {done:>5}/{target:<5} ({pct:5.1f}%)", flush=True)
    if last:
        print(f"    └ {last}", flush=True)
if not any_pol:
    print("  (no policy dirs yet)", flush=True)
print(
    f"  tip: tail -f {log_dir}/<policy>.log   |   "
    f"refresh every {os.environ.get('PROGRESS_EVERY_S', '30')}s",
    flush=True,
)
print("--------------------", flush=True)
PY
}

echo
echo "[progress] live dashboard every ${PROGRESS_EVERY_S}s  (per-policy detail: $LOG_DIR/<policy>.log)"
_print_collect_progress

while true; do
    alive=0
    for pid in ${pids[@]+"${pids[@]}"}; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=1
            break
        fi
    done
    [ "$alive" -eq 0 ] && break
    sleep "$PROGRESS_EVERY_S"
    _print_collect_progress
done

for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" || fail=$((fail + 1))
done
_print_collect_progress

echo "=== Collection finished. failures=$fail ==="

# Fold parallel shard ledgers into all_runs.jsonl (dedupe by scene/policy/variant).
echo "=== Merge worker shards → all_runs.jsonl ==="
"$PYTHON_BIN" - "$OUT_BASE" <<'PY'
import json
import sys
from pathlib import Path

out_base = Path(sys.argv[1])
n_merged = 0
for pol_dir in sorted(p for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("_")):
    for sign_dir in sorted(p for p in pol_dir.iterdir() if p.is_dir()):
        shards = sorted(sign_dir.glob("all_runs.w*.jsonl"))
        main = sign_dir / "all_runs.jsonl"
        if not shards:
            continue
        by_key = {}
        order = []
        for path in ([main] if main.is_file() else []) + shards:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as e:
                print(f"  [warn] {path}: {e}")
                continue
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                key = (row.get("scene_uid"), row.get("policy"), row.get("variant"))
                if key not in by_key:
                    order.append(key)
                by_key[key] = row
        with open(main, "w", encoding="utf-8") as fh:
            for key in order:
                fh.write(json.dumps(by_key[key], default=str) + "\n")
        print(f"  {pol_dir.name}/{sign_dir.name}: {len(by_key)} rows "
              f"(from {len(shards)} shards" + (f" + main" if main.is_file() else "") + ")")
        n_merged += 1
print(f"shard merge: {n_merged} sign dirs")
PY

if [ "$SKIP_MERGE" = "1" ]; then
    echo "[skip] SKIP_MERGE=1"
    exit "$fail"
fi

echo "=== Merge → $MERGED_DIR/all_runs.jsonl ==="
: > "$MERGED_DIR/all_runs.jsonl"
find "$OUT_BASE" -path "*/${SIGN_SLUG}/all_runs.jsonl" ! -path "$MERGED_DIR/*" | sort | while read -r f; do
    n=$(wc -l < "$f")
    cat "$f" >> "$MERGED_DIR/all_runs.jsonl"
    echo "  + $f ($n)"
done
total=$(wc -l < "$MERGED_DIR/all_runs.jsonl" || echo 0)
echo "Total merged rows: $total"

if [ -s "$CATALOG" ]; then
    cp -f "$CATALOG" "$MERGED_DIR/catalog.jsonl"
fi

: "${SKIP_CONSOLIDATE:=0}"
if [ "$SKIP_CONSOLIDATE" != "1" ]; then
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
    baseline = sidecar.parent.name
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
    print("  (no sidecars found under */*/by_sign/*/by_scene/*/*/replay.json)")
PY
fi

n_gif=$(find "$OUT_BASE" -path '*/gifs/*.gif' 2>/dev/null | wc -l || echo 0)
echo
echo "================================================================"
echo "Done."
echo "  SIGN       : $SIGN ($PDD_CODE / $SIGN_SLUG)"
echo "  OUT_BASE   : $OUT_BASE"
echo "  Manifests  : $MANIFESTS_DIR/real_manifest.jsonl"
echo "  Merged     : $MERGED_DIR/all_runs.jsonl  ($total rows)"
echo "  var_0      : $MERGED_DIR/var_0/"
echo "  Catalog    : $CATALOG"
echo "  GIFs       : $n_gif"
echo "  Logs       : $LOG_DIR/"
echo
echo "Next:"
echo "  python -m traffic_bench.oracle.select.coverage --root $OUT_BASE --catalog $CATALOG \\"
echo "      --signs $PDD_CODE --horizon $MAX_STEPS --out-dir $OUT_BASE/experts"
echo "  SIGN=$SIGN ../report/table.sh $OUT_BASE"
echo "================================================================"

exit "$fail"
