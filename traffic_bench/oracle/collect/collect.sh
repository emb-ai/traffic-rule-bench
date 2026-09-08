#!/usr/bin/env bash
# collect.sh — trajectory collector for every eval sign
#
#   SIGN=yield
#   SIGN=yield,stop,direction/right
#   SIGN=all
#
# Smoke / visual QA (3 scenes + GIFs):
#   SIGN=yield SMOKE=1 ./collect.sh
#
# Full collection (auto-resolves data/runs/<sign>/train/real_manifest.jsonl):
#   SIGN=yield PER_SIGN_COMPLIANT_NPC=1 EGO_SAMPLER=styles \
#   POLICIES_CPU="idm_rule ppo_rule" \
#   ./collect.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${SCRIPT_DIR}/run.py"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Sign profile(s) from the eval registry
# ---------------------------------------------------------------------------
: "${SIGN:=yield}"
: "${_COLLECT_INNER:=0}"
: "${SPLIT:=train}"

_list_sign_ids() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from traffic_bench.eval.sign_registry import (
    list_profiles,
    profiles_from_sign_value,
    resolve_sign_token,
)

raw = sys.argv[1].strip()
try:
    if raw.lower() == "all":
        profiles = list(list_profiles())
    else:
        multi = profiles_from_sign_value(raw)
        profiles = list(multi) if multi is not None else [resolve_sign_token(raw)]
except KeyError as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)
for profile in profiles:
    print(profile.id)
PY
}

_load_sign_profile() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from traffic_bench.eval.sign_registry import resolve_sign_token

profile = resolve_sign_token(sys.argv[1])
print(f"SIGN={profile.id}")
print(f"DATA_SUBDIR={profile.data_subdir}")
print(f"SIGN_CODE={profile.sign_code}")
print(f"SIGN_TYPE={profile.sign_type}")
PY
}

_data_subdir_for_sign() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from traffic_bench.eval.sign_registry import resolve_sign_token
print(resolve_sign_token(sys.argv[1]).data_subdir)
PY
}

# Exit 0 if stored collection is a continuation of current_manifest.
# Rule: every scene_uid in stored ⊆ current; optional by_scene evidence
# under out_base must also ⊆ current (guards a reused folder from another run).
_manifests_compatible() {
    local stored="$1" current="$2" out_base="${3:-}"
    "$PYTHON_BIN" - "$stored" "$current" "$out_base" <<'PY'
import json, sys
from pathlib import Path

def scene_uid(row: dict):
    if row.get("scene_uid"):
        return str(row["scene_uid"])
    sid = row.get("scene_id")
    if sid is None:
        return None
    # Match traffic_bench.oracle.collect.run._scene_uid
    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    return (
        f"{sid}_lane{int(row.get('spawn_lane_num', 0) or 0)}"
        f"_seed{seed}_v{int(row.get('var_idx', 0) or 0)}"
    )

def uids(path: Path) -> set:
    out = set()
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if row.get("valid") is False:
            continue
        uid = scene_uid(row)
        if uid:
            out.add(uid)
    return out

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

stored_p, curr_p, out_base = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
stored = uids(stored_p)
current = uids(curr_p)
if not current:
    log("[final] FAIL: current manifest has no valid rows")
    sys.exit(1)
if stored:
    extra = stored - current
    if extra:
        sample = ", ".join(sorted(extra)[:3])
        log(
            f"[final] FAIL: {len(extra)} UID(s) in stored manifest not in current "
            f"(e.g. {sample})"
        )
        sys.exit(1)

evidence = set()
if out_base:
    root = Path(out_base)
    disk = set()
    for p in root.glob("*/by_scene/*"):
        if p.is_dir():
            disk.add(p.name)
    stray = disk - current
    if stray:
        sample = ", ".join(sorted(stray)[:3])
        log(
            f"[final] FAIL: {len(stray)} by_scene UID(s) not in current manifest "
            f"(e.g. {sample})"
        )
        sys.exit(1)
    evidence = disk & current
    if not stored and disk and not evidence:
        log("[final] FAIL: on-disk scenes do not overlap current manifest")
        sys.exit(1)

overlap = stored & current if stored else evidence
log(
    f"[final] OK: stored={len(stored)} current={len(current)} "
    f"overlap={len(overlap)} on_disk={len(evidence)}"
)
sys.exit(0)
PY
}

# Resolve MANIFEST path for a sign (echo absolute path). Uses env SPLIT / SMOKE.
_resolve_manifest_for_sign() {
    local sid="$1" data_subdir="$2" user_manifest="${3:-}"
    local data_runs="$REPO_ROOT/data/runs/$data_subdir"
    local m="" split_use="${SPLIT:-train}"
    if [ "${SMOKE:-0}" = "1" ]; then
        split_use=debug
    fi
    if [ -n "$user_manifest" ]; then
        m="$user_manifest"
        m="${m//\{sign\}/$sid}"
        m="${m//\{id\}/$sid}"
        if [[ "$user_manifest" != *"{sign}"* ]] && [[ "$user_manifest" != *"{id}"* ]]; then
            # Shared path without placeholders — only valid for single-sign.
            m="$user_manifest"
        fi
    else
        if [ "$split_use" = "debug" ]; then
            m="$data_runs/debug"
        else
            m="$data_runs/$split_use/real_manifest.jsonl"
        fi
    fi
    if [[ "$m" != /* ]]; then
        if [ -e "$REPO_ROOT/$m" ]; then
            m="$REPO_ROOT/$m"
        elif [ -e "$m" ]; then
            m="$(cd -- "$(dirname -- "$m")" && pwd)/$(basename -- "$m")"
        fi
    fi
    if [ -d "$m" ]; then
        if [ -s "$m/real_manifest.jsonl" ]; then
            m="$m/real_manifest.jsonl"
        elif [ -e "$m/latest/real_manifest.jsonl" ]; then
            m="$(cd -- "$m/latest" && pwd)/real_manifest.jsonl"
        else
            local _last
            _last=$(ls -1d "$m"/[0-9][0-9][0-9][0-9]-* 2>/dev/null | sort | tail -1 || true)
            if [ -n "$_last" ] && [ -s "$_last/real_manifest.jsonl" ]; then
                m="$_last/real_manifest.jsonl"
            else
                echo "[FAIL] MANIFEST dir has no real_manifest.jsonl: $m" >&2
                return 1
            fi
        fi
    fi
    if [ ! -s "$m" ]; then
        echo "[FAIL] MANIFEST missing/empty: $m" >&2
        return 1
    fi
    echo "$m"
}

# Decide OUT_BASE for a sign. Echoes: OUT_BASE|RESUME(0|1)
# Prefer data/trajectories/<sign>/final when present and compatible.
_resolve_out_base_for_sign() {
    local sid="$1" data_subdir="$2" manifest="$3" user_out="${4:-}" ts="$5"
    local data_traj="$REPO_ROOT/data/trajectories/$data_subdir"
    local out resume=0

    if [ -n "$user_out" ]; then
        out="$user_out"
        if [[ "$out" != /* ]]; then
            out="$REPO_ROOT/$out"
        fi
        # Multi-sign caller may already append /<sid>; single shared root gets /sid.
        if [ -n "${_MULTI_SIGN:-}" ] && [[ "$out" != *"/$sid" ]] && [[ "$out" != *"/$sid/"* ]]; then
            # If user_out is the multi root, caller passes already-joined path.
            :
        fi
        echo "${out}|${resume}"
        return 0
    fi

    local final="$data_traj/final"
    if [ "${USE_FINAL:-1}" = "1" ] && [ -d "$final" ]; then
        local stored="$final/_manifests/real_manifest.jsonl"
        if [ -s "$stored" ]; then
            if ! _manifests_compatible "$stored" "$manifest" "$final"; then
                echo "[FAIL] SIGN=$sid refusing final/ — manifest mismatch" >&2
                echo "       final=$final" >&2
                echo "       current=$manifest" >&2
                return 1
            fi
            echo "[final] SIGN=$sid → $final (RESUME=1)" >&2
            echo "${final}|1"
            return 0
        fi
        # Non-empty final without _manifests: still require disk ⊆ current.
        if [ -n "$(ls -A "$final" 2>/dev/null || true)" ]; then
            if ! _manifests_compatible "/dev/null" "$manifest" "$final"; then
                echo "[FAIL] SIGN=$sid final/ has data but no _manifests/ and disk UIDs mismatch" >&2
                return 1
            fi
            echo "[final] SIGN=$sid using existing final/ without stored manifest (RESUME=1)" >&2
            echo "${final}|1"
            return 0
        fi
        echo "[final] SIGN=$sid empty final/ → fresh collection there" >&2
        echo "${final}|0"
        return 0
    fi

    echo "${data_traj}/trajectories_${ts}|0"
}

if [ "$_COLLECT_INNER" != "1" ]; then
    if ! _SIGN_LIST="$(_list_sign_ids "$SIGN")"; then
        echo "[FAIL] unknown SIGN='$SIGN'"
        exit 1
    fi
    mapfile -t _SIGN_IDS <<< "$_SIGN_LIST"
    if [ "${#_SIGN_IDS[@]}" -eq 0 ] || [ -z "${_SIGN_IDS[0]:-}" ]; then
        echo "[FAIL] unknown SIGN='$SIGN'"
        exit 1
    fi
    if [ "${#_SIGN_IDS[@]}" -gt 1 ]; then
        TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
        _user_out="${OUT_BASE-}"
        _user_manifest="${MANIFEST-}"
        : "${MULTI_SIGN_PARALLEL:=1}"

        # Legacy / safe mode: one sign at a time (full CPU+GPU+merge per sign).
        if [ "$MULTI_SIGN_PARALLEL" = "0" ]; then
            fail=0
            echo "=== multi-sign collect (SEQUENTIAL)  signs=${#_SIGN_IDS[@]}  ts=$TS ==="
            echo "  MULTI_SIGN_PARALLEL=0  (one sign fully finishes before the next)"
            for sid in "${_SIGN_IDS[@]}"; do
                echo
                echo "######## SIGN=$sid ########"
                if ! (
                    export _COLLECT_INNER=1 SIGN="$sid" TS="$TS"
                    if [ -n "${_user_out}" ]; then
                        export OUT_BASE="${_user_out}/${sid}"
                    else
                        unset OUT_BASE
                    fi
                    if [ -n "${_user_manifest}" ]; then
                        _m="$_user_manifest"
                        _m="${_m//\{sign\}/$sid}"
                        _m="${_m//\{id\}/$sid}"
                        if [[ "$_user_manifest" == *"{sign}"* ]] || [[ "$_user_manifest" == *"{id}"* ]]; then
                            export MANIFEST="$_m"
                        else
                            unset MANIFEST
                        fi
                    fi
                    bash "$0"
                ); then
                    fail=$((fail + 1))
                fi
            done
            echo "=== multi-sign done. failures=$fail ==="
            exit "$fail"
        fi

        # -----------------------------------------------------------------
        # Multi-sign orchestrator (MULTI_SIGN_PARALLEL=1, default):
        #   1) GPU policies for all signs in parallel (1 job ↔ 1 GPU slot)
        #   2) CPU policies one sign at a time (N_WORKERS still shards within)
        #   3) merge/consolidate per sign after its CPU phase
        # -----------------------------------------------------------------
        : "${GPU_IDS:=0}"
        : "${JOBS_PER_GPU:=1}"
        : "${SKIP_CPU:=0}"
        : "${SKIP_CARL:=}"
        : "${SKIP_PLANT2:=}"
        : "${SMOKE:=0}"

        # Smoke forces CPU-only (same as single-sign).
        if [ "$SMOKE" = "1" ]; then
            SKIP_CARL=1
            SKIP_PLANT2=1
        fi

        # Resolve default skip from ckpt presence when unset (mirrors inner).
        _carl_ckpt="${CARL_CKPT:-$REPO_ROOT/checkpoints/carl/nuplan_51479_1B/model_best.pth}"
        _plant2_ckpt="${PLANT2_CKPT:-$REPO_ROOT/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt}"
        if [ -z "${SKIP_CARL}" ]; then
            if [ -f "$_carl_ckpt" ]; then SKIP_CARL=0; else SKIP_CARL=1; fi
        fi
        if [ -z "${SKIP_PLANT2}" ]; then
            if [ -f "$_plant2_ckpt" ] || [ -f "$REPO_ROOT/checkpoints/plant2_pretrain/epoch%3D029_final_3.ckpt" ]; then
                SKIP_PLANT2=0
            else
                SKIP_PLANT2=1
            fi
        fi

        IFS=',' read -ra _GPU_LIST <<< "$GPU_IDS"
        _max_gpu=$(( ${#_GPU_LIST[@]} * JOBS_PER_GPU ))
        [ "$_max_gpu" -lt 1 ] && _max_gpu=1

        declare -a _SIGN_OUT=()
        declare -a _SIGN_MAN=()
        declare -a _SIGN_RESUME=()
        fail=0
        echo "=== multi-sign collect  signs=${#_SIGN_IDS[@]}  ts=$TS ==="
        echo "  GPU_IDS=$GPU_IDS  JOBS_PER_GPU=$JOBS_PER_GPU  max_parallel_gpu=$_max_gpu"
        echo "  SKIP_CPU=$SKIP_CPU SKIP_CARL=$SKIP_CARL SKIP_PLANT2=$SKIP_PLANT2"
        echo "  phase1=GPU(all signs parallel)  phase2=CPU(sequential per sign)"
        echo "  tip: MULTI_SIGN_PARALLEL=0 for one-sign-at-a-time"

        for sid in "${_SIGN_IDS[@]}"; do
            _sub="$(_data_subdir_for_sign "$sid")" || { fail=$((fail + 1)); continue; }
            if [ -n "${_user_out}" ]; then
                _out_arg="${_user_out}/${sid}"
            else
                _out_arg=""
            fi
            if [ -n "${_user_manifest}" ]; then
                if [[ "$_user_manifest" == *"{sign}"* ]] || [[ "$_user_manifest" == *"{id}"* ]]; then
                    _man_arg="$_user_manifest"
                else
                    # No placeholders — auto per-sign manifests.
                    _man_arg=""
                fi
            else
                _man_arg=""
            fi
            if ! _man="$(_resolve_manifest_for_sign "$sid" "$_sub" "$_man_arg")"; then
                fail=$((fail + 1))
                continue
            fi
            if ! _resolved="$(_resolve_out_base_for_sign "$sid" "$_sub" "$_man" "$_out_arg" "$TS")"; then
                fail=$((fail + 1))
                continue
            fi
            _out="${_resolved%%|*}"
            _res="${_resolved##*|}"
            _SIGN_OUT+=("$_out")
            _SIGN_MAN+=("$_man")
            _SIGN_RESUME+=("$_res")
            echo "  plan SIGN=$sid  OUT=$_out  RESUME=$_res  MANIFEST=$_man"
        done
        if [ "${#_SIGN_OUT[@]}" -ne "${#_SIGN_IDS[@]}" ]; then
            echo "[FAIL] multi-sign setup incomplete (failures=$fail)"
            exit 1
        fi

        # ---- Phase 1: GPU jobs across signs ----
        declare -a _GPU_JOBS=()  # sid_idx|family
        _idx=0
        for sid in "${_SIGN_IDS[@]}"; do
            if [ "$SKIP_CARL" != "1" ]; then
                _GPU_JOBS+=("${_idx}|carl")
            fi
            if [ "$SKIP_PLANT2" != "1" ]; then
                _GPU_JOBS+=("${_idx}|plant2")
            fi
            _idx=$((_idx + 1))
        done

        _gpu_fail=0
        if [ "${#_GPU_JOBS[@]}" -gt 0 ]; then
            echo
            echo "######## phase1 GPU: ${#_GPU_JOBS[@]} jobs on ${#_GPU_LIST[@]} GPU(s) ########"
            _gi=0
            _gpids=()
            for _job in "${_GPU_JOBS[@]}"; do
                while [ "$(jobs -rp | wc -l)" -ge "$_max_gpu" ]; do
                    wait -n 2>/dev/null || true
                done
                _jidx="${_job%%|*}"
                _fam="${_job##*|}"
                _sid="${_SIGN_IDS[$_jidx]}"
                _out="${_SIGN_OUT[$_jidx]}"
                _man="${_SIGN_MAN[$_jidx]}"
                _res="${_SIGN_RESUME[$_jidx]}"
                _gpu="${_GPU_LIST[$((_gi % ${#_GPU_LIST[@]}))]}"
                _gi=$((_gi + 1))
                echo "[gpu$_gpu] START SIGN=$_sid family=$_fam → $_out"
                (
                    export _COLLECT_INNER=1 SIGN="$_sid" TS="$TS"
                    export OUT_BASE="$_out" MANIFEST="$_man"
                    export RESUME="$_res"
                    export SKIP_CPU=1 SKIP_MERGE=1
                    export GPU_IDS="$_gpu"
                    if [ "$_fam" = "carl" ]; then
                        export SKIP_CARL=0 SKIP_PLANT2=1
                        export GPUS_CARL="$_gpu"
                        unset GPUS_PLANT2 || true
                    else
                        export SKIP_CARL=1 SKIP_PLANT2=0
                        export GPUS_PLANT2="$_gpu"
                        unset GPUS_CARL || true
                    fi
                    bash "$0"
                ) &
                _gpids+=("$!")
            done
            for _p in ${_gpids[@]+"${_gpids[@]}"}; do
                if ! wait "$_p"; then
                    _gpu_fail=$((_gpu_fail + 1))
                fi
            done
            echo "######## phase1 GPU done. failures=$_gpu_fail ########"
            fail=$((fail + _gpu_fail))
        else
            echo "######## phase1 GPU: skipped ########"
        fi

        # ---- Phase 2: CPU (+ merge) sequential per sign ----
        _idx=0
        for sid in "${_SIGN_IDS[@]}"; do
            _out="${_SIGN_OUT[$_idx]}"
            _man="${_SIGN_MAN[$_idx]}"
            _res="${_SIGN_RESUME[$_idx]}"
            echo
            echo "######## phase2 CPU SIGN=$sid → $_out ########"
            if [ "$SKIP_CPU" = "1" ]; then
                # Still merge GPU outputs for this sign.
                if ! (
                    export _COLLECT_INNER=1 SIGN="$sid" TS="$TS"
                    export OUT_BASE="$_out" MANIFEST="$_man" RESUME="$_res"
                    export SKIP_CPU=1 SKIP_CARL=1 SKIP_PLANT2=1 SKIP_MERGE=0
                    bash "$0"
                ); then
                    fail=$((fail + 1))
                fi
            else
                if ! (
                    export _COLLECT_INNER=1 SIGN="$sid" TS="$TS"
                    export OUT_BASE="$_out" MANIFEST="$_man" RESUME="$_res"
                    export SKIP_CARL=1 SKIP_PLANT2=1 SKIP_MERGE=0
                    # leave SKIP_CPU unset/0
                    unset SKIP_CPU || true
                    bash "$0"
                ); then
                    fail=$((fail + 1))
                fi
            fi
            _idx=$((_idx + 1))
        done

        echo "=== multi-sign done. failures=$fail ==="
        exit "$fail"
    fi
    SIGN="${_SIGN_IDS[0]}"
fi

if ! _PROFILE_ENV="$(_load_sign_profile "$SIGN")"; then
    echo "[FAIL] unknown SIGN='$SIGN'"
    exit 1
fi
eval "$_PROFILE_ENV"

DATA_SCENES="$REPO_ROOT/data/scenes/$DATA_SUBDIR"
DATA_RUNS="$REPO_ROOT/data/runs/$DATA_SUBDIR"
DATA_TRAJ="$REPO_ROOT/data/trajectories/$DATA_SUBDIR"
: "${SCENES_ROOT:=$DATA_SCENES}"
PDD_CODE="$SIGN_CODE"

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

: "${POLICIES_CPU:=idm_rule ppo_rule}"
: "${POLICIES_CARL:=carl_rule}"
: "${POLICIES_PLANT2:=plant2_rule}"

: "${GPU_IDS:=0}"
: "${GPUS_CARL:=}"
: "${GPUS_PLANT2:=}"
: "${JOBS_PER_GPU:=1}"
: "${NN_CHUNKS:=1}"
# 0 = finish CPU policies before starting carl/plant2 (avoids MetaDrive CPU
# contention when N_WORKERS/IDM_CHUNKS is high). 1 = old overlap behavior.
: "${OVERLAP_CPU_GPU:=0}"

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

: "${TS:=$(date +%Y%m%d_%H%M%S)}"
: "${NODE_ID:=$(hostname -s 2>/dev/null || echo local)}"

# Remember whether the caller fixed OUT_BASE (multi-sign / resume into final).
_OUT_BASE_FROM_USER=0
if [ -n "${OUT_BASE+x}" ] && [ -n "${OUT_BASE}" ]; then
    _OUT_BASE_FROM_USER=1
    if [[ "$OUT_BASE" != /* ]]; then
        OUT_BASE="$REPO_ROOT/$OUT_BASE"
    fi
fi

# Auto-split GPUs between carl / plant2 if not set.
: "${GPU_IDS:=0}"
IFS=',' read -ra _GPU_LIST <<< "$GPU_IDS"
_NUM_GPUS=${#_GPU_LIST[@]}
if [ -z "${GPUS_CARL:-}" ] && [ -z "${GPUS_PLANT2:-}" ]; then
    if [ "$_NUM_GPUS" -le 1 ]; then
        GPUS_CARL="$GPU_IDS"
        GPUS_PLANT2="$GPU_IDS"
    else
        _half=$((_NUM_GPUS / 2))
        [ "$_half" -lt 1 ] && _half=1
        GPUS_CARL=$(IFS=, ; echo "${_GPU_LIST[*]:0:$_half}")
        GPUS_PLANT2=$(IFS=, ; echo "${_GPU_LIST[*]:$_half}")
    fi
elif [ -z "${GPUS_CARL:-}" ]; then
    GPUS_CARL="$GPU_IDS"
elif [ -z "${GPUS_PLANT2:-}" ]; then
    GPUS_PLANT2="$GPU_IDS"
fi

if [ "$SMOKE" = "1" ]; then
    : "${COUNT:=3}"
    SAVE_GIFS=1
    SKIP_CARL=1
    SKIP_PLANT2=1
    if [ -z "${SMOKE_POLICIES:-}" ]; then
        POLICIES_CPU="idm_rule"
    else
        POLICIES_CPU="$SMOKE_POLICIES"
    fi
    : "${SMOKE_EXTRA_SAMPLES:=4}"
    EXTRA_SAMPLES_COMPREHENSIVE="$SMOKE_EXTRA_SAMPLES"
    echo "=== SMOKE mode: SIGN=$SIGN COUNT=$COUNT SAVE_GIFS=1 policies='$POLICIES_CPU' EXTRA_SAMPLES=$EXTRA_SAMPLES_COMPREHENSIVE ==="
    SPLIT=debug
fi

# Legacy spellings in POLICIES_CPU / SMOKE_POLICIES → canonical ids
# (comprehensive_rule_expert → idm_rule, rule_compliant → ppo_rule; see
# agents/policy_names.py). Applied after SMOKE so SMOKE_POLICIES is covered too.
POLICIES_CPU="${POLICIES_CPU//comprehensive_rule_expert/idm_rule}"
POLICIES_CPU="${POLICIES_CPU//rule_compliant/ppo_rule}"

# Resolve MANIFEST before OUT_BASE so final/ can be checked against it.
if [ -z "$MANIFEST" ]; then
    if [ "$SPLIT" = "debug" ]; then
        MANIFEST="$DATA_RUNS/debug"
    else
        MANIFEST="$DATA_RUNS/$SPLIT/real_manifest.jsonl"
    fi
    echo "[manifests] auto SIGN=$SIGN → $MANIFEST"
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

# Per-sign storage: prefer data/trajectories/<sign>/final when compatible.
if [ "$_OUT_BASE_FROM_USER" = "0" ]; then
    if ! _resolved="$(_resolve_out_base_for_sign "$SIGN" "$DATA_SUBDIR" "$MANIFEST" "" "$TS")"; then
        exit 1
    fi
    OUT_BASE="${_resolved%%|*}"
    _final_resume="${_resolved##*|}"
    if [ "$_final_resume" = "1" ]; then
        RESUME=1
    fi
fi
: "${LOG_DIR:=$OUT_BASE/_logs/run_node${NODE_ID}_${TS}}"
MERGED_DIR="$OUT_BASE/_merged"
MANIFESTS_DIR="$OUT_BASE/_manifests"

mkdir -p "$OUT_BASE" "$LOG_DIR" "$MERGED_DIR" "$MANIFESTS_DIR"
exec > >(tee -a "$LOG_DIR/progress.log") 2>&1

# When resuming final/, verify stored manifest ⊆ current, then refresh
# _manifests to the current file (allows growing the scene set).
if [ -s "$MANIFESTS_DIR/real_manifest.jsonl" ] && [ "$RESUME" = "1" ]; then
    if ! _manifests_compatible "$MANIFESTS_DIR/real_manifest.jsonl" "$MANIFEST" "$OUT_BASE"; then
        echo "[FAIL] existing _manifests/ incompatible with MANIFEST=$MANIFEST"
        exit 1
    fi
    cp -f "$MANIFEST" "$MANIFESTS_DIR/real_manifest.jsonl"
    echo "[manifests] resume OK — refreshed $MANIFESTS_DIR/real_manifest.jsonl"
else
    cp -f "$MANIFEST" "$MANIFESTS_DIR/real_manifest.jsonl"
    echo "[manifests] $MANIFESTS_DIR/real_manifest.jsonl"
fi

CATALOG="$OUT_BASE/catalog.jsonl"

echo "================================================================"
echo "oracle collect  SIGN=$SIGN ($SIGN_CODE)  [$TS]"
echo "  MANIFEST        = $MANIFEST"
echo "  SCENES_ROOT     = $SCENES_ROOT"
echo "  OUT_BASE        = $OUT_BASE"
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
        idm|idm_rule) return 0 ;;
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
    local out_dir="$OUT_BASE/$policy"
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
    # Gate on alive worker pids (not `jobs -rp`) so the progress dashboard
    # background job does not steal a slot from N_WORKERS.
    _cpu_slots_busy() {
        local n=0 p
        for p in ${pids[@]+"${pids[@]}"}; do
            if kill -0 "$p" 2>/dev/null; then
                n=$((n + 1))
            fi
        done
        echo "$n"
    }
    # GIF / ShowBase: one process only (Panda3D is not multi-process safe).
    # COUNT/ROWS_LIMIT still shard via N_USE when IDM_CHUNKS>1 and no GIFs.
    if [ "${SAVE_GIFS:-0}" = "1" ] || [ "${IDM_CHUNKS:-1}" -le 1 ]; then
        if [ "${SAVE_GIFS:-0}" = "1" ] && [ "${IDM_CHUNKS:-1}" -gt 1 ]; then
            echo "[shard] $policy: SAVE_GIFS=1 → single process (no IDM_CHUNKS)"
        fi
        while [ "$(_cpu_slots_busy)" -ge "$N_WORKERS" ]; do
            sleep 1
        done
        run_one "$policy" &
        pids+=("$!")
        return 0
    fi

    # Shard over the rows that will actually be used. ROWS_LIMIT/COUNT are
    # already folded into N_USE above; sharding over the whole manifest
    # collected every row while the banner still printed the limit -- measured
    # as 240 of 240 scenes under ROWS_LIMIT=192. A limit that prints but does
    # not bind is worse than no limit at all.
    local n_rows
    n_rows="${N_USE:-}"
    if [ -z "$n_rows" ] || [ "$n_rows" -le 0 ]; then
        n_rows=$(_manifest_nrows "$MANIFEST")
    fi
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
        while [ "$(_cpu_slots_busy)" -ge "$N_WORKERS" ]; do
            sleep 1
        done
        run_one "$policy" --start "$start" --count "$count" --worker-id "$i" &
        pids+=("$!")
    done
}

fail=0
pids=()

# Live dashboard — defined before launch so we can print while workers are
# still being queued (IDM may fill N_WORKERS before PPO/GPU join).
_print_collect_progress() {
    # Pass N_USE so COUNT/ROWS_LIMIT caps the denominator (not full manifest size).
    "$PYTHON_BIN" - "$OUT_BASE" "$MANIFEST" "$EXTRA_SAMPLES_COMPREHENSIVE" "$LOG_DIR" "${N_USE:-}" <<'PY'
import os, re, sys, json
from pathlib import Path

out_base = Path(sys.argv[1])
manifest = Path(sys.argv[2])
extra = int(sys.argv[3] or 0)
log_dir = Path(sys.argv[4])
n_use_raw = (sys.argv[5] if len(sys.argv) > 5 else "").strip()

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

# COUNT / ROWS_LIMIT already folded into N_USE by the shell plan.
if n_use_raw:
    try:
        n_use = int(n_use_raw)
        if n_use > 0:
            n_rows = min(n_rows, n_use) if n_rows else n_use
    except ValueError:
        pass

idm = {"idm", "idm_rule"}
print("----- progress -----", flush=True)
any_pol = False
for pol_dir in sorted(p for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("_")):
    any_pol = True
    pol = pol_dir.name
    n_var = (1 + max(0, extra)) if pol in idm else 1
    target = n_rows * n_var if n_rows else 0
    done = 0
    seen = set()
    ledgers = list(pol_dir.glob("all_runs.jsonl"))
    ledgers += list(pol_dir.glob("all_runs.w*.jsonl"))
    # legacy <policy>/<slug>/all_runs.jsonl
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

# Background dashboard from the first worker onward (not after the full queue).
_PROGRESS_PID=""
_start_progress_dashboard() {
    if [ -n "$_PROGRESS_PID" ]; then
        return 0
    fi
    echo
    echo "[progress] live dashboard every ${PROGRESS_EVERY_S}s  (per-policy detail: $LOG_DIR/<policy>.log)"
    (
        # Stop when parent writes the sentinel or we get killed.
        while [ ! -f "$LOG_DIR/_progress_stop" ]; do
            _print_collect_progress
            sleep "$PROGRESS_EVERY_S"
        done
    ) &
    _PROGRESS_PID=$!
}

_stop_progress_dashboard() {
    mkdir -p "$LOG_DIR"
    : > "$LOG_DIR/_progress_stop"
    if [ -n "$_PROGRESS_PID" ] && kill -0 "$_PROGRESS_PID" 2>/dev/null; then
        kill "$_PROGRESS_PID" 2>/dev/null || true
        wait "$_PROGRESS_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_DIR/_progress_stop"
    _PROGRESS_PID=""
}

_run_gpu_pool() {
    local policies="$1" gpus="$2" ckpt="$3"
    shift 3
    local extra=("$@")
    IFS=',' read -ra gpu_list <<< "$gpus"
    local gi=0
    local -a pool_pids=()
    local max_parallel=$(( ${#gpu_list[@]} * JOBS_PER_GPU ))
    [ "$max_parallel" -lt 1 ] && max_parallel=1

    # Reap only THIS pool's pids — never gate on CPU jobs via `jobs -rp`.
    _gpu_pool_reap() {
        local -a alive=()
        local p
        for p in ${pool_pids[@]+"${pool_pids[@]}"}; do
            if kill -0 "$p" 2>/dev/null; then
                alive+=("$p")
            else
                wait "$p" 2>/dev/null || true
            fi
        done
        pool_pids=("${alive[@]}")
    }

    for policy in $policies; do
        _gpu_pool_reap
        while [ "${#pool_pids[@]}" -ge "$max_parallel" ]; do
            wait -n 2>/dev/null || true
            _gpu_pool_reap
            sleep 0.2
        done
        local gpu="${gpu_list[$((gi % ${#gpu_list[@]}))]}"
        gi=$((gi + 1))
        CUDA_VISIBLE_DEVICES="$gpu" run_one "$policy" --model-path "$ckpt" "${extra[@]}" &
        pool_pids+=("$!")
        pids+=("$!")
    done
}

# Wait until every pid in pids[] has exited; refresh progress while waiting.
_wait_phase_pids() {
    local phase_label="$1"
    echo
    echo "[progress] phase=$phase_label  waiting  (detail: $LOG_DIR/<policy>.log)"
    _print_collect_progress
    while true; do
        local alive=0
        local pid
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
    local pid
    for pid in ${pids[@]+"${pids[@]}"}; do
        wait "$pid" || fail=$((fail + 1))
    done
    _print_collect_progress
}

rm -f "$LOG_DIR/_progress_stop"
_start_progress_dashboard

# Phase 1: CPU policies to completion, then Phase 2: GPU.
# (No overlap — MetaDrive CPU contention was starving carl/plant2.)
if [ "$SKIP_CPU" != "1" ]; then
    echo
    echo "======== phase CPU ========"
    pids=()
    for policy in $POLICIES_CPU; do
        _run_cpu_policy "$policy"
    done
    _stop_progress_dashboard
    _wait_phase_pids "CPU"
    rm -f "$LOG_DIR/_progress_stop"
    _start_progress_dashboard
fi

pids=()
if [ "$SKIP_CARL" != "1" ]; then
    if [ -z "$CARL_CKPT" ]; then
        echo "[FAIL] CARL pool enabled but CARL_CKPT unset"
        fail=$((fail + 1))
    else
        echo
        echo "======== phase GPU (CARL) ========"
        pids=()
        _run_gpu_pool "$POLICIES_CARL" "$GPUS_CARL" "$CARL_CKPT"
        _stop_progress_dashboard
        _wait_phase_pids "GPU-CARL"
        rm -f "$LOG_DIR/_progress_stop"
        _start_progress_dashboard
    fi
fi

if [ "$SKIP_PLANT2" != "1" ]; then
    if [ -z "$PLANT2_CKPT" ]; then
        echo "[FAIL] PLANT2 pool enabled but PLANT2_CKPT unset"
        fail=$((fail + 1))
    else
        echo
        echo "======== phase GPU (PLANT2) ========"
        pids=()
        _run_gpu_pool "$POLICIES_PLANT2" "$GPUS_PLANT2" "$PLANT2_CKPT" \
            --plant2-action-mode "$PLANT2_ACTION_MODE"
        _stop_progress_dashboard
        _wait_phase_pids "GPU-PLANT2"
    fi
fi

_stop_progress_dashboard

echo "=== Collection finished. failures=$fail ==="

# Fold parallel shard ledgers into all_runs.jsonl (dedupe by scene/policy/variant).
echo "=== Merge worker shards → all_runs.jsonl ==="
"$PYTHON_BIN" - "$OUT_BASE" <<'PY'
import json
import sys
from pathlib import Path

def merge_dir(dir_path):
    shards = sorted(dir_path.glob("all_runs.w*.jsonl"))
    main = dir_path / "all_runs.jsonl"
    if not shards:
        return None
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
    label = dir_path.relative_to(Path(sys.argv[1]))
    print(f"  {label}: {len(by_key)} rows (from {len(shards)} shards)")
    return len(by_key)

out_base = Path(sys.argv[1])
n_merged = 0
for pol_dir in sorted(p for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("_")):
    if merge_dir(pol_dir) is not None:
        n_merged += 1
        continue
    for child in sorted(p for p in pol_dir.iterdir() if p.is_dir()):
        if merge_dir(child) is not None:
            n_merged += 1
print(f"shard merge: {n_merged} dirs")
PY

if [ "$SKIP_MERGE" = "1" ]; then
    echo "[skip] SKIP_MERGE=1"
    exit "$fail"
fi

echo "=== Merge → $MERGED_DIR/all_runs.jsonl ==="
: > "$MERGED_DIR/all_runs.jsonl"
find "$OUT_BASE" -name all_runs.jsonl \
    ! -path "$MERGED_DIR/*" ! -path "*/_logs/*" ! -path "*/_manifests/*" \
    | sort | while read -r f; do
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
sidecars = list(out_base.glob("*/by_scene/*/*/replay.json"))
if not sidecars:
    sidecars = list(out_base.glob("*/*/by_sign/*/by_scene/*/*/replay.json"))
for sidecar in sidecars:
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
    print("  (no sidecars found under */by_scene/*/*/replay.json)")
PY
fi

n_gif=$(find "$OUT_BASE" -path '*/gifs/*.gif' 2>/dev/null | wc -l || echo 0)
echo
echo "================================================================"
echo "Done."
echo "  SIGN       : $SIGN ($SIGN_CODE)"
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
echo "      --signs $SIGN_CODE --horizon $MAX_STEPS --out-dir $OUT_BASE/experts"
echo "  SIGN=$SIGN ../report/table.sh $OUT_BASE"
echo "================================================================"

exit "$fail"
