#!/usr/bin/env bash
# Oracle expert selection for a collected trajectory tree.
#
#   SIGN=crosswalk ./coverage.sh
#   SIGN=yield HORIZON=1500 ./coverage.sh
#   SIGN=crosswalk ROOT=data/trajectories/crosswalk/trajectories_<ts> ./coverage.sh
#
# Defaults: ROOT=data/trajectories/<sign>/final, OUT_DIR=$ROOT/experts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${SIGN:?usage: SIGN=<eval_id> $0   (e.g. SIGN=crosswalk $0)}"

_profile() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from traffic_bench.eval.sign_registry import resolve_sign_token
p = resolve_sign_token(sys.argv[1])
print(p.id)
print(p.data_subdir)
print(p.sign_code)
PY
}

mapfile -t _PROF < <(_profile "$SIGN")
SIGN_ID="${_PROF[0]}"
DATA_SUBDIR="${_PROF[1]}"
SIGN_CODE="${_PROF[2]}"

_resolve_dir() {
    local raw="$1"
    if [[ "$raw" = /* ]] && [ -d "$raw" ]; then
        (cd -- "$raw" && pwd)
        return 0
    fi
    if [ -d "$REPO_ROOT/$raw" ]; then
        (cd -- "$REPO_ROOT/$raw" && pwd)
        return 0
    fi
    if [ -d "$raw" ]; then
        (cd -- "$raw" && pwd)
        return 0
    fi
    echo "[FAIL] directory not found: $raw (tried $REPO_ROOT/$raw)" >&2
    return 1
}

: "${ROOT:=data/trajectories/${DATA_SUBDIR}/final}"
ROOT="$(_resolve_dir "$ROOT")"
: "${OUT_DIR:=$ROOT/experts}"
: "${HORIZON:=1500}"
: "${CATALOG:=$ROOT/catalog.jsonl}"

if [ ! -f "$CATALOG" ] && [ -f "$ROOT/_merged/catalog.jsonl" ]; then
    CATALOG="$ROOT/_merged/catalog.jsonl"
fi

EXTRA=()
if [ -n "${POLICIES:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA+=(--policies ${POLICIES})
fi
if [ "${MIN_JOIN_RATE+x}" = x ]; then
    EXTRA+=(--min-join-rate "$MIN_JOIN_RATE")
fi

echo "SIGN=$SIGN_ID ($SIGN_CODE)  ROOT=$ROOT  HORIZON=$HORIZON"

exec "$PYTHON_BIN" -m traffic_bench.oracle.select.coverage \
    --root "$ROOT" \
    --catalog "$CATALOG" \
    --signs "$SIGN_ID" \
    --horizon "$HORIZON" \
    --out-dir "$OUT_DIR" \
    "${EXTRA[@]}" \
    "$@"
