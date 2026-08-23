#!/usr/bin/env bash
# Build oracle_metrics_summary_top2.md for a collection OUT_BASE.
#
# Usage:
#   SIGN=yield ./table.sh data/trajectories/yield/trajectories_<ts>
#   SIGN=main ./table.sh data/trajectories/main_road/trajectories_<ts>
#   ./table.sh data/trajectories/<sign>/trajectories_<ts> .../experts
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_BASE="${1:?usage: $0 <OUT_BASE> [experts_dir]}"
# Resolve relative OUT_BASE against the caller's cwd *before* we cd elsewhere.
OUT_BASE="$(cd -- "$OUT_BASE" && pwd)"
EXPERTS_DIR="${2:-$OUT_BASE/experts}"
if [ -d "$EXPERTS_DIR" ]; then
    EXPERTS_DIR="$(cd -- "$EXPERTS_DIR" && pwd)"
fi
HORIZON="${HORIZON:-1500}"

# Official plate code from SIGN, catalog, or all_runs.
: "${SIGN:=}"
: "${PDD_CODE:=}"
SCRIPT_PARENT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$SCRIPT_PARENT${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "$PDD_CODE" ] && [ -n "$SIGN" ]; then
    PDD_CODE="$("$PYTHON_BIN" - "$SIGN" <<'PY'
import sys
from traffic_bench.eval.sign_registry import resolve_sign_token
print(resolve_sign_token(sys.argv[1]).sign_code)
PY
)" || PDD_CODE=""
fi
if [ -z "$PDD_CODE" ]; then
    PDD_CODE="$("$PYTHON_BIN" - "$OUT_BASE" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for cand in (root / "catalog.jsonl", root / "_merged" / "all_runs.jsonl"):
    if not cand.is_file():
        continue
    for ln in cand.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        code = row.get("sign_code") or row.get("pdd_code")
        if code:
            print(code)
            raise SystemExit
print("2.4")
PY
)"
fi

ALL_RUNS="$OUT_BASE/_merged/all_runs.jsonl"
if [ ! -s "$ALL_RUNS" ]; then
    ALL_RUNS="$EXPERTS_DIR/all_runs_dedup.jsonl"
fi
PICKS="$EXPERTS_DIR/experts_scene_uid_top2.jsonl"

if [ ! -s "$ALL_RUNS" ]; then
    echo "ERROR: missing all_runs at $ALL_RUNS"
    exit 1
fi
if [ ! -s "$PICKS" ]; then
    echo "ERROR: missing picks $PICKS — run python -m traffic_bench.oracle.select.coverage first"
    exit 1
fi

TABLE_OUT="$OUT_BASE/oracle_metrics"
mkdir -p "$TABLE_OUT"

"$PYTHON_BIN" "$SCRIPT_DIR/table.py" \
    --jsonl "$ALL_RUNS" \
    --picks "$PICKS" \
    --signs "$PDD_CODE" \
    --horizon "$HORIZON" \
    --beta 0.25 \
    --output-dir "$TABLE_OUT"

cp -f "$TABLE_OUT/oracle_metrics_summary.md" \
      "$TABLE_OUT/oracle_metrics_summary_top2.md"
echo "Wrote $TABLE_OUT/oracle_metrics_summary_top2.md (signs=$PDD_CODE)"
