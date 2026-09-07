#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-zinkovich-plant2}"
ROOT="${1:-$REPO_ROOT/data/runs/failure_cases}"
LOG_DIR="$ROOT/_gif_render_logs"
mkdir -p "$LOG_DIR"

render_sign() {
  local sign="$1"
  local log="$LOG_DIR/render_${sign}.log"
  echo "[start] $sign -> $log"
  conda run -n "$ENV_NAME" env PYTHONPATH="$REPO_ROOT/third_party/metadrive:$REPO_ROOT" \
    python -u "$REPO_ROOT/tools/render_failure_case_gifs.py" \
    --root "$ROOT" --sign "$sign" \
    >"$log" 2>&1
  echo "[done] $sign"
}

for sign in main_road secondary_road yield stop; do
  render_sign "$sign" &
done

wait
python3 "$REPO_ROOT/tools/make_failure_cases_gif_index.py" --root "$ROOT"
echo "All GIF renders finished. Index: $ROOT/gifs_index.html"
