#!/usr/bin/env bash
# Thin wrapper — all logic in eval_full.py
set -euo pipefail
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$PIPELINE_DIR/eval_full.py" "$@"
