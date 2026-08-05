#!/usr/bin/env bash
# Thin wrapper — all logic in launch_ft.py
set -euo pipefail
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$PIPELINE_DIR/launch_ft.py" "$@"
