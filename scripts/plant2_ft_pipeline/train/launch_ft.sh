#!/usr/bin/env bash
# Thin wrapper — all logic in launch_ft.py (same directory)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$DIR/launch_ft.py" "$@"
