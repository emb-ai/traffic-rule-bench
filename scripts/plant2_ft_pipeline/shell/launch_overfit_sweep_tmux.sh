#!/usr/bin/env bash
# Run fixed-config 1traj overfit under tmux session `overfit_sweep`.
set -euo pipefail
PIPELINE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/jovyan/shares/SR006.nfs3/shepelev/conda_envs/arbelyaev-sdc/bin/python
SESSION=overfit_sweep
CONFIG="${OVERFIT_CONFIG:-$PIPELINE/tools/configs/overfit_1traj.yaml}"
LOG="${OVERFIT_SWEEP_LOG:-/tmp/overfit_1traj_run.log}"
GPU="${1:-0}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi
if pgrep -f '/overfit_1traj_sweep\.py' >/dev/null; then
  echo "overfit_1traj_sweep.py already running (pid $(pgrep -f '/overfit_1traj_sweep\.py' | head -1)). Refusing to double-start."
  exit 1
fi

tmux new-session -d -s "$SESSION" "cd '$PIPELINE' && $PY -u tools/overfit_1traj_sweep.py --config '$CONFIG' --gpu $GPU 2>&1 | tee -a '$LOG'; echo EXIT=\$?; sleep 3600"
echo "Started tmux session '$SESSION'"
echo "  attach: tmux attach -t $SESSION"
echo "  config: $CONFIG"
echo "  log:    $LOG"
