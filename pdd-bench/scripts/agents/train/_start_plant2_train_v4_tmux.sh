#!/usr/bin/env bash
# Helper for tmux: batch 128 only (no fallback to 64).
export SKIP_BATCH_FALLBACK=1
export BATCH_FIRST=128
exec /home/jovyan/shares/SR006.nfs2/arbelyaev/sdc/pdd-bench/scripts/agents/train/run_plant2_train_benchmark_v4.sh
