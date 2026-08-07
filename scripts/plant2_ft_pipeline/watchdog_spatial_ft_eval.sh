#!/usr/bin/env bash
# Watch spatial FT (7 tmux) + eval orchestrator; on crash: diagnose, fix, restart.
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PIPELINE_DIR/_env.sh"

CT="$PIPELINE_DIR"
PLAN_T="$TRB_ROOT/plant2/PlanT"
CKPT_ROOT="$PLAN_T/checkpoints_ft"
LOGDIR="$CT/logs_pipeline_spatial_signs"
WATCH="$LOGDIR/watchdog_spatial_ft_eval.log"
STATE=/tmp/pipeline_spatial_signs.state
LAUNCH_FT="$CT/launch_plant2_ft_spatial_lr_sweep.sh"
LAUNCH_EVAL="$CT/launch_spatial_ft_eval_7gpu.sh"
RUN_FT="$CT/run_plant2_finetune.sh"
PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"

declare -A GPU LR ADDON
GPU[1e6]=0;  LR[1e6]=1e-6;  ADDON[1e6]=fvexp30_spatial_lr1e6
GPU[5e6]=1;  LR[5e6]=5e-6;  ADDON[5e6]=fvexp30_spatial_lr5e6
GPU[1e5]=2;  LR[1e5]=1e-5;  ADDON[1e5]=fvexp30_spatial_lr1e5
GPU[3e5]=3;  LR[3e5]=3e-5;  ADDON[3e5]=fvexp30_spatial_lr3e5
GPU[5e5]=4;  LR[5e5]=5e-5;  ADDON[5e5]=fvexp30_spatial_lr5e5
GPU[7e5]=5;  LR[7e5]=7e-5;  ADDON[7e5]=fvexp30_spatial_lr7e5
GPU[1e4]=6;  LR[1e4]=1e-4;  ADDON[1e4]=fvexp30_spatial_lr1e4
LRS=(1e6 5e6 1e5 3e5 5e5 7e5 1e4)

declare -A RESTARTS
MAX_RESTARTS=15
POLL_SEC=90

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$WATCH"; }

stage() {
  [[ -f "$STATE" ]] && awk -F= '/^stage=/{print $2; exit}' "$STATE" || echo unknown
}

ep029_path() {
  local lr="$1"
  echo "$CKPT_ROOT/fvexp30_spatial_lr${lr}/epoch=029_fvexp30_spatial_lr${lr}_1.ckpt"
}

ft_log() { echo "/tmp/plant2_ft_spatial_lr${1}.log"; }
ft_session() { echo "arbelyaev-ft-spatial-lr${1}"; }

ft_trainer_alive() {
  local lr="$1" addon="${ADDON[$lr]}"
  pgrep -af "run_lit_finetune|lit_finetune" 2>/dev/null | rg -q "$addon"
}

session_alive() {
  tmux has-session -t "$(ft_session "$1")" 2>/dev/null
}

log_tail_since() {
  local f="$1" pos="${2:-0}"
  [[ -f "$f" ]] || return 0
  tail -c +"$((pos + 1))" "$f" 2>/dev/null | tr -d '\r'
}

apply_fix() {
  local lr="$1" snip="$2"
  log "apply_fix lr=$lr"

  if echo "$snip" | rg -q 'Directory not empty.*plant_index'; then
    log "fix: clean stale plant_index_cache tmp dirs"
    rm -rf /tmp/plant_index_cache/*.tmp 2>/dev/null || true
    find /tmp/plant_index_cache -maxdepth 1 -name '*.tmp' -type d -exec rm -rf {} + 2>/dev/null || true
    return 0
  fi
  if echo "$snip" | rg -q 'CUDA out of memory|OutOfMemoryError'; then
    log "fix: reduce BATCH_SIZE for lr=$lr (env override on restart)"
    export BATCH_SIZE=672
    return 0
  fi
  if echo "$snip" | rg -q 'DataLoader worker|exited unexpectedly|_queue.Empty'; then
    log "fix: NUM_WORKERS=0 for lr=$lr"
    export NUM_WORKERS=0
    return 0
  fi
  log "no known auto-fix — will restart as-is"
  return 0
}

restart_ft_lr() {
  local lr="$1"
  local gpu="${GPU[$lr]}" lr_val="${LR[$lr]}" addon="${ADDON[$lr]}"
  local sess logf ckpt_dir
  sess=$(ft_session "$lr")
  logf=$(ft_log "$lr")
  ckpt_dir="$CKPT_ROOT/$addon"

  RESTARTS[$lr]=$((${RESTARTS[$lr]:-0} + 1))
  if ((${RESTARTS[$lr]} > MAX_RESTARTS)); then
    log "ABORT lr=$lr: too many restarts (${RESTARTS[$lr]})"
    return 1
  fi

  local pos=0
  [[ -f "$logf" ]] && pos=$(wc -c <"$logf" | tr -d ' ')
  local snip
  snip="$(log_tail_since "$logf" "$pos" | tail -n 50)"
  log "restart_ft lr=$lr n=${RESTARTS[$lr]} gpu=$gpu"
  log "crash tail:"
  echo "$snip" | tail -n 20 | tee -a "$WATCH"
  apply_fix "$lr" "$snip" || true

  tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
  sleep 2
  mkdir -p "$ckpt_dir"

  export SPLIT="${SPLIT:-$SHEPELEV/plant2_l1_fv_experts_split_signs}"
  export DS="${DS:-$SPLIT/train}"
  export DS_VAL="${DS_VAL:-$SPLIT/val}"
  export DS_LOCAL="${DS_LOCAL:-/tmp/plant2_ds_cache_spatial_aug}"
  export CACHE_SIZE_GB="${CACHE_SIZE_GB:-1800}"
  export MAX_EPOCHS="${MAX_EPOCHS:-30}"
  export BATCH_SIZE="${BATCH_SIZE:-1344}"
  export NUM_WORKERS="${NUM_WORKERS:-4}"
  export CKPT_EVERY_N_EPOCHS="${CKPT_EVERY_N_EPOCHS:-5}"
  export LR_SCHEDULER="${LR_SCHEDULER:-cosine_warmup}"
  export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
  export SEED="${SEED:-1}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTHONNOUSERSITE=1
  export PYTHON="$PY"

  tmux new-session -d -s "$sess" bash -lc "
set -uo pipefail
cd '$PLAN_T'
export CUDA_VISIBLE_DEVICES=$gpu
export LEARNING_RATE=$lr_val
export CHECKPOINT_ADDON=$addon
export SPLIT='$SPLIT' DS='$DS' DS_VAL='$DS_VAL' DS_LOCAL='$DS_LOCAL'
export CACHE_SIZE_GB=$CACHE_SIZE_GB MAX_EPOCHS=$MAX_EPOCHS
export BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS
export CKPT_EVERY_N_EPOCHS=$CKPT_EVERY_N_EPOCHS
export LR_SCHEDULER=$LR_SCHEDULER WARMUP_RATIO=$WARMUP_RATIO SEED=$SEED
export WANDB_MODE=$WANDB_MODE PYTHONNOUSERSITE=1 PYTHON='$PY'
echo \"FT_RESTART \$(date -Is) gpu=$gpu lr=$lr_val addon=$addon\" | tee -a '$logf'
bash '$RUN_FT' 2>&1 | tee -a '$logf'
echo \"FT_EXIT=\$? \$(date -Is)\" | tee -a '$logf'
exec bash
"
  log "restarted tmux $sess log=$logf"
}

check_ft_jobs() {
  local lr missing=0 dead=0
  for lr in "${LRS[@]}"; do
    if [[ -f "$(ep029_path "$lr")" ]]; then
      continue
    fi
    missing=$((missing + 1))
    if ft_trainer_alive "$lr"; then
      continue
    fi
    if session_alive "$lr"; then
      local logf pane
      logf=$(ft_log "$lr")
      pane=$(tmux capture-pane -t "$(ft_session "$lr")" -p 2>/dev/null | tail -3 || true)
      if echo "$pane" | rg -q 'FT_EXIT=|Error executing job|Traceback'; then
        log "FT crashed (idle tmux) lr=$lr"
        restart_ft_lr "$lr" || dead=$((dead + 1))
      elif echo "$pane" | rg -q 'Epoch [0-9]+:'; then
        : # progress bar in pane but pgrep missed — transient
      else
        log "FT idle tmux lr=$lr — checking log"
        if [[ -f "$logf" ]] && tail -n 30 "$logf" | rg -q 'Error executing job|Traceback|FT_EXIT=[^0]'; then
          restart_ft_lr "$lr" || dead=$((dead + 1))
        fi
      fi
    else
      log "FT session missing lr=$lr"
      restart_ft_lr "$lr" || dead=$((dead + 1))
    fi
  done
  log "ft_check missing_ep029=$missing dead_restarts=$dead"
}

eval_launcher_alive() {
  pgrep -f 'launch_spatial_ft_eval_7gpu\.sh' >/dev/null 2>&1
}

ensure_eval_launcher() {
  local st
  st=$(stage)
  if [[ "$st" == "eval_done" ]]; then
    return 0
  fi
  if eval_launcher_alive; then
    return 0
  fi
  log "eval launcher dead (stage=$st) — restarting"
  nohup bash "$LAUNCH_EVAL" >>"$LOGDIR/nohup_spatial_eval.out" 2>&1 &
  disown || true
  sleep 2
  if eval_launcher_alive; then
    log "eval launcher restarted pid=$(pgrep -f 'launch_spatial_ft_eval_7gpu\.sh' | head -1)"
  else
    log "ERROR: eval launcher failed to start"
  fi
}

log "watchdog_spatial_ft_eval start poll=${POLL_SEC}s"
log "watch=$WATCH eval_log=$LOGDIR/nohup_spatial_eval.out"

while true; do
  st=$(stage)
  if [[ "$st" == "eval_done" ]]; then
    # Still watch FT in case something odd; mostly exit when all ep029 + eval done
    all_done=1
    for lr in "${LRS[@]}"; do
      [[ -f "$(ep029_path "$lr")" ]] || all_done=0
    done
    if (( all_done )); then
      log "pipeline complete — watchdog exit"
      exit 0
    fi
  fi

  check_ft_jobs
  ensure_eval_launcher

  sleep "$POLL_SEC"
done
