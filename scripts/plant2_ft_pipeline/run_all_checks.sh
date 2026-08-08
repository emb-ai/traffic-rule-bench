#!/usr/bin/env bash
# The whole campaign in one queue, so the results can be read in one sitting.
#
#   SM=/mnt/.../smirnova CKPT=/abs/best_024...ckpt bash run_all_checks.sh
#
# Order is deliberate:
#   1. wait out any dump still in flight (the data stages own the machine)
#   2. control A/B on 4.3 against the baseline checkpoint — answers whether the
#      zero sign SR is specific to the stop sign, and costs a quarter hour
#   3. three finetunes, each cached / trained / evaluated in turn:
#        d2    new dumps (convoy), stride 1
#        d3    old dumps,          stride 2
#        d2d3  new dumps,          stride 2
#   4. one table over every run on disk
#
# Each step writes its own log under $LOGDIR and never aborts the queue: a
# finetune that dies still leaves the others comparable.
set -uo pipefail

SM=${SM:?set SM=/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova}
TRB=$SM/traffic-rule-bench
PIPE=$TRB/scripts/plant2_ft_pipeline/run_fix_pipeline.sh
TEST=$TRB/pdd-bench/scripts/per_sign_bench/plant2_rule_test
CKPT=${CKPT:?set CKPT=/abs/path/best_024_*.ckpt (the baseline)}
LABELS=${LABELS:-"2.5 4.3"}
RUNS=${RUNS:-"d2 d3 d2d3"}
FIX_ROOT=${FIX_ROOT:-$SM/plant2_fix}
SPLIT_NEW=${SPLIT_NEW:-$FIX_ROOT/plant2_l1_fv_experts_split_signs}
DUMP_NEW=${DUMP_NEW:-$FIX_ROOT/plant2_l1_from_experts_signs}
LOGDIR=${LOGDIR:-$SM/fix_logs}
mkdir -p "$LOGDIR"

step () { echo; echo "======== [$(date +'%F %H:%M:%S')] $*"; }

# --- 1. let the data stages finish --------------------------------------------
if pgrep -f expert_replay_inenv.py >/dev/null; then
    step "waiting for the running dump to finish"
    while pgrep -f expert_replay_inenv.py >/dev/null; do sleep 60; done
fi
if pgrep -f make_train_val_split_fv_experts_signs.py >/dev/null; then
    step "waiting for the running split to finish"
    while pgrep -f make_train_val_split_fv_experts_signs.py >/dev/null; do sleep 30; done
fi

step "data check"
echo "-- convoys registered per size:"
grep -h '^\[aux\]' "$DUMP_NEW"/logs/*.log 2>/dev/null | sed 's/(.*//' | sort | uniq -c
echo "-- routes in the new dump: $(find "$DUMP_NEW/data" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
if [ ! -f "$SPLIT_NEW/split_meta.json" ]; then
    echo "-- no split yet, running assemble + split now"
    SM="$SM" LABELS="$LABELS" STAGES="assemble split" bash "$PIPE" \
        > "$LOGDIR/data.log" 2>&1 || true
fi
if [ ! -f "$SPLIT_NEW/split_meta.json" ]; then
    echo "!! still no $SPLIT_NEW/split_meta.json — nothing to train on, stopping"
    echo "   see $LOGDIR/data.log"
    exit 1
fi
python3 -c "
import json
m = json.load(open('$SPLIT_NEW/split_meta.json'))
print('-- routes per sign:', {k: v['N'] for k, v in sorted(m['per_sign'].items())})
"

# --- 2. control A/B on 4.3 (baseline checkpoint) ------------------------------
# Off by default: it delays the finetunes by ~15 min and only re-confirms on a
# second sign what 2.5 already showed. AB43=1 puts it back in front.
if [ "${AB43:-0}" = 1 ]; then
    step "control A/B on 4.3 (prefix vs both) -> $LOGDIR/ab43.log"
    ( cd "$TEST" && LABELS=4.3 CKPT="$CKPT" JOBS=8 CONFIGS="prefix both" \
      bash run_sign_ab.sh ) > "$LOGDIR/ab43.log" 2>&1 || true
    grep -E '^\| 4\.3|episode lines' "$LOGDIR/ab43.log" | tail -8
fi

# --- 3. finetunes -------------------------------------------------------------
for run in $RUNS; do
    step "finetune $run -> $LOGDIR/train_$run.log"
    SM="$SM" LABELS="$LABELS" RUNS="$run" STAGES="cache train eval" \
        bash "$PIPE" > "$LOGDIR/train_$run.log" 2>&1 || true
    ck=$(ls -t "$TRB/plant2/PlanT/checkpoints_ft/fix_$run"/best_*.ckpt 2>/dev/null | head -1)
    if [ -z "$ck" ]; then
        echo "!! $run produced no checkpoint — see $LOGDIR/train_$run.log"
    else
        echo "   best: $ck"
    fi
done

# --- 4. everything in one table ----------------------------------------------
step "final table"
( cd "$TEST" && python3 collect_metrics.py --runs '*' --baseline plant2_default >/dev/null 2>&1 \
  && grep -v 'data:image' output/_all_metrics/all_metrics.md ) || echo "collect_metrics failed"

cat <<'EOF'

======== how to read it
Compare like with like — the same eval switches on both sides:
  fxd2_both   vs ab25_both / ab43_both   what the missing convoy cost
  fxd3_both   vs the same                what the 0.8 s waypoint horizon cost
  fxd2d3_both vs the same                whether the two compose

A run only counts if its log has no "get_action failed", N matches the label's
scene count (153 for 2.5, 210 for 4.3), and dest rate does not collapse: sign
compliance measured on a car that never reaches the sign is vacuous.
EOF
