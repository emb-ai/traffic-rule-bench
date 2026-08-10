#!/usr/bin/env bash
# Single-sign experiment: does auxiliary traffic in the training frames teach
# the model to obey the sign?
#
#   SM=/mnt/.../smirnova bash run_sign_pair_experiment.sh
#   SIGN=4.3 EPOCHS=8 bash run_sign_pair_experiment.sh
#
# Two finetunes on the same scenes, same recipe, same train/val halves. The one
# variable is the frames: `new` carries the convoy, `old` is what the baseline
# saw. They are compared against each other -- NOT against best_024, which was
# trained on every sign and would differ by the data mixture as well.
set -uo pipefail

SM=${SM:?set SM=/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova}
# shepelev's tree on nfs3 supplies the DEFAULTS for the old dump, the base
# checkpoint and the training python. Nodes without nfs3 can run too — each of
# those just has to be passed explicitly; the checks below name what is missing.
if [ -z "${SHEP:-}" ]; then
    for c in /home/jovyan/shares/SR006.nfs3/shepelev \
             /mnt/virtual_ai0001053-01202_SR006-nfs3/shepelev; do
        [ -d "$c" ] && SHEP=$c && break
    done
fi
SHEP=${SHEP:-}
TRB=$SM/traffic-rule-bench
PIPE=$TRB/scripts/plant2_ft_pipeline
PLANT=$TRB/plant2/PlanT
TEST=$TRB/pdd-bench/scripts/per_sign_bench/plant2_rule_test
PY=${PY:-$(command -v python3)}
# The prefill imports diskcache and the PlanT dataset, so it needs the training
# environment, not whatever python3 the shell happens to have.
PREFILL_PY=${PREFILL_PY:-${SHEP:+$SHEP/conda_envs/arbelyaev-sdc/bin/python}}
[ -n "$PREFILL_PY" ] && [ -x "$PREFILL_PY" ] || PREFILL_PY=$PY

SIGN=${SIGN:-2.5}
# Label mode and loss rebalance for the v2 pair (defect D4). TS_LOOKAHEAD=1
# relabels target_speed with the min ego speed over the future window inside
# the dataset; the tag and cache change with it so v1/v2 runs never mix.
TS_LOOKAHEAD=${TS_LOOKAHEAD:-0}
STOP_SPEED_LOSS_WEIGHT=${STOP_SPEED_LOSS_WEIGHT:-1.0}
_v2=""; [ "$TS_LOOKAHEAD" != 0 ] && _v2="v2"
TAG=${TAG:-only$(echo "$SIGN" | tr -d '.')${_v2}}
FIX_ROOT=${FIX_ROOT:-$SM/plant2_fix}
DUMP_NEW=${DUMP_NEW:-$FIX_ROOT/plant2_l1_from_experts_signs}
DUMP_OLD=${DUMP_OLD:-${SHEP:+$SHEP/plant2_l1_from_experts_signs}}
# Without nfs3 the baseline dump is unreachable; synthesize it from the new
# dump instead (same replayed episodes minus the convoy's vehicle boxes).
if [ -z "$DUMP_OLD" ] || [ ! -d "$DUMP_OLD/data" ]; then
    DUMP_OLD=$FIX_ROOT/old_synth_${SIGN}
    SYNTH_OLD=1
fi
SPLIT_NEW=${SPLIT_NEW:-$FIX_ROOT/split_${SIGN}_new}
SPLIT_OLD=${SPLIT_OLD:-$FIX_ROOT/split_${SIGN}_old}
CACHE_ROOT=${CACHE_ROOT:-/tmp/plant2_cache_pair${_v2}}
BASE_CKPT=${BASE_CKPT:-${SHEP:+$SHEP/plant2_checkpoints/epoch=029_final_1.ckpt}}

# Fail on what is actually missing, by name, before any stage runs.
[ -n "$BASE_CKPT" ] && [ -f "$BASE_CKPT" ] \
    || { echo "!! BASE_CKPT not found (${BASE_CKPT:-unset}) — pass BASE_CKPT=/abs/base.ckpt"; exit 1; }

EPOCHS=${EPOCHS:-12}
LR=${LR:-1e-5}
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "${GPUS:-0}" -ge 1 ] || GPUS=1
JOBS=${JOBS:-8}
HALVES=${HALVES:-"new old"}
STAGES=${STAGES:-all}

has_stage () { [ "$STAGES" = all ] || [[ " $STAGES " == *" $1 "* ]]; }
say () { echo; echo "######## [$(date +%H:%M:%S)] $*"; }

say "sign=$SIGN gpus=$GPUS epochs=$EPOCHS lr=$LR halves='$HALVES' ts_lookahead=$TS_LOOKAHEAD stop_w=$STOP_SPEED_LOSS_WEIGHT tag=$TAG"
echo "  new frames: $DUMP_NEW"
echo "  old frames: $DUMP_OLD"

# --- splits -------------------------------------------------------------------
if has_stage split; then
    # The synthesizer is resume-safe (complete routes are skipped), so run it
    # whenever the old half is synthetic — a partial tree from an interrupted
    # attempt gets finished instead of being mistaken for done.
    if [ "${SYNTH_OLD:-0}" = 1 ]; then
        say "old half: synthesizing no-traffic frames -> $DUMP_OLD"
        $PY "$PIPE/make_old_half_from_new.py" \
            --new-dump "$DUMP_NEW" --out "$DUMP_OLD" --jobs ${JOBS:-16} \
            || { echo "!! synthesis failed"; exit 1; }
    fi
    say "matched splits -> $SPLIT_NEW / $SPLIT_OLD"
    ( cd "$PIPE" && SHEPELEV="$SHEP" PLAN_T="$PLANT" $PY make_sign_pair_splits.py \
        --sign "$SIGN" --new-dump "$DUMP_NEW" --old-dump "$DUMP_OLD" \
        --out-new "$SPLIT_NEW" --out-old "$SPLIT_OLD" ) || {
        echo "!! split failed — stopping"; exit 1; }
fi

for half in $HALVES; do
    case $half in
        new) split=$SPLIT_NEW ;;
        old) split=$SPLIT_OLD ;;
        *)   echo "!! unknown half '$half'"; continue ;;
    esac
    addon="${TAG}_${half}"
    [ -f "$split/split_meta.json" ] || { echo "!! no $split — run the split stage"; continue; }

    if has_stage cache; then
        say "cache $half -> $CACHE_ROOT/$half"
        ( cd "$PIPE" && DS="$split/train" DS_VAL="$split/val" \
          DS_LOCAL="$CACHE_ROOT/$half" CACHE_SIZE_GB=${CACHE_SIZE_GB:-200} \
          TS_LOOKAHEAD="$TS_LOOKAHEAD" \
          PLAN_T="$PLANT" $PREFILL_PY prefill_plant2_diskcache.py 2>&1 | tail -3 ) \
          || echo "!! prefill failed for $half"
    fi

    if has_stage train; then
        say "train $addon (12 epochs on $(basename "$split"))"
        env SPLIT="$split" DS="$split/train" DS_VAL="$split/val" \
            DS_LOCAL="$CACHE_ROOT/$half" CACHE_SIZE_GB=${CACHE_SIZE_GB:-200} \
            SEED=1 CHECKPOINT_ADDON="$addon" CKPT_EVERY_N_EPOCHS=3 \
            GPUS="$GPUS" CKPT0="$BASE_CKPT" SHEPELEV="$SHEP" \
            TS_LOOKAHEAD="$TS_LOOKAHEAD" STOP_SPEED_LOSS_WEIGHT="$STOP_SPEED_LOSS_WEIGHT" \
            DDP_STRATEGY="${DDP_STRATEGY:-ddp_find_unused_parameters_true}" \
            LEARNING_RATE="$LR" MAX_EPOCHS="$EPOCHS" LR_SCHEDULER=cosine_warmup \
            bash "$PIPE/run_plant2_finetune.sh" > "$TRB/plant2/ft_${addon}.log" 2>&1
        grep -E "val/loss_all|Epoch |Error|Traceback" "$TRB/plant2/ft_${addon}.log" | tail -8
        echo "   log: $TRB/plant2/ft_${addon}.log"
    fi

    if has_stage eval; then
        ck=$(ls -t "$PLANT/checkpoints_ft/$addon"/best_*.ckpt 2>/dev/null | head -1)
        if [ -z "$ck" ]; then
            echo "!! $addon produced no checkpoint — see $TRB/plant2/ft_${addon}.log"
            continue
        fi
        say "eval $addon -> $(basename "$ck")"
        ( cd "$TEST" && LABELS="$SIGN" CKPT="$ck" JOBS="$JOBS" \
          PREFIX_OVERRIDE="${TAG}${half}" CONFIGS="both" \
          bash run_sign_ab.sh 2>&1 | tail -20 )
    fi
done

say "comparison"
( cd "$TEST" && $PY collect_metrics.py --runs "${TAG}*" --baseline plant2_default >/dev/null 2>&1 \
  && grep -v 'data:image' output/_all_metrics/all_metrics.md | tail -30 ) \
  || echo "collect_metrics failed"

cat <<EOF

######## how to read it
  ${TAG}new_both  vs  ${TAG}old_both   — the convoy, and nothing else
Same scenes, same halves, same recipe. A rise in Sign SR with the dest rate
holding is the effect; a rise with dest collapsing is a car that stopped
forever and never reached the sign. Neither run is comparable to best_024:
both were trained on $SIGN alone.
EOF
