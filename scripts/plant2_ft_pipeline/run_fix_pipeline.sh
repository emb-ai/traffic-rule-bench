#!/usr/bin/env bash
# Re-dump junction signs with traffic, re-finetune, and evaluate — one queue.
#
# Tests the two training-side defects found in the sign investigation:
#   D2  junction-sign dumps contain no vehicles, because the replay env never
#       registered the bench's AuxiliaryAgentsManager (expert_replay_inenv
#       --aux-agents fixes it)
#   D3  waypoints are 0.1 s apart while the pretrained model and both
#       controllers assume 0.25 s (PlanT dataset WPS_STRIDE fixes it)
#
# Stages are separately runnable so a failure does not cost the whole queue:
#   STAGES="dump split cache train eval"      (default: all)
#
#   SM=/mnt/.../smirnova EXPERTS=/abs/experts_top1.jsonl \
#   BASE_CKPT=/abs/pretrained.ckpt STAGES=all bash run_fix_pipeline.sh
#
# What varies between the three finetunes (so each defect is attributable):
#   d2      new dumps (traffic), stride 1
#   d3      old dumps,            stride 2
#   d2d3    new dumps,            stride 2
set -uo pipefail

SM=${SM:?set SM=/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova}
TRB=$SM/traffic-rule-bench
PSB=$TRB/pdd-bench/scripts/per_sign_bench
PLANT=$TRB/plant2/PlanT
PY=${PY:-$SM/.conda-envs/plant2/bin/python3}
TRAIN_PY=${TRAIN_PY:-$PY}

# Data in / out. Never reuse an existing dump or cache dir: the stride and the
# traffic are properties of the frames, so a stale cache silently mixes configs.
EXPERTS=${EXPERTS:?set EXPERTS=/abs/path/experts_scene_uid_top1.jsonl}
SCENES_ROOT=${SCENES_ROOT:?set SCENES_ROOT=/abs/scenes root for the label}
DUMP_OLD=${DUMP_OLD:?set DUMP_OLD=/abs/existing dump root (for the d3 run)}
# The split script looks for the three *_signs dirs under $FIX_ROOT; the
# "from_experts" slot is ours.
FIX_ROOT=${FIX_ROOT:-$SM/plant2_fix}
DUMP_NEW=${DUMP_NEW:-$FIX_ROOT/plant2_l1_from_experts_signs}
SPLIT_NEW=${SPLIT_NEW:-$FIX_ROOT/plant2_l1_fv_experts_split_signs}
CACHE_ROOT=${CACHE_ROOT:-/tmp/plant2_cache_fix}
BASE_CKPT=${BASE_CKPT:?set BASE_CKPT=/abs/pretrained.ckpt}

LABELS=${LABELS:-"2.5 4.3"}
COUNT=${COUNT:-0}                 # 0 = all expert rows; small value = smoke test
JOBS=${JOBS:-8}
EPOCHS=${EPOCHS:-12}
LR=${LR:-1e-5}
GPUS=${GPUS:-8}
STAGES=${STAGES:-all}
RUNS=${RUNS:-"d2 d3 d2d3"}

export PYTHONPATH=$TRB/pdd-bench:$TRB/metadrive:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

has_stage () { [ "$STAGES" = all ] || [[ " $STAGES " == *" $1 "* ]]; }
say () { echo; echo "######## [$(date +%H:%M:%S)] $*"; }

# --- dump ---------------------------------------------------------------------
if has_stage dump; then
    say "dump: replaying experts WITH auxiliary traffic -> $DUMP_NEW"
    mkdir -p "$DUMP_NEW"
    extra=""
    [ "$COUNT" -gt 0 ] && extra="--count $COUNT"
    $PY "$PSB/expert_replay_inenv.py" \
        --experts "$EXPERTS" --scenes-root "$SCENES_ROOT" \
        --save-plant2-dir "$DUMP_NEW" --aux-agents \
        --ego-mode recorded --npc-mode recorded $extra \
        2>&1 | tee "$DUMP_NEW/dump.log" | grep -E '^\[aux\]|^\[replay\]|ERROR' | tail -20

    say "dump check: do the new frames contain cars?"
    $PY "$PSB/dump_model_input.py" --split "$DUMP_NEW" --plant-dir "$PLANT" \
        --samples 3 --stride 40 2>&1 | grep -E "состав|сэмпл" || true
    echo "   (expect 'car: N' next to the sign — old dumps show the sign alone)"
fi

# --- split --------------------------------------------------------------------
if has_stage split; then
    say "split: $DUMP_NEW -> $SPLIT_NEW"
    # No CLI: sources and OUT come from SHEPELEV via _paths.py, and missing
    # source dirs are skipped — so pointing SHEPELEV at $FIX_ROOT splits exactly
    # the dump we just made.
    ( cd "$(dirname "$0")" && SHEPELEV="$FIX_ROOT" PLAN_T="$PLANT" \
      $PY make_train_val_split_fv_experts_signs.py 2>&1 | tail -8 ) \
      || echo "!! split failed"
    ls -d "$SPLIT_NEW/train/data" "$SPLIT_NEW/val/data" 2>/dev/null \
      || echo "!! split produced no train/val — check the log above"
fi

# --- cache --------------------------------------------------------------------
if has_stage cache; then
    for run in $RUNS; do
        case $run in d3) src=$DUMP_OLD; split=${SPLIT_OLD:?set SPLIT_OLD for the d3 run};;
                     *)  src=$DUMP_NEW; split=$SPLIT_NEW;; esac
        stride=1; [ "$run" = d3 ] && stride=2; [ "$run" = d2d3 ] && stride=2
        say "cache: $run (stride $stride) from $split"
        ( cd "$(dirname "$0")" && \
          DS="$split/train" DS_VAL="$split/val" DS_LOCAL="$CACHE_ROOT/$run" \
          CACHE_SIZE_GB=${CACHE_SIZE_GB:-700} WPS_STRIDE=$stride \
          PLAN_T="$PLANT" $PY prefill_plant2_diskcache.py 2>&1 | tail -3 ) \
          || echo "!! prefill failed for $run"
    done
fi

# --- train --------------------------------------------------------------------
if has_stage train; then
    for run in $RUNS; do
        case $run in d3) split=${SPLIT_OLD:-$SPLIT_NEW};; *) split=$SPLIT_NEW;; esac
        stride=1; [ "$run" = d3 ] && stride=2; [ "$run" = d2d3 ] && stride=2
        addon="fix_${run}"
        say "train: $addon (split=$(basename "$split") stride=$stride lr=$LR epochs=$EPOCHS)"
        ( cd "$PLANT" && \
          DS="$split/train" DS_VAL="$split/val" DS_LOCAL="$CACHE_ROOT/$run" \
          CACHE_SIZE_GB=${CACHE_SIZE_GB:-700} SEED=1 CHECKPOINT_ADDON="$addon" \
          CKPT_EVERY_N_EPOCHS=3 WPS_STRIDE=$stride \
          $TRAIN_PY lit_finetune.py \
            user.working_dir="$TRB/plant2" \
            resume_path="$BASE_CKPT" \
            gpus=$GPUS lr_scheduler=cosine_warmup \
            model.training.max_epochs=$EPOCHS \
            model.training.learning_rate=$LR \
            model.training.augment_parked=False \
            +model.training.filter_routes=false \
            +model.training.wps_stride=$stride \
          2>&1 | tee "$TRB/plant2/ft_${addon}.log" \
          | grep -E "Trainable routes|sign_id resolve|val/loss_all|Epoch" | tail -20 )
        echo "   checkpoints: $TRB/plant2/PlanT/checkpoints_ft/$addon/"
    done
fi

# --- eval ---------------------------------------------------------------------
if has_stage eval; then
    TEST=$PSB/plant2_rule_test
    for run in $RUNS; do
        addon="fix_${run}"
        ck=$(ls -t "$TRB/plant2/PlanT/checkpoints_ft/$addon"/best_*.ckpt 2>/dev/null | head -1)
        [ -z "$ck" ] && { echo "!! no best_*.ckpt for $addon — skipped"; continue; }
        say "eval: $addon -> $(basename "$ck")"
        ( cd "$TEST" && LABELS="$LABELS" CKPT="$ck" JOBS=$JOBS \
          PREFIX_OVERRIDE="fx${run}" CONFIGS="both narrow" \
          bash run_sign_ab.sh 2>&1 | tail -30 )
    done

    say "final table (new runs next to the best_024 baseline)"
    ( cd "$TEST" && $PY collect_metrics.py --runs '*' --baseline plant2_default > /dev/null 2>&1 \
      && grep -v 'data:image' output/_all_metrics/all_metrics.md | head -80 )
fi

cat <<'EOF'

######## how to read it
  d2 vs the best_024 baseline  — what the missing convoy cost
  d3 vs the same baseline      — what the 0.8 s waypoint horizon cost
  d2d3                         — whether the two compose

Gates for every eval run: no "get_action failed" in the log (else the model
never loaded and the numbers are fiction), N equal to the label's scene count,
and dest rate that does not collapse — a car that stops forever scores perfect
sign compliance while arriving nowhere.
EOF
