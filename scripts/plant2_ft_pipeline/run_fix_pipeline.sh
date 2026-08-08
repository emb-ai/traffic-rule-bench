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
# Experts, scenes and the old dump follow from the label, exactly as in
# dump_plant2_l1_from_experts.sh — nothing has to be spelled out by hand:
#
#   SM=/mnt/.../smirnova COUNT=10 STAGES=dump bash run_fix_pipeline.sh
#   SM=/mnt/.../smirnova STAGES=all bash run_fix_pipeline.sh
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

# Roots the trajectories and scene packs live under. Same defaults the existing
# dump wrappers use; override any of them if a node mounts things elsewhere.
SHEP=${SHEP:-/home/jovyan/shares/SR006.nfs3/shepelev}
ZINK=${ZINK:-/mnt/virtual_ai0001053-01202_SR006-nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench/scripts/per_sign_bench}
EXPERTS_FILE=${EXPERTS_FILE:-experts_scene_uid_top1.jsonl}
TRAIN_PY=${TRAIN_PY:-$SHEP/conda_envs/arbelyaev-sdc/bin/python}

# Data out. Never reuse an existing dump or cache dir: the stride and the
# traffic are properties of the frames, so a stale cache silently mixes configs.
# The split script looks for the three *_signs dirs under $FIX_ROOT; the
# "from_experts" slot is ours.
FIX_ROOT=${FIX_ROOT:-$SM/plant2_fix}
DUMP_NEW=${DUMP_NEW:-$FIX_ROOT/plant2_l1_from_experts_signs}
SPLIT_NEW=${SPLIT_NEW:-$FIX_ROOT/plant2_l1_fv_experts_split_signs}
CACHE_ROOT=${CACHE_ROOT:-/tmp/plant2_cache_fix}
# Baseline data for the d3 run (old dumps, new stride) — the tree best_024 saw.
DUMP_OLD=${DUMP_OLD:-$SHEP/plant2_l1_from_experts_signs}
SPLIT_OLD=${SPLIT_OLD:-$SHEP/plant2_l1_fv_experts_split_signs}
BASE_CKPT=${BASE_CKPT:-$SHEP/plant2_checkpoints/epoch=029_final_1.ckpt}

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

pick_file () { for c in "$@"; do [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }; done; return 1; }
pick_dir  () { for c in "$@"; do [ -d "$c" ] && { printf '%s\n' "$c"; return 0; }; done; return 1; }

# label -> trajectory family / scene pack, as in dump_plant2_l1_from_experts.sh
traj_of () {
    case "$1" in
        2.1)   echo traj_main_2_1_train80 ;;
        2.3*)  echo traj_secondary_2_3_train80 ;;
        2.4)   echo traj_yield_2_4_train80 ;;
        2.5)   echo traj_stop_2_5_train80 ;;
        4.3)   echo traj_roundabout_4_3_train80 ;;
        *)     return 1 ;;
    esac
}
scenes_sub () {
    case "$1" in
        2.1)   echo main_sign/scenes/2_1 ;;
        2.3*)  echo secondary_sign/scenes/2_3 ;;
        2.4)   echo yield_sign/scenes/2_4 ;;
        2.5)   echo stop_sign/scenes/2_5 ;;
        4.3)   echo roundabout_sign/scenes/4_3 ;;
        *)     return 1 ;;
    esac
}
experts_for () {
    [ -n "${EXPERTS:-}" ] && { printf '%s\n' "$EXPERTS"; return 0; }
    local t; t=$(traj_of "$1") || return 1
    pick_file "$SM/collected_trajectories/traj-priority-signs/$t/experts/$EXPERTS_FILE" \
              "$SHEP/collected_trajectories/traj-priority-signs/$t/experts/$EXPERTS_FILE" \
              "$TRB/scripts/plant2_ft_pipeline/traj-priority-signs/$t/experts/$EXPERTS_FILE"
}
scenes_for () {
    [ -n "${SCENES_ROOT:-}" ] && { printf '%s\n' "$SCENES_ROOT"; return 0; }
    local s; s=$(scenes_sub "$1") || return 1
    pick_dir "$ZINK/$s" "$PSB/$s" "$SM/sdc/pdd-bench/scripts/per_sign_bench/$s"
}

# --- preflight ----------------------------------------------------------------
say "resolved paths (stages: $STAGES, labels: $LABELS)"
for lbl in $LABELS; do
    printf "  %-5s experts: %s\n" "$lbl" "$(experts_for "$lbl" || echo 'NOT FOUND')"
    printf "  %-5s scenes : %s\n" "$lbl" "$(scenes_for  "$lbl" || echo 'NOT FOUND')"
done
printf "  dump new: %s\n  dump old: %s\n  base ckpt: %s%s\n" \
    "$DUMP_NEW" "$DUMP_OLD" "$BASE_CKPT" \
    "$([ -f "$BASE_CKPT" ] || echo '   <-- MISSING')"
if [ ! -x "$PY" ]; then
    # The eval scripts run under whatever python3 is active; follow them rather
    # than dying on a hardcoded env path that only exists on some nodes.
    PY=$(command -v python3) || { echo "  !! no python3 on PATH"; exit 1; }
    echo "  note: PY not found, falling back to $PY"
fi
if has_stage train && [ ! -x "$TRAIN_PY" ]; then
    echo "  !! TRAIN_PY=$TRAIN_PY is not executable — set TRAIN_PY to the training env"
fi

# --- dump ---------------------------------------------------------------------
if has_stage dump; then
    mkdir -p "$DUMP_NEW/logs"
    for lbl in $LABELS; do
        experts=$(experts_for "$lbl") || {
            echo "!! $lbl: experts jsonl not found under \$SM/\$SHEP/collected_trajectories"
            echo "   pass EXPERTS=/abs/path/$EXPERTS_FILE to override"; continue; }
        scenes=$(scenes_for "$lbl") || {
            echo "!! $lbl: scenes root not found (tried \$ZINK and \$PSB)"
            echo "   pass SCENES_ROOT=/abs/scenes/dir to override"; continue; }

        n=$(wc -l < "$experts" | tr -d ' ')
        [ "$COUNT" -gt 0 ] && [ "$COUNT" -lt "$n" ] && n=$COUNT
        say "dump $lbl: $n scene(s) WITH auxiliary traffic -> $DUMP_NEW"
        echo "   experts = $experts"
        echo "   scenes  = $scenes"

        # Shard by expert row: one replay is a whole episode, so wall-clock is
        # linear in scenes and only sharding makes a full re-dump affordable.
        shards=$JOBS
        [ "$shards" -gt "$n" ] && shards=$n
        pids=()
        for ((s = 0; s < shards; s++)); do
            st=$(( s * n / shards )); en=$(( (s + 1) * n / shards ))
            cnt=$(( en - st )); [ "$cnt" -gt 0 ] || continue
            slug="${lbl//./_}_s$s"
            ( cd "$PSB" && $PY expert_replay_inenv.py \
                --experts "$experts" --scenes-root "$scenes" \
                --save-plant2-dir "$DUMP_NEW" --aux-agents \
                --ego-mode recorded --npc-mode recorded \
                --start "$st" --count "$cnt" \
                > "$DUMP_NEW/logs/${slug}.log" 2>&1 ) &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || true; done

        # The convoy is the whole point of this stage: report it, do not assume.
        aux_ok=$(cat "$DUMP_NEW/logs/${lbl//./_}"_s*.log 2>/dev/null \
                 | grep -c 'convoy of' || true)
        aux_bad=$(cat "$DUMP_NEW/logs/${lbl//./_}"_s*.log 2>/dev/null \
                  | grep -cE '\[aux\].*(skipped|could not register)' || true)
        routes=$(find "$DUMP_NEW/data" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        printf "   convoys registered=%s  aux failures=%s  routes on disk=%s\n" \
               "$aux_ok" "$aux_bad" "$routes"
        [ "$aux_ok" -eq 0 ] && grep -hE '^\[aux\]' "$DUMP_NEW/logs/${lbl//./_}"_s*.log \
            2>/dev/null | sort | uniq -c | head -5
    done

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
        case $run in d3) split=$SPLIT_OLD;; *) split=$SPLIT_NEW;; esac
        if [ ! -d "$split/train/data" ]; then
            echo "!! $run: no $split/train/data — skipped"; continue
        fi
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
        case $run in d3) split=$SPLIT_OLD;; *) split=$SPLIT_NEW;; esac
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
