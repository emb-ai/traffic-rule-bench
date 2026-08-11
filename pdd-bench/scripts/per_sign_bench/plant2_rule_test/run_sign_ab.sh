#!/usr/bin/env bash
# A/B the eval-side sign channels on one PDD label, with sanity gates.
#
#   LABEL=2.5 CKPT=/abs/ckpt.ckpt bash run_sign_ab.sh
#   LABELS="2.5 4.3" CKPT=... bash run_sign_ab.sh          # queue several signs
#   LABEL=4.3 CKPT=... CONFIGS="both narrow" bash run_sign_ab.sh
#
# Channels under test (see plant2_adapter / metadrive_obs_to_plant2):
#   prefix — signs demoted to `static`, no sign token  (pre-fix behaviour)
#   objs   — signs carry their own PDD class
#   token  — global sign_id token only
#   both   — both channels (the shipped default)
#   narrow — both + the object filter the training used (50 m / front x2)
#   remap  — both + NPC cars rewritten as the sign class (does the model react
#            to the sign token at all, even on a moving object?)
#
# Every run is gated: a swallowed model-load failure prints "get_action failed"
# per step and yields a car that never moves — which scores perfect compliance
# at zero arrival. Such a run is rejected, not reported.
set -uo pipefail
cd "$(dirname "$0")"

# The eval must see the repo's OWN metadrive submodule (it carries the sign
# channels) and pdd-bench, regardless of what the active env has installed —
# some nodes ship a site-packages metadrive without even a `manager` module.
# Self-export like eval_ckpt_sequential.sh does, and fail fast if it's wrong.
TRB_ROOT="$(cd ../../../.. && pwd)"
export PYTHONPATH="$TRB_ROOT/metadrive:$TRB_ROOT/pdd-bench:${PYTHONPATH:-}"
python3 - <<'PYCHECK' || exit 1
from metadrive.manager.base_manager import BaseManager  # noqa: F401
import metadrive, os
root = os.environ["PYTHONPATH"].split(":")[0]
assert os.path.realpath(metadrive.__file__).startswith(os.path.realpath(root)),     f"metadrive resolves outside the repo: {metadrive.__file__}"
PYCHECK

LABELS=${LABELS:-${LABEL:-2.5}}
CKPT=${CKPT:?set CKPT=/abs/path/to.ckpt}
JOBS=${JOBS:-8}
POLICY=${POLICY:-plant2}
CONFIGS=${CONFIGS:-"prefix objs token both narrow remap"}
EXPECT_N=${EXPECT_N:-0}          # 0 = do not check the scene count

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

env_for () {                      # config name -> env assignments
    case "$1" in
        prefix) echo "PLANT2_SIGN_OBJS=0 PLANT2_SIGN_TOKEN=0" ;;
        objs)   echo "PLANT2_SIGN_OBJS=1 PLANT2_SIGN_TOKEN=0" ;;
        token)  echo "PLANT2_SIGN_OBJS=0 PLANT2_SIGN_TOKEN=1" ;;
        both)   echo "PLANT2_SIGN_OBJS=1 PLANT2_SIGN_TOKEN=1" ;;
        narrow) echo "PLANT2_SIGN_OBJS=1 PLANT2_SIGN_TOKEN=1 PLANT2_OBJ_MAX_DIST=50 PLANT2_OBJ_FRONT_FACTOR=2" ;;
        remap)  echo "PLANT2_SIGN_OBJS=1 PLANT2_SIGN_TOKEN=1 PLANT2_REMAP_NPC_TO_SIGN=$LABEL" ;;
        *)      echo "" ;;
    esac
}

run_label () {
LABEL=$1
# PREFIX_OVERRIDE replaces only the leading tag; the label digits are always
# appended, so queueing several signs cannot collide (ab25/ab43, fxd225/fxd243).
PREFIX="${PREFIX_OVERRIDE:-ab}$(echo "$LABEL" | tr -d '.')"

echo
echo "############ label=$LABEL policy=$POLICY jobs=$JOBS ckpt=$(basename "$CKPT")"
echo "############ configs: $CONFIGS"

for cfg in $CONFIGS; do
    vars=$(env_for "$cfg")
    if [ -z "$vars" ]; then
        echo "!! unknown config '$cfg' — skipped"; continue
    fi
    rn="${PREFIX}_${cfg}"
    log="${rn}.log"
    echo
    echo "=== [$(date +%H:%M:%S)] $rn   [$vars]"

    # Per-step speed distribution: shows whether the model *wants* to stop
    # (mass on bin 0) even when the controller does not deliver it.
    env $vars PLANT2_SPEED_LOG_PATH="$(pwd)/${rn}_speed.jsonl" \
        python3 eval_checkpoint_on_test.py \
            --policies "$POLICY" --model-paths "$POLICY:$CKPT" \
            --only "$LABEL" --jobs "$JOBS" --keep-going --run-name "$rn" \
            > "$log" 2>&1
    rc=$?

    fails=$(grep -c "get_action failed" "$log" || true)
    eps=$(find "output/$rn" -name 'episodes_*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
    printf "    exit=%s  model-load failures=%s  episode lines=%s\n" "$rc" "$fails" "$eps"
    if [ "$fails" -gt 0 ]; then
        echo "    !! REJECTED: model never loaded — metrics from this run are meaningless"
        echo "       (usually the plant2 submodule is on the wrong commit)"
    fi
done

echo
echo "=== summary for $LABEL"
# Rebuild any report the eval left unfinalized; a no-op when they all exist.
bash refinalize.sh "${PREFIX}_*" > "${PREFIX}_refinalize.log" 2>&1 || true
if python3 collect_metrics.py --runs "${PREFIX}_*" --baseline "${POLICY}_default"; then
    grep -v 'data:image' output/_all_metrics/all_metrics.md
else
    # Never hide why: a missing script and missing reports need different fixes.
    echo "!! collect_metrics failed (error above)"
    echo "   reports found: $(ls -d output/${PREFIX}_*/*/eval_out/reports 2>/dev/null | wc -l)"
    echo "   refinalize log: ${PREFIX}_refinalize.log"
fi
}

for lbl in $LABELS; do
    run_label "$lbl"
done

cat <<'EOF'

Read the table on one row per run, for this label only:
  * sign SR must rise vs the `prefix` run — that is the D1 effect;
  * dest rate must NOT collapse — high SR at zero arrival means the car never
    reached the sign, i.e. vacuous compliance, and counts as a negative result;
  * `remap` answers a different question: if it moves the metric, the model does
    read the sign token; if nothing moves it, the token was never learnt and no
    eval-side change can help — only re-dumping and re-finetuning.
EOF
