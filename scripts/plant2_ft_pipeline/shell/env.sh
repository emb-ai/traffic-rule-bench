# Shared env for PlanT2 spatial FT pipeline scripts.
# Source from pipeline root:
#   source "$PIPELINE_DIR/_env.sh"
# or from shell/:
#   source "$PIPELINE_DIR/shell/env.sh"
#
# Override before sourcing:
#   SHEPELEV, PYTHON, CKPT0, METRICS_ROOT, DS_LOCAL, …

: "${PIPELINE_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${TRB_ROOT:=$(cd "$PIPELINE_DIR/../.." && pwd)}"
: "${SHEPELEV:=$(dirname "$TRB_ROOT")}"
: "${CT:=$PIPELINE_DIR}"
: "${REPO:=$TRB_ROOT}"

export PIPELINE_DIR TRB_ROOT SHEPELEV CT REPO
export INSPECT_BOXES="$PIPELINE_DIR/tools/inspect_boxes.py"
export PLAN_T="${PLAN_T:-$TRB_ROOT/plant2/PlanT}"
export PLANT="${PLANT:-$TRB_ROOT/plant2}"
export CKPT0="${CKPT0:-$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt}"
if [[ -z "${SHIM:-}" ]]; then
  if [[ -f "$PIPELINE_DIR/shims/run_lit_finetune.py" ]]; then
    export SHIM="$PIPELINE_DIR/shims/run_lit_finetune.py"
  else
    export SHIM="$PIPELINE_DIR/plant2_py_shims/run_lit_finetune.py"
  fi
fi
export SIGNS_DIR="${SIGNS_DIR:-$TRB_ROOT/pdd-bench/scripts/per_sign_bench/plant2_rule_test}"
export BENCH_DIR="${BENCH_DIR:-$TRB_ROOT/pdd-bench/scripts/per_sign_bench}"
export METRICS_ROOT="${METRICS_ROOT:-$SHEPELEV/plant2_ft_metrics}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python" ]]; then
    export PY="$SHEPELEV/conda_envs/arbelyaev-sdc/bin/python"
  elif [[ -x /home/user/conda/envs/zinkovich-sdc/bin/python ]]; then
    export PY=/home/user/conda/envs/zinkovich-sdc/bin/python
  else
    export PY="${PY:-python3}"
  fi
else
  export PY="$PYTHON"
fi
