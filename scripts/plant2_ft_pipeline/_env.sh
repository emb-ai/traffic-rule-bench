# Shared env for PlanT2 spatial FT pipeline scripts.
# Source from any script in this directory:
#   PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # shellcheck disable=SC1091
#   source "$PIPELINE_DIR/_env.sh"
#
# Override before sourcing:
#   SHEPELEV, PYTHON, CKPT0, METRICS_ROOT, DS_LOCAL, …

: "${PIPELINE_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${TRB_ROOT:=$(cd "$PIPELINE_DIR/../.." && pwd)}"
: "${SHEPELEV:=$(dirname "$TRB_ROOT")}"
: "${CT:=$PIPELINE_DIR}"
: "${REPO:=$TRB_ROOT}"

export PIPELINE_DIR TRB_ROOT SHEPELEV CT REPO
export PLAN_T="${PLAN_T:-$TRB_ROOT/plant2/PlanT}"
export PLANT="${PLANT:-$TRB_ROOT/plant2}"
export CKPT0="${CKPT0:-$SHEPELEV/plant2_checkpoints/epoch=029_final_1.ckpt}"
export SHIM="${SHIM:-$PIPELINE_DIR/plant2_py_shims/run_lit_finetune.py}"
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
