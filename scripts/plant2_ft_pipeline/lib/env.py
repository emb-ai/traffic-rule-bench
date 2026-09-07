"""Shared paths and runtime env for plant2_ft_pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_TRB_ROOT = "/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench"

# SR006-nfs2 is bind-mounted at different paths on different pods.
_NFS2_MOUNT_CANDIDATES = (
    Path("/mnt/virtual_ai0001053-01202_SR006-nfs2"),
    Path("/home/jovyan/shares/SR006.nfs2"),
)


def nfs2_root() -> Path:
    """Live SR006-nfs2 mount point (first candidate that exists on this pod)."""
    for c in _NFS2_MOUNT_CANDIDATES:
        if c.is_dir():
            return c
    return _NFS2_MOUNT_CANDIDATES[0]


def shepelev() -> Path:
    # No single correct default — real runs always set $SHEPELEV explicitly
    # (often a per-experiment work dir). Falling back to dirname($TRB_ROOT)
    # matches shell/env.sh's own fallback instead of a separate, driftable path.
    env_val = os.environ.get("SHEPELEV")
    return Path(env_val) if env_val else trb_root().parent


def trb_root() -> Path:
    return Path(os.environ.get("TRB_ROOT", _DEFAULT_TRB_ROOT))


def plan_t() -> Path:
    return Path(os.environ.get("PLAN_T", trb_root() / "plant2" / "PlanT"))


def pipeline_dir() -> Path:
    return Path(
        os.environ.get("PIPELINE_DIR", trb_root() / "scripts" / "plant2_ft_pipeline")
    )


def bench_dir() -> Path:
    return Path(
        os.environ.get("BENCH_DIR", trb_root() / "pdd-bench" / "scripts" / "per_sign_bench")
    )


def signs_dir() -> Path:
    return Path(
        os.environ.get(
            "SIGNS_DIR",
            trb_root() / "pdd-bench" / "scripts" / "per_sign_bench" / "plant2_rule_test",
        )
    )


def default_ckpt0() -> Path:
    return Path(
        os.environ.get("CKPT0", shepelev() / "plant2_checkpoints" / "epoch=029_final_1.ckpt")
    )


def shim_path() -> Path:
    env_shim = os.environ.get("SHIM")
    if env_shim:
        return Path(env_shim)
    for candidate in (
        pipeline_dir() / "shims" / "run_lit_finetune.py",
        pipeline_dir() / "plant2_py_shims" / "run_lit_finetune.py",
    ):
        if candidate.is_file():
            return candidate
    return pipeline_dir() / "shims" / "run_lit_finetune.py"


def metrics_root() -> Path:
    return Path(os.environ.get("METRICS_ROOT", shepelev() / "plant2_ft_metrics"))


def resolve_python(explicit: str | None = None) -> Path:
    """Python executable to launch child processes with.

    Explicit argument, then $PYTHON/$PY, then the interpreter running this
    process -- never a guess. The previous version walked a list of candidate
    conda environments and returned whichever existed first; on a pod where
    none of them did it fell through to `which python3` and an eval ran to
    completion against an unrelated interpreter without the model's
    dependencies, producing metrics that looked ordinary and were wrong.
    """
    if explicit:
        chosen = Path(explicit)
        if not os.access(chosen, os.X_OK):
            raise SystemExit(f"--python {chosen} is not an executable")
        return chosen
    env_py = os.environ.get("PYTHON") or os.environ.get("PY")
    if env_py:
        chosen = Path(env_py)
        if not os.access(chosen, os.X_OK):
            raise SystemExit(f"$PYTHON={chosen} is not an executable")
        return chosen
    return Path(sys.executable)


def hydra_escape(value: str | Path) -> str:
    return str(value).replace("=", "\\=")


def setup_eval_thread_env() -> None:
    """Limit BLAS/torch threads for eval workers."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PER_SIGN_COMPLIANT_NPC", "1")
