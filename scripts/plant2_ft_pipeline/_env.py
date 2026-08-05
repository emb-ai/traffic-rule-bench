"""Shared paths and runtime env for plant2_ft_pipeline Python scripts."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_DEFAULT_SHEPELEV = "/home/jovyan/shares/SR006.nfs3/shepelev"


def shepelev() -> Path:
    return Path(os.environ.get("SHEPELEV", _DEFAULT_SHEPELEV))


def trb_root() -> Path:
    return Path(os.environ.get("TRB_ROOT", shepelev() / "traffic-rule-bench"))


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
    return Path(
        os.environ.get("SHIM", pipeline_dir() / "plant2_py_shims" / "run_lit_finetune.py")
    )


def metrics_root() -> Path:
    return Path(os.environ.get("METRICS_ROOT", shepelev() / "plant2_ft_metrics"))


def resolve_python(explicit: str | None = None) -> Path:
    """Pick python executable (arbelyaev-sdc preferred)."""
    if explicit:
        return Path(explicit)
    env_py = os.environ.get("PYTHON") or os.environ.get("PY")
    if env_py:
        return Path(env_py)
    candidates = [
        shepelev() / "conda_envs" / "arbelyaev-sdc" / "bin" / "python",
        Path("/home/user/conda/envs/zinkovich-sdc/bin/python"),
        Path("/home/user/conda/bin/python"),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    found = shutil.which("python3") or shutil.which("python")
    if found:
        return Path(found)
    return Path("python3")


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
