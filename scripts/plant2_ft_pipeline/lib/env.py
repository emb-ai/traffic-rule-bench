"""Shared paths and runtime env for plant2_ft_pipeline."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

def _checkout_root() -> Path:
    """Repo root of the checkout this file belongs to: <root>/scripts/plant2_ft_pipeline/lib/env.py."""
    return Path(__file__).resolve().parents[3]


# PlanT lives under third_party/ in this repository and at the top level in the
# legacy checkout. Both layouts are recognised so one pipeline serves either.
_PLANT_SUBDIRS = (Path("third_party") / "plant2", Path("plant2"))


def _plant_root_in(root: Path) -> Path | None:
    for sub in _PLANT_SUBDIRS:
        if (root / sub / "PlanT").is_dir():
            return root / sub
    return None


def trb_root() -> Path:
    env_root = os.environ.get("TRB_ROOT")
    if env_root:
        return Path(env_root)
    # This checkout is the repository; the legacy pipeline resolved it from a
    # personal cluster share instead, which made every default unusable to
    # anyone else and silently pointed at someone else's data when it existed.
    return _checkout_root()


def plan_t() -> Path:
    env_plant = os.environ.get("PLAN_T")
    if env_plant:
        return Path(env_plant)
    root = trb_root()
    found = _plant_root_in(root)
    if found is not None:
        return found / "PlanT"
    # Neither layout is present: name the one this repository ships, so the
    # error points at a path that is supposed to exist.
    return root / "third_party" / "plant2" / "PlanT"


def pipeline_dir() -> Path:
    return Path(
        os.environ.get("PIPELINE_DIR", trb_root() / "scripts" / "plant2_ft_pipeline")
    )


# Closed-loop eval moved into the `traffic_bench` package here; the legacy
# `pdd-bench/scripts/per_sign_bench` tree is kept as the fallback so the
# pipeline still runs against an old checkout. Only the eval half of the
# pipeline reads these -- training uses plan_t() alone.
def bench_dir() -> Path:
    env_bench = os.environ.get("BENCH_DIR")
    if env_bench:
        return Path(env_bench)
    root = trb_root()
    local = root / "traffic_bench" / "eval"
    if local.is_dir():
        return local
    return root / "pdd-bench" / "scripts" / "per_sign_bench"


def signs_dir() -> Path:
    return Path(os.environ.get("SIGNS_DIR", bench_dir() / "plant2_rule_test"))


def default_ckpt0() -> Path:
    return Path(
        os.environ.get("CKPT0")
        or (trb_root() / "checkpoints" / "plant2_pretrain" / "epoch=029_final_3.ckpt")
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
    return Path(os.environ.get("METRICS_ROOT") or (trb_root() / "data" / "plant2_ft_metrics"))


def resolve_python(explicit: str | None = None) -> Path:
    """The interpreter to launch training with: explicit, PYTHON/PY, or this one.

    The legacy list named conda environments on one cluster's shares, so on any
    other machine it fell through to whatever `python3` was first on PATH --
    which is rarely the environment the caller meant.
    """
    if explicit:
        return Path(explicit)
    env_py = os.environ.get("PYTHON") or os.environ.get("PY")
    if env_py:
        return Path(env_py)
    if sys.executable:
        return Path(sys.executable)
    found = shutil.which("python3") or shutil.which("python")
    return Path(found) if found else Path("python3")


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
