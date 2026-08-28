"""Shared paths and runtime env for plant2_ft_pipeline."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_DEFAULT_SHEPELEV = "/home/jovyan/shares/SR006.nfs3/shepelev"


def shepelev() -> Path:
    return Path(os.environ.get("SHEPELEV", _DEFAULT_SHEPELEV))


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
    local = _checkout_root()
    if _plant_root_in(local) is not None:
        return local
    return shepelev() / "traffic-rule-bench"


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
