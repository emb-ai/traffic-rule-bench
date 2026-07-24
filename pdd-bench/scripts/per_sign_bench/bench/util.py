"""Small shared helpers for the per-sign benchmark runner."""
from __future__ import annotations

import random

import numpy as np


def _slug_to_code(slug: str) -> str:
    return slug.replace("_", ".")


def _row_seed(row: dict, default: int = 0) -> int:
    """Deterministic per-scene seed from a manifest row."""
    return int(row.get("seed") or row.get("deterministic_seed") or default)


def _row_sign_code(row: dict):
    """Resolve a row's sign code, preferring the collect-tagged `_sign_code`."""
    return (row.get("_sign_code") or row.get("sign_code")
            or row.get("pdd_code") or row.get("sign_type"))


def _seed_everything(seed: int) -> None:
    """Seed numpy/random (+ torch if importable) for a reproducible episode."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _unwrap_base_env(env):
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def _error_result(row: dict, error, backend=None) -> dict:
    """Uniform `ok=False` episode record (used on setup failure and on exception)."""
    return {
        "ok": False,
        "error": str(error),
        "backend": backend if backend is not None else row.get("_backend"),
        "scene_id": row.get("scene_id"),
        "sign_type": _row_sign_code(row),
        "seed": _row_seed(row, -1),
    }
