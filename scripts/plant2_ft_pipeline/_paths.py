"""Workspace path helpers for plant2_ft_pipeline Python scripts."""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_SHEPELEV = "/home/jovyan/shares/SR006.nfs3/shepelev"


def shepelev() -> Path:
    return Path(os.environ.get("SHEPELEV", _DEFAULT_SHEPELEV))


def trb_root() -> Path:
    return Path(os.environ.get("TRB_ROOT", shepelev() / "traffic-rule-bench"))


def plan_t() -> Path:
    return Path(os.environ.get("PLAN_T", trb_root() / "plant2" / "PlanT"))


def pipeline_dir() -> Path:
    return Path(os.environ.get("PIPELINE_DIR", trb_root() / "scripts" / "plant2_ft_pipeline"))
