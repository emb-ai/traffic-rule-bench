"""Path bootstrap shared by the bench/* modules.

Mirrors the sys.path setup in run_benchmark.py so env_builders/policy_factory can
resolve `envs.*` and the CaRL/plant2 adapters whether imported from
run_benchmark or standalone.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_pdd_bench_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "envs").is_dir() and (parent / "traffic_signs").is_dir():
            return parent
    raise RuntimeError("Could not locate pdd-bench root")


_THIS = Path(__file__).resolve()
PDD_BENCH_DIR = _find_pdd_bench_root(_THIS)
SDC_ROOT = PDD_BENCH_DIR.parent
PLANT2_REPO = Path(os.environ["PLANT2_REPO"]) if os.environ.get("PLANT2_REPO") else SDC_ROOT / "plant2"
METADRIVE_DIR = SDC_ROOT / "metadrive"

for _p in (PDD_BENCH_DIR, METADRIVE_DIR, PDD_BENCH_DIR / "scripts" / "per_sign_bench"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

CARL_NUPLAN_DIR = SDC_ROOT / "CaRL" / "nuPlan"
if CARL_NUPLAN_DIR.exists() and str(CARL_NUPLAN_DIR) not in sys.path:
    sys.path.insert(0, str(CARL_NUPLAN_DIR))
