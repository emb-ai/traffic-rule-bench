#!/usr/bin/env python3
"""Bootstrap PlanT lit_finetune with broken flash_attn disabled.

zinkovich-sdc is gone on this node; base conda has an ABI-mismatched
flash_attn (root-owned, uninstallable). Patch transformers availability
checks so Bert loads via SDPA/eager instead of importing flash_attn_2_cuda.
"""
from __future__ import annotations

import os
import runpy
import sys

_SHIM_DIR = os.path.dirname(os.path.abspath(__file__))
if _SHIM_DIR not in sys.path:
    sys.path.insert(0, _SHIM_DIR)

from disable_flash_attn import apply  # noqa: E402

apply()

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

lit = os.path.join(cwd, "lit_finetune.py")
if not os.path.isfile(lit):
    raise FileNotFoundError(f"expected lit_finetune.py in cwd={cwd!r}")

sys.argv[0] = lit
runpy.run_path(lit, run_name="__main__")
