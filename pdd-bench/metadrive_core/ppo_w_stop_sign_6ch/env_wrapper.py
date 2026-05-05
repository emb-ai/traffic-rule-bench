"""
Environment with STOP sign 

"""
import sys
from pathlib import Path


def _find_sdc_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "pdd-bench").is_dir() and (parent / "metadrive").is_dir():
            return parent
    raise RuntimeError("Could not locate SDC root (expected pdd-bench and metadrive)")


FILE_PATH = Path(__file__).resolve()
SDC_ROOT = _find_sdc_root(FILE_PATH)
PDD_BENCH_DIR = SDC_ROOT / "pdd-bench"
METADRIVE_DIR = SDC_ROOT / "metadrive"

for path in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from metadrive_core.obs.top_down_obs_6channel import TopDownMultiChannel6Channels
from metadrive_core.env_wrappers import build_topdown_env


TopDownMetaDriveWithStopSigns = build_topdown_env(TopDownMultiChannel6Channels)

__all__ = ["TopDownMetaDriveWithStopSigns"]
        
        
