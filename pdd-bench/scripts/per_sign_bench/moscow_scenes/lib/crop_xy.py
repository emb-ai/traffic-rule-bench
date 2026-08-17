"""XY-boundary crop helper (netconvert) for dual-path path-union bboxes."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import sys

_PRIORITY = Path(__file__).resolve().parents[2] / "priority_bench"
if str(_PRIORITY) not in sys.path:
    sys.path.insert(0, str(_PRIORITY))

from core.layout.junction_crop import _find_netconvert  # noqa: E402
from core.layout.junction_priority_layout import JunctionLayoutError  # noqa: E402


BBox = Tuple[float, float, float, float]


def crop_net_to_xy_boundary(net_path: Path, bbox_xy: BBox, out_path: Path) -> None:
    """Crop a SUMO net to cartesian boundary ``(xmin, ymin, xmax, ymax)``."""
    xmin, ymin, xmax, ymax = bbox_xy
    if xmax <= xmin or ymax <= ymin:
        raise JunctionLayoutError(f"Degenerate XY boundary: {bbox_xy}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    boundary = f"{xmin},{ymin},{xmax},{ymax}"
    cmd = [
        _find_netconvert(),
        "--sumo-net-file",
        str(net_path),
        "-o",
        str(out_path),
        "--keep-edges.in-boundary",
        boundary,
        "--geometry.remove",
        "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise JunctionLayoutError(
            f"netconvert XY-boundary crop failed for {net_path}:\n"
            f"{result.stderr or result.stdout}"
        )
    if not out_path.is_file():
        raise JunctionLayoutError(f"netconvert did not write {out_path}")

    tree = ET.parse(out_path)
    root = tree.getroot()
    # Refresh location convBoundary from remaining geometry when possible.
    xs: list[float] = []
    ys: list[float] = []
    for lane in root.findall("./edge/lane"):
        for token in (lane.get("shape") or "").split():
            if "," not in token:
                continue
            x_s, y_s = token.split(",", 1)
            xs.append(float(x_s))
            ys.append(float(y_s))
    loc = root.find("location")
    if loc is not None and xs and ys:
        loc.set("convBoundary", f"{min(xs)},{min(ys)},{max(xs)},{max(ys)}")
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)
