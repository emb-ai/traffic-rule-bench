#!/usr/bin/env python3
"""Shared geo helpers for collect (UTM lat/lon)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Tuple


@lru_cache(maxsize=4)
def _net_proj(net_path: str) -> tuple:
    """Return (netOffset_x, netOffset_y, Transformer utm→wgs84) for a SUMO net."""
    from pyproj import Transformer

    root = ET.parse(net_path).getroot()
    loc = root.find("location")
    if loc is None:
        raise ValueError(f"No <location> in {net_path}")
    offset = tuple(float(v) for v in (loc.get("netOffset") or "0,0").split(","))
    if len(offset) != 2:
        raise ValueError(f"Bad netOffset in {net_path}")
    proj = loc.get("projParameter") or "+proj=utm +zone=37 +datum=WGS84"
    # netOffset was subtracted from projected coords → projected = net_xy - netOffset
    transformer = Transformer.from_pipeline(
        f"+proj=pipeline +step {proj} +inv +step +proj=latlong +datum=WGS84"
    )
    # Prefer CRS-based transform when UTM zone 37
    try:
        transformer = Transformer.from_crs("EPSG:32637", "EPSG:4326", always_xy=True)
    except Exception:
        pass
    return float(offset[0]), float(offset[1]), transformer


def net_xy_to_latlon_proj(net_path: Path, x: float, y: float) -> Tuple[float, float]:
    """Accurate WGS84 lat/lon from SUMO net XY using UTM + netOffset."""
    ox, oy, transformer = _net_proj(str(net_path.resolve()))
    # SUMO: network = projected - netOffset  ⇒  projected = network - netOffset
    # (netOffset is typically large negative for UTM, so this adds).
    easting = x - ox
    northing = y - oy
    lon, lat = transformer.transform(easting, northing)
    return float(lat), float(lon)
