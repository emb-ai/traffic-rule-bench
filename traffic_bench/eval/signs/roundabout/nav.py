"""Roundabout background-traffic helpers (circular ring vs spokes)."""
from __future__ import annotations

from typing import List


def ring_edge_ids_from_roundabout_layout(layout: dict | None) -> List[str]:
    """Circular roadway edges only — not spoke in/out arms.

    For ``mode=roundabout`` layouts, ``main_edge_ids`` are the SUMO
    ``<roundabout edges=...>`` ring segments from
    ``build_roundabout_layout`` / moscow meta. Spoke approaches live in
    ``secondary_edge_ids`` and must not be treated as the ring.
    """
    if not isinstance(layout, dict):
        return []
    if layout.get("mode") != "roundabout" and layout.get("shape") != "O":
        return []

    main = layout.get("main_edge_ids") or []
    if main:
        return sorted({str(e) for e in main if e})

    # Fallback: arms marked road_class=main (ring arms from topology).
    ring: set[str] = set()
    for arm in layout.get("arms") or []:
        if arm.get("road_class") != "main":
            continue
        eid = arm.get("edge_id")
        if eid:
            ring.add(str(eid))
    return sorted(ring)
