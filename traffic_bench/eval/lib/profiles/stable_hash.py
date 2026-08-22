"""Deterministic hashing helpers for manifest seeds / NPC profile draws."""
from __future__ import annotations

import hashlib


def stable_hash(*parts) -> int:
    """SHA-256-based deterministic hash → 32-bit int.

    Copied from ``sumo_space.sumo_catalog.stable_hash`` so priority_bench does
    not depend on the colleague SUMO catalog for this one helper.

    Used so that the seed for a (scene_id, v_idx, var_idx) triple is reproducible
    across machines and Python versions.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "big")
