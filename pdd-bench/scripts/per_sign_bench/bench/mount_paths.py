"""Resolve a recorded absolute path when the volume is mounted elsewhere.

Expert rows, sidecars and scene catalogs store ABSOLUTE paths. The same NFS
volume is reachable under more than one mount point here — a node may see
``/mnt/virtual_..._SR006-nfs2/...`` or ``/home/jovyan/shares/SR006.nfs2/...``
for the identical bytes — so a file list written on one node reads as 79%
missing on another, and the replay fails with FileNotFoundError on paths that
are in fact present.

Only equivalent roots are swapped: this rewrites the mount prefix and nothing
else, so a genuinely absent file stays absent instead of silently resolving to
a different file. Extra pairs can be supplied through ``PDD_MOUNT_ALIASES`` as
``old=new`` entries separated by ``;`` (both directions are tried).
"""
from __future__ import annotations

import os
from pathlib import Path

# Equivalent roots for the same volume. Tried in both directions.
_DEFAULT_ALIASES: tuple[tuple[str, str], ...] = (
    ("/mnt/virtual_ai0001053-01202_SR006-nfs2", "/home/jovyan/shares/SR006.nfs2"),
    ("/mnt/virtual_ai0001053-01202_SR006-nfs3", "/home/jovyan/shares/SR006.nfs3"),
)


def _aliases() -> tuple[tuple[str, str], ...]:
    extra = os.environ.get("PDD_MOUNT_ALIASES", "").strip()
    if not extra:
        return _DEFAULT_ALIASES
    pairs = []
    for item in extra.split(";"):
        if "=" in item:
            a, b = item.split("=", 1)
            if a.strip() and b.strip():
                pairs.append((a.strip(), b.strip()))
    return _DEFAULT_ALIASES + tuple(pairs)


def candidates(path: str | os.PathLike) -> list[Path]:
    """Every equivalent spelling of *path*, the given one first."""
    p = str(path)
    out = [p]
    for a, b in _aliases():
        if p.startswith(a):
            out.append(b + p[len(a):])
        elif p.startswith(b):
            out.append(a + p[len(b):])
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(Path(c))
    return uniq


def resolve_shared_path(path: str | os.PathLike) -> Path:
    """First spelling of *path* that exists, else the path as given.

    Returning the original on failure keeps the caller's own error message and
    its path intact, which is what a reader of the log needs to see.
    """
    for c in candidates(path):
        if c.exists():
            return c
    return Path(path)
