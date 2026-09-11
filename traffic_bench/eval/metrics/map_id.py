"""Physical map of an episode: the directory of its manifest row's net_path.

Scene ids are not parsed: their format differs by expander version.
"""
from __future__ import annotations

from pathlib import PurePosixPath

# ci.map_id in cumulative.json; report / plot refuse files without it.
MAP_ID_SOURCE = "manifest net_path directory"


class ManifestError(ValueError):
    """The manifest is missing, malformed, or does not give an episode's map."""


def map_id_from_manifest_row(row: dict) -> str:
    """``seg_1/map.net.xml`` -> ``seg_1``."""
    net_path = row.get("net_path")
    where = f"scene_id={row.get('scene_id')!r}"
    if not isinstance(net_path, str) or not net_path.strip():
        raise ManifestError(f"manifest row {where} has no net_path; its map cannot be determined")
    p = PurePosixPath(net_path.strip())
    if not p.suffix or not p.parent.name:
        raise ManifestError(f"net_path {net_path!r} ({where}) is not <scene_dir>/<net file>")
    return p.parent.name
