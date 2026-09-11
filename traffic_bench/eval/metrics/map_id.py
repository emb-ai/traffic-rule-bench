"""Physical-map identity of an episode: from the manifest, and only from it.

A manifest row names the net its episode runs on: ``net_path`` =
``<scene_dir>/map.net.xml``. Every augmented row of one catalog scene points
at that net whatever its ``scene_id`` looks like (the expanders have written
``seg_x_l0_v3_rl90_td50_v2``, ``junc_x_rl90_td25_sv0_v0``, ``seg_x_l0_td2_v0``,
``dual_T_x_rl100`` and more), so the map is the directory of ``net_path``.

There is no fallback. Scene-id formats change with the expanders and cannot
be parsed reliably, so a row without a usable ``net_path`` raises
``ManifestError``, and so does every consumer that meets an episode without a
manifest map (``metrics csv``, ``metrics aggregate``, ``metrics report``,
``metrics plot --agg map``).
"""
from __future__ import annotations

from pathlib import PurePosixPath

# Written into cumulative.json as ``ci.map_id``: the per-map blocks are keyed
# on this map identity. report.py and plot_benchmark.py refuse files without it.
MAP_ID_SOURCE = "manifest net_path directory"


class ManifestError(ValueError):
    """The manifest is missing, malformed, or does not give an episode's map."""


def map_id_from_manifest_row(row: dict) -> str:
    """Directory of the row's ``net_path``: ``seg_1/map.net.xml`` -> ``seg_1``."""
    net_path = row.get("net_path")
    where = f"scene_id={row.get('scene_id')!r}"
    if not isinstance(net_path, str) or not net_path.strip():
        raise ManifestError(
            f"manifest row {where} has no net_path; the map of its episodes "
            "cannot be determined")
    p = PurePosixPath(net_path.strip())
    if not p.suffix or not p.parent.name:
        raise ManifestError(
            f"net_path {net_path!r} ({where}) does not name a net file inside a "
            "scene directory (<scene_dir>/map.net.xml)")
    return p.parent.name
