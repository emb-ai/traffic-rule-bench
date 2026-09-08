"""Physical-map identity of an episode.

A manifest row names its net (``net_path`` = ``<scene_dir>/map.net.xml``) and
expands it into augmented variants whose ``scene_id`` carries the variant and
the world-axis cell as a suffix:

    seg_1067603714_v0                        nominal
    seg_1067603714_v0_z60a60n2_td50_sv1_v0   manifest variant 0, world cell
    seg_1067603714_v2_z100a100n1_td50_sv2_v2 manifest variant 2, world cell
    seg_100833537_0_l0_v3                    spawn lane 0, manifest variant 3

All of these are the same map. Metrics are averaged per map first (see
``aggregate.aggregate_by_map``), so the CSV builder stamps every episode with
one stable ``map_id`` and the aggregator groups on it. The manifest's
``net_path`` is authoritative; without a manifest the suffix is stripped from
the scene id. Junction scenes (``junc_…``) carry no ``_v<k>`` suffix — their
variants differ by seed only — so their id is already the map.
"""
from __future__ import annotations

from pathlib import PurePosixPath
import re

# ``<base>[_l<lane>]_v<k>`` optionally followed by the world-axis cell
# (``_rl90_td50_sv1_v2``, ``_z60a60n2_td50_sv1_v0``). ``_sv1`` does not match:
# the variant marker is ``_v`` directly after an underscore.
_VARIANT_SUFFIX_RE = re.compile(r"(?:_l\d+)?_v\d+(?:_.*)?$")


def base_scene_id(scene_id: str) -> str:
    """Scene id with its manifest-variant / world-axis suffix stripped."""
    sid = str(scene_id or "")
    return _VARIANT_SUFFIX_RE.sub("", sid, count=1) or sid


def map_id_from_manifest_row(row: dict | None) -> str | None:
    """Map id of a manifest row: the scene directory named by ``net_path``."""
    if not row:
        return None
    net_path = row.get("net_path")
    if not net_path:
        return None
    p = PurePosixPath(str(net_path))
    # "<scene_dir>/map.net.xml" -> scene_dir; a bare directory name stays as is.
    name = p.parent.name if p.suffix else p.name
    return name or None


def map_id_for(scene_id: str, manifest_row: dict | None = None) -> str:
    """``net_path`` directory when the manifest row is known, else the scene
    id without its suffix."""
    return map_id_from_manifest_row(manifest_row) or base_scene_id(scene_id)
