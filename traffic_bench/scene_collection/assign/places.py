"""Physical place identity for tiered assign (junction_id / osm_way_id)."""

from __future__ import annotations

import re
from typing import Mapping, Optional

_JUNC_RE = re.compile(r"^junc_(.+)$")
_SEG_RE = re.compile(r"^seg_(\d+)(?:_\d+)?$")
_RB_RE = re.compile(r"^rb_")


def place_id_from_junction(junction_id: str) -> str:
    jid = str(junction_id or "").strip()
    if not jid:
        raise ValueError("empty junction_id")
    return f"junction:{jid}"


def place_id_from_way(osm_way_id: str) -> str:
    way = str(osm_way_id or "").strip()
    if not way:
        raise ValueError("empty osm_way_id")
    return f"way:{way}"


def place_id_from_meta(meta: Mapping, *, scene_id: str = "", crop_kind: str = "") -> Optional[str]:
    jid = meta.get("junction_id")
    way = meta.get("osm_way_id") or meta.get("way_id")
    crop = str(meta.get("crop_kind") or crop_kind or "")
    sid = str(scene_id or meta.get("scene_id") or "")

    if crop == "segment" or sid.startswith("seg_"):
        if way is not None and str(way).strip():
            return place_id_from_way(str(way))
        m = _SEG_RE.match(sid)
        if m:
            return place_id_from_way(m.group(1))
    if jid is not None and str(jid).strip():
        return place_id_from_junction(str(jid))
    if way is not None and str(way).strip():
        return place_id_from_way(str(way))
    if _JUNC_RE.match(sid):
        return place_id_from_junction(_JUNC_RE.match(sid).group(1))  # type: ignore[union-attr]
    if _RB_RE.match(sid):
        return f"scene:{sid}"
    return None


def place_id_for_junction_scene(scene_id: str, scene_to_junc: Mapping[str, str]) -> str:
    sid = str(scene_id or "").strip()
    jid = scene_to_junc.get(sid)
    if jid:
        return place_id_from_junction(jid)
    m = _JUNC_RE.match(sid)
    if m:
        return place_id_from_junction(m.group(1))
    if _RB_RE.match(sid):
        return f"scene:{sid}"
    return f"scene:{sid or 'unknown'}"
