"""Shared SUMO utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_NET_FILE = "map.net.xml"


def resolve_net_file(scene_dir: Path, meta: dict) -> str:
    """Resolve SUMO net filename (neutral map.net.xml, with legacy fallback).
    
    Args:
        scene_dir: Path to the scene directory.
        meta: Scene metadata dict (from meta.json).
        
    Returns:
        Name of the .net.xml file in the scene directory.
        
    Raises:
        FileNotFoundError: If no .net.xml file is found.
    """
    net_file = meta.get("net_file", DEFAULT_NET_FILE)
    if (scene_dir / net_file).exists():
        return net_file

    net_files = sorted(scene_dir.glob("*.net.xml"))
    if net_files:
        return net_files[0].name

    raise FileNotFoundError(
        f"No .net.xml file found in {scene_dir} "
        f"(expected {DEFAULT_NET_FILE} or net_file in meta.json)"
    )


def load_scene_meta(scene_dir: Path) -> dict:
    """Load meta.json from a scene directory.
    
    Args:
        scene_dir: Path to the scene directory.
        
    Returns:
        Dictionary with scene metadata.
        
    Raises:
        FileNotFoundError: If meta.json is not found.
    """
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {scene_dir}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_scene_dir(scenes_dir: Path, scene_name: str) -> Path:
    """Resolve and validate a scene directory path.
    
    Args:
        scenes_dir: Root directory containing scenes.
        scene_name: Name of the scene subdirectory.
        
    Returns:
        Path to the scene directory.
        
    Raises:
        FileNotFoundError: If the scene directory doesn't exist.
    """
    scene_dir = scenes_dir / scene_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {scene_dir}")
    return scene_dir


def find_first_edge_id(net_path: Path) -> Optional[str]:
    """Find the first non-internal edge ID from a SUMO net.xml file.
    
    Args:
        net_path: Path to the .net.xml file.
        
    Returns:
        First non-internal edge ID, or None if not found.
    """
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(net_path)
        root = tree.getroot()
        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            if not edge_id.startswith(":"):
                return edge_id
    except Exception:
        pass
    return None
