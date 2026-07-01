"""SUMO roundabout fingerprints and a scenes-folder registry for deduplication."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .junction_priority_layout import JunctionLayoutError
from .roundabout_topology import SumoRoundabout, resolve_sumo_roundabout
from .sumo_utils import is_roundabout_scene_meta, load_scene_meta, resolve_net_file

REGISTRY_FILENAME = "roundabout_fingerprints.json"
REGISTRY_VERSION = 1
RESERVED_SCENE_DIRS = frozenset({"core"})


def fingerprint_from_sumo_roundabout(rb: SumoRoundabout) -> str:
    """Stable global key from SUMO ``<roundabout nodes=\"...\">`` OSM node ids."""
    return ",".join(sorted(rb.node_ids))


def sumo_roundabout_record(rb: SumoRoundabout) -> dict[str, Any]:
    return {
        "sumo_roundabout_fingerprint": fingerprint_from_sumo_roundabout(rb),
        "sumo_roundabout_nodes": sorted(rb.node_ids),
        "sumo_roundabout_ring_edges": sorted(rb.ring_edge_ids),
    }


def fingerprint_for_net(
    net_path: Path,
    *,
    sign_edge_id: Optional[str] = None,
) -> tuple[str, SumoRoundabout]:
    rb = resolve_sumo_roundabout(net_path, sign_edge_id=sign_edge_id)
    return fingerprint_from_sumo_roundabout(rb), rb


def registry_path_for_scenes_root(scenes_root: Path) -> Path:
    return scenes_root.expanduser().resolve() / REGISTRY_FILENAME


class RoundaboutFingerprintRegistry:
    """Maps SUMO roundabout fingerprints to scene folders under ``scenes/``."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.fingerprints: Dict[str, dict[str, Any]] = {}
        self.by_scene: Dict[str, str] = {}
        self.load()

    @classmethod
    def for_scenes_root(cls, scenes_root: Path) -> "RoundaboutFingerprintRegistry":
        return cls(registry_path_for_scenes_root(scenes_root))

    def load(self) -> None:
        self.fingerprints = {}
        self.by_scene = {}
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        fps = data.get("fingerprints")
        if isinstance(fps, dict):
            self.fingerprints = {str(k): dict(v) for k, v in fps.items() if isinstance(v, dict)}
        by_scene = data.get("by_scene")
        if isinstance(by_scene, dict):
            self.by_scene = {str(k): str(v) for k, v in by_scene.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REGISTRY_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprints": self.fingerprints,
            "by_scene": self.by_scene,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def owner_of(self, fingerprint: str) -> Optional[dict[str, Any]]:
        return self.fingerprints.get(fingerprint)

    def duplicate_owner(
        self,
        fingerprint: str,
        *,
        scene_name: str,
        core_scene_name: Optional[str] = None,
        one_per_core: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Return registry entry if another scene already owns this fingerprint.

        Cropping ``sign_foo_rb`` from core ``sign_foo`` is allowed when the registry
        only contains the matching ``kind=core`` entry from import. Multiple catalog
        signs on the same SUMO roundabout are also allowed until a cropped scene
        exists for that fingerprint.

        When ``one_per_core=False`` (per-spoke crops), only block if ``scene_name``
        is already registered — the same physical ring may have many spoke folders.
        """
        if scene_name in self.by_scene:
            owner = self.owner_of(self.by_scene[scene_name])
            if owner is not None and owner.get("scene_name") != scene_name:
                return owner
            return None

        if not one_per_core:
            return None

        owner = self.owner_of(fingerprint)
        if owner is None:
            return None
        if owner.get("scene_name") == scene_name:
            return None
        if core_scene_name:
            if owner.get("core_scene_name") == core_scene_name:
                return None
            if owner.get("kind") == "core" and owner.get("scene_name") == core_scene_name:
                return None
        if owner.get("kind") == "core":
            return None
        return owner

    def upsert(
        self,
        fingerprint: str,
        *,
        scene_name: str,
        core_scene_name: str,
        kind: str,
        sign_id: Optional[int] = None,
        sumo_roundabout_nodes: Optional[Iterable[str]] = None,
        sumo_roundabout_ring_edges: Optional[Iterable[str]] = None,
    ) -> None:
        previous_scene = self.by_scene.get(scene_name)
        if previous_scene and previous_scene != fingerprint:
            if self.fingerprints.get(previous_scene, {}).get("scene_name") == scene_name:
                self.fingerprints.pop(previous_scene, None)

        entry = {
            "scene_name": scene_name,
            "core_scene_name": core_scene_name,
            "kind": kind,
            "sign_id": sign_id,
            "sumo_roundabout_nodes": sorted(sumo_roundabout_nodes or []),
            "sumo_roundabout_ring_edges": sorted(sumo_roundabout_ring_edges or []),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.fingerprints[fingerprint] = entry
        self.by_scene[scene_name] = fingerprint

    def remove_scene(self, scene_name: str) -> None:
        fingerprint = self.by_scene.pop(scene_name, None)
        if fingerprint and self.fingerprints.get(fingerprint, {}).get("scene_name") == scene_name:
            self.fingerprints.pop(fingerprint, None)


def fingerprint_from_scene_dir(scene_dir: Path) -> Optional[str]:
    """Read or compute fingerprint for an existing cropped/core scene folder."""
    scene_dir = scene_dir.resolve()
    meta_path = scene_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = load_scene_meta(scene_dir)
    except (FileNotFoundError, ValueError):
        return None

    stored = meta.get("sumo_roundabout_fingerprint")
    if isinstance(stored, str) and stored:
        return stored

    net_path = scene_dir / resolve_net_file(scene_dir, meta)
    if not net_path.is_file():
        return None
    try:
        fp, _ = fingerprint_for_net(
            net_path,
            sign_edge_id=meta.get("road_id") or meta.get("catalog_sign_road_id"),
        )
    except JunctionLayoutError:
        return None
    return fp


def rebuild_registry_from_scenes(
    scenes_root: Path,
    *,
    include_core: bool = True,
) -> RoundaboutFingerprintRegistry:
    """Rebuild ``roundabout_fingerprints.json`` by scanning scene folders."""
    scenes_root = scenes_root.expanduser().resolve()
    registry = RoundaboutFingerprintRegistry.for_scenes_root(scenes_root)
    registry.fingerprints = {}
    registry.by_scene = {}

    def _register_dir(scene_dir: Path, *, kind: str) -> None:
        if not (scene_dir / "meta.json").is_file():
            return
        try:
            meta = load_scene_meta(scene_dir)
        except (FileNotFoundError, ValueError):
            return
        if kind == "cropped" and not is_roundabout_scene_meta(meta):
            return
        fp = fingerprint_from_scene_dir(scene_dir)
        if not fp:
            return
        nodes = meta.get("sumo_roundabout_nodes")
        ring_edges = meta.get("sumo_roundabout_ring_edges")
        if not nodes or not ring_edges:
            try:
                net_path = scene_dir / resolve_net_file(scene_dir, meta)
                _, rb = fingerprint_for_net(
                    net_path,
                    sign_edge_id=meta.get("road_id") or meta.get("catalog_sign_road_id"),
                )
                nodes = sorted(rb.node_ids)
                ring_edges = sorted(rb.ring_edge_ids)
            except JunctionLayoutError:
                nodes = nodes or []
                ring_edges = ring_edges or []
        registry.upsert(
            fp,
            scene_name=meta.get("scene_name", scene_dir.name),
            core_scene_name=meta.get("core_scene_name", scene_dir.name),
            kind=kind,
            sign_id=meta.get("sign_id"),
            sumo_roundabout_nodes=nodes,
            sumo_roundabout_ring_edges=ring_edges,
        )

    if include_core:
        core_root = scenes_root / "core"
        if core_root.is_dir():
            for entry in sorted(core_root.iterdir()):
                if entry.is_dir():
                    _register_dir(entry, kind="core")

    if scenes_root.is_dir():
        for entry in sorted(scenes_root.iterdir()):
            if not entry.is_dir() or entry.name in RESERVED_SCENE_DIRS:
                continue
            if entry.name == REGISTRY_FILENAME:
                continue
            _register_dir(entry, kind="cropped")

    registry.save()
    return registry
