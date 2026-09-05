"""Build place-id catalogs from ``data/scenes/<sign>/moscow_pool.json``.

**Place identity** (map / geography), not scenario augmentation:

- junction / dual_path / roundabout crops → ``junction:<junction_id>``
- segment crops → ``way:<osm_way_id>``

Families for roll-ups come from assign taxonomy:
behavioral family (e.g. ``direction_control``) and semantic group
(``priority`` / ``speed`` / ``obstacle`` / ``reroute``).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from traffic_bench.eval.sign_registry import list_profiles
from traffic_bench.scene_collection.assign.taxonomy import (
    behavioral_family as beh_of_pdd,
    semantic_group as sem_of_pdd,
)
from traffic_bench.scene_collection.paths import DATA_SCENES

_JUNC_RE = re.compile(r"^junc_(.+)$")
_SEG_RE = re.compile(r"^seg_(\d+)(?:_\d+)?$")
_DUAL_RE = re.compile(r"^dual_")
_RB_RE = re.compile(r"^rb_")

# Eval folder name → PDD code (from sign registry). Both the eval folder
# (``main_road``) and the HF / registry id (``main``) resolve to the same code.
_PDD_BY_FOLDER: Dict[str, str] = {}
for _profile in list_profiles():
    _PDD_BY_FOLDER[_profile.data_subdir] = _profile.pdd_code
    _PDD_BY_FOLDER.setdefault(_profile.id, _profile.pdd_code)

# Behavioral family of each eval sign folder (reviewer roll-ups).
SIGN_FAMILY: Dict[str, str] = {
    folder: beh_of_pdd(pdd) for folder, pdd in _PDD_BY_FOLDER.items()
}
SIGN_SEMANTIC: Dict[str, str] = {
    folder: sem_of_pdd(pdd) for folder, pdd in _PDD_BY_FOLDER.items()
}


@dataclass(frozen=True)
class SceneRecord:
    sign: str
    scene_id: str
    split: str
    place_id: str
    crop_kind: str
    path: Optional[str] = None


@dataclass
class OverlapCatalog:
    records: List[SceneRecord] = field(default_factory=list)
    places: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    scenes: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)

    @property
    def signs(self) -> List[str]:
        return sorted(self.places.keys())


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _place_from_meta(
    meta: Mapping,
    *,
    scene_id: str,
    pool_crop_kind: str,
) -> Optional[str]:
    jid = meta.get("junction_id")
    way = meta.get("osm_way_id") or meta.get("way_id")
    crop = str(meta.get("crop_kind") or pool_crop_kind or "")
    sid = str(scene_id or "")

    # Segments must key by OSM way even when a nearby junction_id is present.
    if crop == "segment" or sid.startswith("seg_"):
        if way is not None and str(way).strip():
            return f"way:{way}"
        m = _SEG_RE.match(sid)
        if m:
            return f"way:{m.group(1)}"
    if jid is not None and str(jid).strip():
        return f"junction:{jid}"
    if way is not None and str(way).strip():
        return f"way:{way}"
    return None


def _place_from_scene_id(scene_id: str, crop_kind: str) -> Optional[str]:
    sid = str(scene_id or "").strip()
    if not sid:
        return None
    m = _JUNC_RE.match(sid)
    if m:
        return f"junction:{m.group(1)}"
    m = _SEG_RE.match(sid)
    if m:
        return f"way:{m.group(1)}"
    if crop_kind == "segment":
        return f"scene:{sid}"
    if _DUAL_RE.match(sid) or _RB_RE.match(sid):
        return f"scene:{sid}"
    return f"scene:{sid}"


def _resolve_scene_dir(rec: Mapping, sign_dir: Path) -> Optional[Path]:
    raw = rec.get("path") or ""
    if raw:
        p = Path(str(raw))
        if p.is_dir():
            return p
    sid = str(rec.get("scene_id") or "")
    cand = sign_dir / sid
    if cand.is_dir():
        return cand
    moscow = rec.get("moscow_path")
    if moscow:
        mp = Path(str(moscow))
        if mp.is_dir():
            return mp
    return None


def place_id_for_record(
    rec: Mapping,
    *,
    sign_dir: Path,
    read_meta: bool,
) -> Tuple[str, str]:
    crop_kind = str(rec.get("crop_kind") or "")
    scene_id = str(rec.get("scene_id") or "")
    place: Optional[str] = None
    if read_meta:
        scene_dir = _resolve_scene_dir(rec, sign_dir)
        if scene_dir is not None:
            meta = _read_json(scene_dir / "meta.json")
            if meta:
                if not crop_kind:
                    crop_kind = str(meta.get("crop_kind") or "")
                place = _place_from_meta(
                    meta, scene_id=scene_id, pool_crop_kind=crop_kind
                )
    if place is None:
        place = _place_from_scene_id(scene_id, crop_kind) or f"scene:{scene_id or 'unknown'}"
    return place, crop_kind or "unknown"


def load_catalog(
    scenes_root: Path | None = None,
    *,
    read_meta: bool = True,
    splits: Sequence[str] = ("train", "test"),
) -> OverlapCatalog:
    root = Path(scenes_root) if scenes_root is not None else DATA_SCENES
    allowed = {str(s) for s in splits}
    cat = OverlapCatalog()
    if not root.is_dir():
        return cat

    seen_dirs: Set[Path] = set()
    for sign_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        real_dir = sign_dir.resolve()
        if real_dir in seen_dirs:
            continue  # alias symlink (main_road -> main) — same pool, do not double count
        seen_dirs.add(real_dir)
        pool_path = sign_dir / "moscow_pool.json"
        if not pool_path.is_file():
            continue
        payload = _read_json(pool_path)
        if not payload:
            continue
        sign = sign_dir.name
        cat.places.setdefault(sign, {"train": set(), "test": set(), "unknown": set()})
        cat.scenes.setdefault(sign, {"train": set(), "test": set(), "unknown": set()})

        for rec in payload.get("scenes") or []:
            if not isinstance(rec, dict):
                continue
            split = str(rec.get("split") or "unknown").lower()
            if split not in ("train", "test"):
                split = "unknown"
            if allowed and split not in allowed and split != "unknown":
                continue
            scene_id = str(rec.get("scene_id") or "")
            if not scene_id:
                continue
            place, crop_kind = place_id_for_record(
                rec, sign_dir=sign_dir, read_meta=read_meta
            )
            cat.records.append(
                SceneRecord(
                    sign=sign,
                    scene_id=scene_id,
                    split=split,
                    place_id=place,
                    crop_kind=crop_kind,
                    path=str(rec.get("path") or "") or None,
                )
            )
            cat.places[sign][split].add(place)
            cat.scenes[sign][split].add(scene_id)
    return cat


def per_sign_counts(cat: OverlapCatalog) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for sign in cat.signs:
        out[sign] = {
            "train_places": len(cat.places[sign]["train"]),
            "test_places": len(cat.places[sign]["test"]),
            "unknown_places": len(cat.places[sign]["unknown"]),
            "train_scenes": len(cat.scenes[sign]["train"]),
            "test_scenes": len(cat.scenes[sign]["test"]),
            "family": SIGN_FAMILY.get(sign, "other"),
            "semantic_group": SIGN_SEMANTIC.get(sign, "other"),
            "pdd_code": _PDD_BY_FOLDER.get(sign, ""),
        }
    return out


def places_for_split(cat: OverlapCatalog, split: str) -> Dict[str, Set[str]]:
    return {sign: set(cat.places[sign].get(split) or ()) for sign in cat.signs}


def scenes_for_split(cat: OverlapCatalog, split: str) -> Dict[str, Set[str]]:
    return {sign: set(cat.scenes[sign].get(split) or ()) for sign in cat.signs}


def pairwise_intersection_matrix(
    place_sets: Mapping[str, Set[str]],
    signs: Optional[List[str]] = None,
) -> Tuple[List[str], List[List[int]]]:
    labels = list(signs) if signs is not None else sorted(place_sets.keys())
    mat = [
        [len(place_sets[a] & place_sets[b]) if a != b else len(place_sets[a]) for b in labels]
        for a in labels
    ]
    return labels, mat


def top_overlapping_pairs(
    place_sets: Mapping[str, Set[str]],
    *,
    top_n: int = 20,
) -> List[Tuple[int, str, str]]:
    labels = sorted(place_sets.keys())
    pairs: List[Tuple[int, str, str]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            n = len(place_sets[a] & place_sets[b])
            if n:
                pairs.append((n, a, b))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    return pairs[:top_n]


def place_sign_degree(place_sets: Mapping[str, Set[str]]) -> Dict[str, Set[str]]:
    inv: Dict[str, Set[str]] = defaultdict(set)
    for sign, places in place_sets.items():
        for pid in places:
            inv[pid].add(sign)
    return dict(inv)


def shared_places(
    place_sets: Mapping[str, Set[str]],
    *,
    min_signs: int = 2,
) -> List[Tuple[str, Tuple[str, ...]]]:
    deg = place_sign_degree(place_sets)
    rows = [
        (pid, tuple(sorted(signs)))
        for pid, signs in deg.items()
        if len(signs) >= min_signs
    ]
    rows.sort(key=lambda r: (-len(r[1]), r[0]))
    return rows


def unique_vs_shared_per_sign(
    place_sets: Mapping[str, Set[str]],
) -> Dict[str, Dict[str, float]]:
    deg = place_sign_degree(place_sets)
    out: Dict[str, Dict[str, float]] = {}
    for sign, places in place_sets.items():
        unique = sum(1 for p in places if len(deg.get(p, ())) == 1)
        shared = sum(1 for p in places if len(deg.get(p, ())) >= 2)
        total = len(places)
        out[sign] = {
            "unique": unique,
            "shared": shared,
            "total": total,
            "shared_pct": (100.0 * shared / total) if total else 0.0,
            "family": SIGN_FAMILY.get(sign, "other"),
            "semantic_group": SIGN_SEMANTIC.get(sign, "other"),
        }
    return out


def family_place_sets(
    place_sets: Mapping[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Union of places per behavioral family."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for sign, places in place_sets.items():
        out[SIGN_FAMILY.get(sign, "other")] |= set(places)
    return dict(out)


def semantic_place_sets(
    place_sets: Mapping[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Union of places per semantic group."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for sign, places in place_sets.items():
        out[SIGN_SEMANTIC.get(sign, "other")] |= set(places)
    return dict(out)


def family_scene_sets(
    scene_sets: Mapping[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Union of ``scene_id`` strings per behavioral family."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for sign, scenes in scene_sets.items():
        out[SIGN_FAMILY.get(sign, "other")] |= set(scenes)
    return dict(out)


def reuse_bucket_counts(
    place_sets: Mapping[str, Set[str]],
) -> Dict[str, int]:
    """Classify each place: unique / within_behavioral / within_semantic / across."""
    deg = place_sign_degree(place_sets)
    counts = {
        "unique": 0,
        "within_behavioral": 0,
        "within_semantic_diff_family": 0,
        "across_semantic": 0,
    }
    for signs in deg.values():
        if len(signs) == 1:
            counts["unique"] += 1
            continue
        behs = {SIGN_FAMILY.get(s, "other") for s in signs}
        sems = {SIGN_SEMANTIC.get(s, "other") for s in signs}
        if len(behs) == 1:
            counts["within_behavioral"] += 1
        elif len(sems) == 1:
            counts["within_semantic_diff_family"] += 1
        else:
            counts["across_semantic"] += 1
    return counts


def train_test_leakage_within_sign(cat: OverlapCatalog) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for sign in cat.signs:
        leak = cat.places[sign]["train"] & cat.places[sign]["test"]
        if leak:
            out[sign] = set(leak)
    return out


def train_test_leakage_across_signs(
    cat: OverlapCatalog,
) -> Tuple[List[str], List[List[int]]]:
    signs = cat.signs
    mat = [
        [len(cat.places[a]["train"] & cat.places[b]["test"]) for b in signs]
        for a in signs
    ]
    return signs, mat


def global_train_test_place_overlap(cat: OverlapCatalog) -> Set[str]:
    train: Set[str] = set()
    test: Set[str] = set()
    for sign in cat.signs:
        train |= cat.places[sign]["train"]
        test |= cat.places[sign]["test"]
    return train & test


def degree_histogram(place_sets: Mapping[str, Set[str]]) -> Dict[int, int]:
    deg = place_sign_degree(place_sets)
    hist: Dict[int, int] = defaultdict(int)
    for signs in deg.values():
        hist[len(signs)] += 1
    return dict(sorted(hist.items()))


def mean_offdiag(mat: List[List[int]]) -> float:
    n = len(mat)
    if n < 2:
        return 0.0
    total = 0
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total += mat[i][j]
            count += 1
    return total / count if count else 0.0
