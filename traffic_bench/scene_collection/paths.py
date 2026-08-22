"""Locations of maps/, splits, and data/scenes. One place for path constants."""

from __future__ import annotations

from pathlib import Path

SCENE_COLLECTION = Path(__file__).resolve().parent
REPO_ROOT = SCENE_COLLECTION.parent.parent

MAPS = SCENE_COLLECTION / "maps"
RAW = MAPS / "raw"
NETS = MAPS / "nets"
INDEX = MAPS / "index"
CROPS = MAPS / "crops"
JUNCTION_CROPS = CROPS / "junction"
DUAL_PATH_CROPS = CROPS / "dual_path"
SEGMENT_CROPS = CROPS / "segment"
SPLITS = MAPS / "splits"
PREVIEWS = MAPS / "previews"
ANALYSIS = SCENE_COLLECTION / "analysis" / "figures"
DATA_SCENES = REPO_ROOT / "data" / "scenes"

MOSCOW_NET = NETS / "moscow.net.xml"
JUNCTIONS_INDEX = INDEX / "junctions.jsonl"
SEGMENTS_INDEX = INDEX / "segments.jsonl"
DUAL_PATH_CANDIDATES = INDEX / "dual_path_candidates.jsonl"
SIGNS_YAML = SPLITS / "signs.yaml"
TRAIN_IDS = SPLITS / "train_ids.json"
TEST_IDS = SPLITS / "test_ids.json"
SIGN_ALLOCATIONS = SPLITS / "sign_allocations.json"
JUNCTIONS_OVERVIEW = PREVIEWS / "junctions_overview.png"
