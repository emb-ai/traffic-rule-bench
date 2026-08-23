"""Named run folders: ``debug/<timestamp>``, ``train``, ``test``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from traffic_bench.eval.sign_registry import SignProfile, resolve_repo_path, runs_dir

DEBUG_LATEST = "latest"


def experiment_folder(split: str) -> str:
    value = str(split or "debug").strip().lower()
    if value == "debug":
        return datetime.now().strftime("debug/%Y-%m-%d_%H-%M-%S")
    return value


def register_path_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "repo",
        lambda rel="": str(resolve_repo_path(rel)),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "run_folder",
        experiment_folder,
        replace=True,
    )


def latest_debug_dir(debug_root: Path) -> Optional[Path]:
    """``debug/latest`` if it points at a run, else the last timestamp child."""
    if not debug_root.is_dir():
        return None
    link = debug_root / DEBUG_LATEST
    if link.exists() or link.is_symlink():
        try:
            target = link.resolve()
        except OSError:
            target = None
        if target is not None and (target / "real_manifest.jsonl").is_file():
            return target
    kids = [
        child
        for child in debug_root.iterdir()
        if child.name != DEBUG_LATEST and (child / "real_manifest.jsonl").is_file()
    ]
    if not kids:
        return None
    return sorted(kids, key=lambda p: p.name)[-1]


def point_debug_latest(experiment_dir: Path) -> Optional[Path]:
    """If this is ``…/debug/<ts>``, point ``debug/latest`` at it."""
    if experiment_dir.parent.name != "debug":
        return None
    link = experiment_dir.parent / DEBUG_LATEST
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(experiment_dir.name, target_is_directory=True)
    return link


def resolve_manifest_in_dir(path: Path) -> Path:
    """``real_manifest.jsonl`` in ``path``, or latest debug child."""
    direct = path / "real_manifest.jsonl"
    if direct.is_file():
        return direct
    if path.name == "debug":
        latest = latest_debug_dir(path)
        if latest is not None:
            return latest / "real_manifest.jsonl"
    raise FileNotFoundError(f"No real_manifest.jsonl in {path}")


def default_run_manifest_dir(profile: SignProfile) -> Optional[Path]:
    """Prefer ``test/``, else ``debug/`` (latest timestamp or flat jsonl)."""
    test = runs_dir(profile) / "test"
    if (test / "real_manifest.jsonl").is_file():
        return test
    debug = runs_dir(profile) / "debug"
    try:
        resolve_manifest_in_dir(debug)
    except FileNotFoundError:
        return None
    return debug


def snapshot_output_base_and_name(experiment_dir: Path) -> tuple[str, str]:
    """``(data/runs/<sign>, train|test|debug/<ts>)`` relative to the repo when possible."""
    def _rel(p: Path) -> str:
        root = resolve_repo_path("")
        try:
            return str(p.resolve().relative_to(root))
        except ValueError:
            return str(p.resolve())

    if experiment_dir.parent.name == "debug":
        return _rel(experiment_dir.parent.parent), f"debug/{experiment_dir.name}"
    return _rel(experiment_dir.parent), experiment_dir.name
