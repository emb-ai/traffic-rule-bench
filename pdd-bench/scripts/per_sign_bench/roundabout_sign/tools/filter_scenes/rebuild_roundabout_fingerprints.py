#!/usr/bin/env python3
"""Rebuild scenes/roundabout_fingerprints.json from existing core and cropped scenes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROUNDABOUT_SIGN_DIR = TOOLS_DIR.parent.parent
SCENES_DIR_DEFAULT = ROUNDABOUT_SIGN_DIR / "scenes"

sys.path.insert(0, str(ROUNDABOUT_SIGN_DIR))

from lib.roundabout_fingerprint import rebuild_registry_from_scenes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild roundabout fingerprint registry")
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Scenes root (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--no-core",
        action="store_true",
        help="Only scan cropped scenes, skip scenes/core/",
    )
    args = parser.parse_args()

    scenes_root = args.scenes_dir.expanduser().resolve()
    registry = rebuild_registry_from_scenes(scenes_root, include_core=not args.no_core)
    print(
        f"Wrote {registry.path} with {len(registry.fingerprints)} unique roundabout(s), "
        f"{len(registry.by_scene)} scene mapping(s)."
    )


if __name__ == "__main__":
    main()
