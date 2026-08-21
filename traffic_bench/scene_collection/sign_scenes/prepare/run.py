"""Dispatch sign-specific map surgery after materialize.

Currently: ``prepare: crosswalk`` in signs.yaml → zebra in the middle of copied segments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from traffic_bench.eval.sign_registry import get_profile, list_profiles, scenes_dir as profile_scenes_dir
from traffic_bench.scene_collection.assign.assign import load_signs_yaml
from traffic_bench.scene_collection.paths import SIGNS_YAML


def _prepare_field(pdd_code: str) -> str:
    cfg = load_signs_yaml(SIGNS_YAML)
    spec = (cfg.get("signs") or {}).get(pdd_code) or {}
    return str(spec.get("prepare") or "").strip()


def signs_with_prepare() -> List[str]:
    cfg = load_signs_yaml(SIGNS_YAML)
    out: List[str] = []
    for pdd, spec in (cfg.get("signs") or {}).items():
        if spec and spec.get("prepare"):
            out.append(str(pdd))
    return out


def prepare_sign(sign: str, *, scenes_dir: Path | None = None) -> int:
    profile = get_profile(sign)
    hook = _prepare_field(profile.pdd_code)
    dest = scenes_dir or profile_scenes_dir(profile)
    if hook == "crosswalk":
        from traffic_bench.scene_collection.sign_scenes.prepare.crosswalk.add_zebra import (
            add_zebra_to_scenes_dir,
        )

        print(f"[prepare] sign={profile.id} ({profile.pdd_code}) zebra → {dest}")
        stats = add_zebra_to_scenes_dir(dest)
        print(f"[prepare] {stats}")
        return 0 if stats.get("fail", 0) == 0 else 1
    if not hook:
        print(
            f"sign {profile.id!r} has no prepare: in signs.yaml; "
            "materialize already placed the maps."
        )
        return 0
    print(f"ERROR: unknown prepare={hook!r} for {profile.id}", file=sys.stderr)
    return 2


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--sign",
        metavar="ID",
        help=f"Eval id (e.g. crosswalk). Known: {', '.join(sorted(p.id for p in list_profiles()))}",
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="Run every sign with prepare: in signs.yaml",
    )
    ap.add_argument("--scenes-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.all:
        pdds = signs_with_prepare()
        if not pdds:
            print("[prepare] no signs with prepare: in signs.yaml")
            return 0
        rc = 0
        for pdd in pdds:
            profile = get_profile(pdd)
            rc = max(rc, prepare_sign(profile.id, scenes_dir=args.scenes_dir))
        return rc
    return prepare_sign(str(args.sign), scenes_dir=args.scenes_dir)


if __name__ == "__main__":
    raise SystemExit(main())
