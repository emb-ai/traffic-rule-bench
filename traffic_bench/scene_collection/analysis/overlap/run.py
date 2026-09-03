"""CLI for cross-sign place overlap analysis + allocation verify."""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_bench.scene_collection.analysis.overlap.catalog import load_catalog
from traffic_bench.scene_collection.analysis.overlap.figures import write_all
from traffic_bench.scene_collection.analysis.overlap.report import write_report
from traffic_bench.scene_collection.paths import DATA_SCENES, SCENE_COLLECTION

DEFAULT_OUT = SCENE_COLLECTION / "analysis" / "overlap"

_STALE_FIGURES = (
    "train_family_pairwise.png",
    "test_family_pairwise.png",
    "train_family_scene_pairwise.png",
    "test_family_scene_pairwise.png",
    "train_family_unique_vs_shared.png",
    "place_type_by_split.png",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m traffic_bench.scene_collection analysis overlap",
        description="Cross-sign map overlap (train/test place reuse) with figures.",
    )
    ap.add_argument("--scenes-root", type=Path, default=DATA_SCENES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    print(f"[overlap] Loading pools from {args.scenes_root} (meta={not args.no_meta})…")
    cat = load_catalog(args.scenes_root, read_meta=not args.no_meta)
    print(f"[overlap] signs={len(cat.signs)} records={len(cat.records)}")

    out = args.out.expanduser().resolve()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name in _STALE_FIGURES:
        stale = fig_dir / name
        if stale.is_file():
            stale.unlink()
            print(f"[overlap] removed stale {name}")

    print(f"[overlap] Writing figures → {fig_dir}")
    written = write_all(cat, fig_dir, pdf=args.pdf)
    report = write_report(cat, out, figures_rel="figures")
    print(f"[overlap] {len(written)} figure files, report {report}")

    if not args.skip_verify:
        from traffic_bench.scene_collection.analysis.assign_verify import main as verify_main

        print("[overlap] Refreshing allocation_verify.* …")
        verify_main(["--out-dir", str(out)])
        splits = SCENE_COLLECTION / "maps" / "splits"
        for name in ("allocation_verify.md", "allocation_verify.json"):
            leftover = splits / name
            if leftover.is_file() and leftover.resolve() != (out / name).resolve():
                leftover.unlink()
                print(f"[overlap] removed leftover {leftover}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
