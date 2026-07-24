import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter scenes where expert succeeded without violations."
    )
    parser.add_argument(
        "--episodes", type=str, required=True,
        help="Path to benchmark_episodes_<sign>_<run>.json"
    )
    parser.add_argument(
        "--scenes-dir", type=str, default="pdd-bench/scenes",
        help="Source scenes root"
    )
    parser.add_argument(
        "--output-dir", type=str, default="pdd-bench/scenes_filtered",
        help="Output filtered scenes root"
    )
    args = parser.parse_args()

    scenes_root = Path(args.scenes_dir)
    out_root = Path(args.output_dir)

    with open(args.episodes, "r") as f:
        episodes = json.load(f)

    good_scenes = {}  # sign_type -> set of scene names
    for ep in episodes:
        if ep.get("success") and ep.get("violations", 1) == 0 and not ep.get("crashed"):
            st = ep.get("sign_type")
            sc = ep.get("scene")
            if st and sc:
                good_scenes.setdefault(st, set()).add(sc)

    if not good_scenes:
        print("No successful expert scenes found.")
        return

    total_copied = 0
    for sign_type, scene_names in good_scenes.items():
        src_sign = scenes_root / sign_type
        dst_sign = out_root / sign_type
        for name in sorted(scene_names):
            src = src_sign / name
            dst = dst_sign / name
            if not src.exists():
                print(f"  [skip] {src} not found")
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst / item.name)
            total_copied += 1
            print(f"  [copy] {sign_type}/{name}")

    print(f"\n✅ Copied {total_copied} scenes to {out_root}")


if __name__ == "__main__":
    main()
