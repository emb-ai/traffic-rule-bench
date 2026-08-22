"""Upload a packed HF folder to emb-ai/traffic-sign-bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from traffic_bench.scene_collection.publish.layout import (
    DEFAULT_STAGING,
    HF_REPO,
    pack_hf_dataset,
)


def upload_folder(staging: Path, repo: str) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 2
    if not staging.is_dir():
        print(f"ERROR: staging not found: {staging}", file=sys.stderr)
        return 1
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    upload = getattr(api, "upload_large_folder", None)
    print(f"[publish] {staging} → hf://datasets/{repo}")
    if upload is not None:
        upload(folder_path=str(staging), repo_id=repo, repo_type="dataset")
    else:
        api.upload_folder(folder_path=str(staging), repo_id=repo, repo_type="dataset")
    print(f"[publish] https://huggingface.co/datasets/{repo}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m traffic_bench.scene_collection publish",
        description="Pack official sign scenes and/or upload to Hugging Face.",
    )
    ap.add_argument("--repo", default=HF_REPO, help=f"Dataset repo (default: {HF_REPO})")
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_STAGING,
        help=f"Staging folder (default: {DEFAULT_STAGING})",
    )
    ap.add_argument(
        "--no-pack",
        action="store_true",
        help="Upload an already-packed --out folder",
    )
    ap.add_argument("--scenes-dir", type=Path, default=None, help="Override data/scenes")
    args = ap.parse_args(argv)
    if not args.no_pack:
        pack_hf_dataset(args.out, scenes_root=args.scenes_dir)
    return upload_folder(args.out.expanduser().resolve(), args.repo)


if __name__ == "__main__":
    raise SystemExit(main())
