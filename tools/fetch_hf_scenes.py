#!/usr/bin/env python
"""Fetch the official scene set from Hugging Face into ``data/`` — catalog scenes only.

``huggingface-cli download`` mirrors the whole repo tree, which after an
``upload_large_folder`` re-publish also contains every *previous* scene folder.
This script instead:

1. pins one dataset revision,
2. builds the file list from ``metadata/catalog.jsonl`` (2500 scenes),
3. deletes the previous ``data/scenes`` + ``data/metadata`` if the catalog changed
   (``--keep-previous`` moves them aside instead),
4. downloads only those scenes + metadata + assets (per file, resumable),
5. verifies every catalog scene has ``map.net.xml`` and ``meta.json``,
6. creates the eval aliases (``main_road -> main``, ``secondary_road -> secondary``),
7. writes ``data/scenes/<sign>/moscow_pool.json`` so ``paths.split=train|test`` works,
8. records the pinned revision in ``data/HF_REVISION``.

Usage:
    python tools/fetch_hf_scenes.py                 # into <repo>/data
    python tools/fetch_hf_scenes.py --workers 16 --revision <sha>
    python tools/fetch_hf_scenes.py --prune         # also delete scene dirs not in the catalog
    python tools/fetch_hf_scenes.py --skip-download # re-verify + aliases + pools only

Set ``HF_TOKEN`` (or ``hf auth login``) for higher rate limits; anonymous runs get
HTTP 429 bursts and finish slower. Re-running resumes and fills the gaps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "emb-ai/traffic-sign-bench"
META_FILES = (
    "README.md",
    ".gitattributes",
    "metadata/catalog.jsonl",
    "metadata/catalog.parquet",
    "metadata/sign_allocations.json",
    "metadata/signs.yaml",
    "metadata/train_ids.json",
    "metadata/test_ids.json",
)
REQUIRED_SCENE_FILES = ("map.net.xml", "meta.json")
# eval data_subdir -> HF sign folder (registry-derived aliases are added on top)
ALIASES: Dict[str, str] = {"main_road": "main", "secondary_road": "secondary"}
POOL_FILE = "moscow_pool.json"


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_revision(api, repo: str, revision: Optional[str]) -> str:
    info = api.dataset_info(repo, revision=revision)
    return str(info.sha)


def _fetch_catalog(repo: str, sha: str, tmp: Path):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo, "metadata/catalog.jsonl", repo_type="dataset", revision=sha, local_dir=str(tmp)
    )
    return Path(p)


def _clear_previous(data_dir: Path, new_catalog: Path, *, keep: bool) -> None:
    """Drop (or move aside with ``keep``) the previous scenes/ + metadata/ when the catalog changed."""
    old_catalog = data_dir / "metadata" / "catalog.jsonl"
    scenes = data_dir / "scenes"
    if not scenes.exists():
        return
    if old_catalog.is_file() and old_catalog.read_bytes() == new_catalog.read_bytes():
        print("[fetch] catalog unchanged → resuming in place")
        return
    targets = [scenes, data_dir / "metadata"]
    if keep:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for t in targets:
            if t.exists():
                dst = t.with_name(f"{t.name}_prev_{stamp}")
                shutil.move(str(t), str(dst))
                print(f"[fetch] previous {t.name} moved → {dst}")
        return
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            print(f"[fetch] previous {t.name}/ removed (catalog changed)")


def _list_files(api, repo: str, sha: str) -> List[str]:
    """One request for the whole tree — the paginated recursive listing is flaky on 14k files."""
    info = api.dataset_info(repo, revision=sha)
    return [s.rfilename for s in info.siblings]


def _select_files(all_files: List[str], rows: List[dict]) -> List[str]:
    wanted_dirs = {f"scenes/{r['sign_id']}/{r['scene_id']}/" for r in rows}
    keep: List[str] = []
    for f in all_files:
        if f in META_FILES or f.startswith("assets/"):
            keep.append(f)
            continue
        if f.startswith("scenes/"):
            parts = f.split("/")
            if len(parts) >= 4 and f"{parts[0]}/{parts[1]}/{parts[2]}/" in wanted_dirs:
                keep.append(f)
    return keep


def _download(repo: str, sha: str, data_dir: Path, files: List[str], workers: int) -> List[str]:
    """Per-file download in a thread pool; returns the files that failed after retries."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import hf_hub_download

    def one(fname: str) -> Optional[str]:
        # Anonymous downloads get HTTP 429 in bursts: back off up to ~1.5 min per file.
        for attempt in range(7):
            try:
                hf_hub_download(
                    repo, fname, repo_type="dataset", revision=sha, local_dir=str(data_dir)
                )
                return None
            except Exception as exc:  # noqa: BLE001 — retried, reported at the end
                if attempt == 6:
                    return f"{fname}: {exc}"
                time.sleep(min(90, 3 * 2**attempt))
        return None

    failed: List[str] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, f): f for f in files}
        for fut in as_completed(futs):
            done += 1
            err = fut.result()
            if err:
                failed.append(err)
            if done % 500 == 0 or done == len(files):
                print(f"[fetch] {done}/{len(files)} files, {len(failed)} failed, {time.time() - t0:.0f}s", flush=True)
    return failed


def _verify(data_dir: Path, rows: List[dict]) -> List[str]:
    missing: List[str] = []
    for r in rows:
        d = data_dir / "scenes" / r["sign_id"] / r["scene_id"]
        for name in REQUIRED_SCENE_FILES:
            f = d / name
            if not f.is_file() or f.stat().st_size == 0:
                missing.append(f"{r['sign_id']}/{r['scene_id']}/{name}")
    return missing


def _stale_dirs(data_dir: Path, by_sign: Dict[str, List[dict]]) -> List[Path]:
    stale: List[Path] = []
    for sign, rows in by_sign.items():
        sign_dir = data_dir / "scenes" / sign
        if not sign_dir.is_dir():
            continue
        wanted = {r["scene_id"] for r in rows}
        for child in sign_dir.iterdir():
            if child.is_dir() and not child.is_symlink() and child.name not in wanted:
                stale.append(child)
    return stale


def _aliases(catalog_signs: set) -> Dict[str, str]:
    out = dict(ALIASES)
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from traffic_bench.eval.sign_registry import list_profiles  # type: ignore

        for p in list_profiles():
            if p.data_subdir not in catalog_signs and p.id in catalog_signs:
                out[p.data_subdir] = p.id
    except Exception as exc:  # noqa: BLE001 — registry is optional here
        print(f"[fetch] sign registry unavailable ({exc}); using built-in aliases")
    return out


def _make_aliases(scenes_root: Path, aliases: Dict[str, str]) -> None:
    for alias, target in aliases.items():
        link = scenes_root / alias
        if not (scenes_root / target).is_dir():
            print(f"[fetch] alias {alias}: target {target} missing, skipped")
            continue
        if link.is_symlink():
            if os.readlink(link).rstrip("/").split("/")[-1] == target:
                continue
            link.unlink()
        elif link.exists():
            print(f"[fetch] alias {alias}: a real directory is in the way, left untouched")
            continue
        link.symlink_to(target)
        print(f"[fetch] alias {alias} -> {target}")


def _write_pools(data_dir: Path, by_sign: Dict[str, List[dict]], sha: str) -> None:
    alloc_file = data_dir / "metadata" / "sign_allocations.json"
    for sign, rows in sorted(by_sign.items()):
        sign_dir = data_dir / "scenes" / sign
        records = []
        n_fail = 0
        for r in sorted(rows, key=lambda x: x["scene_id"]):
            scene_dir = sign_dir / r["scene_id"]
            meta = _read_json(scene_dir / "meta.json") or {}
            if not meta:
                n_fail += 1
            records.append(
                {
                    "scene_id": r["scene_id"],
                    "shape": r.get("shape") or meta.get("shape"),
                    "crop_kind": r.get("crop_kind") or meta.get("crop_kind"),
                    "slot": meta.get("slot"),
                    "split": r["split"],
                    "path": str(scene_dir),
                    "moscow_path": f"hf:{HF_REPO}@{sha[:12]}",
                }
            )
        pool = {
            "sign": rows[0].get("pdd_code") or sign,
            "split": "all",
            "mode": "hf_download",
            "crop_kind": rows[0].get("crop_kind"),
            "allocations_file": str(alloc_file),
            "hf_repo": HF_REPO,
            "hf_revision": sha,
            "n_ok": len(records) - n_fail,
            "n_fail": n_fail,
            "scenes": records,
        }
        sign_dir.mkdir(parents=True, exist_ok=True)
        (sign_dir / POOL_FILE).write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--revision", default=None, help="commit sha / branch (default: main)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-previous", action="store_true", help="move the previous scenes/ aside instead of deleting it")
    ap.add_argument("--prune", action="store_true", help="delete scene dirs that are not in the catalog")
    ap.add_argument("--skip-download", action="store_true", help="only verify / aliases / pools")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    sha = _resolve_revision(api, args.repo, args.revision)
    print(f"[fetch] {args.repo} @ {sha}")

    with tempfile.TemporaryDirectory(prefix="hf_catalog_") as tmp:
        catalog = _fetch_catalog(args.repo, sha, Path(tmp))
        rows = _read_jsonl(catalog)
        if not args.skip_download:
            _clear_previous(data_dir, catalog, keep=args.keep_previous)

    by_sign: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_sign[r["sign_id"]].append(r)
    print(f"[fetch] catalog: {len(rows)} scenes, {len(by_sign)} signs")

    failed: List[str] = []
    if not args.skip_download:
        all_files = _list_files(api, args.repo, sha)
        files = _select_files(all_files, rows)
        print(f"[fetch] repo has {len(all_files)} files; downloading {len(files)} with {args.workers} workers …")
        failed = _download(args.repo, sha, data_dir, files, args.workers)
        if failed:
            print(f"[fetch] {len(failed)} files failed after retries, e.g.:")
            for f in failed[:10]:
                print(f"    {f}")

    missing = _verify(data_dir, rows)
    stale = _stale_dirs(data_dir, by_sign)
    if stale:
        print(f"[fetch] {len(stale)} scene dirs not in catalog" + (" → pruning" if args.prune else " (use --prune to delete)"))
        for d in stale[:10]:
            print(f"    {d.relative_to(data_dir)}")
        if args.prune:
            for d in stale:
                shutil.rmtree(d)

    scenes_root = data_dir / "scenes"
    _make_aliases(scenes_root, _aliases(set(by_sign)))
    _write_pools(data_dir, by_sign, sha)
    (data_dir / "HF_REVISION").write_text(f"{args.repo} {sha}\n", encoding="utf-8")

    print("\n[fetch] sign                        train  test  dirs")
    for sign, srows in sorted(by_sign.items()):
        n_dirs = sum(1 for p in (scenes_root / sign).iterdir() if p.is_dir()) if (scenes_root / sign).is_dir() else 0
        tr = sum(r["split"] == "train" for r in srows)
        te = sum(r["split"] == "test" for r in srows)
        print(f"[fetch] {sign:26s} {tr:5d} {te:5d} {n_dirs:5d}")
    if missing or failed:
        print(f"\n[fetch] MISSING {len(missing)} files ({len(failed)} download failures), e.g.:")
        for m in missing[:20]:
            print(f"    {m}")
        print("[fetch] re-run the same command to resume")
        return 1
    print(f"\n[fetch] OK: {len(rows)} scenes verified, revision {sha[:12]} pinned in {data_dir / 'HF_REVISION'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
