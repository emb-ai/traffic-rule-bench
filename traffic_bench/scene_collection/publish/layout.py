"""Build the Hugging Face folder: scenes/ + metadata/ + assets/ + README.md."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from traffic_bench.eval.sign_registry import (
    SignProfile,
    get_profile,
    list_profiles,
    scenes_dir as profile_scenes_dir,
)
from traffic_bench.scene_collection.paths import (
    ANALYSIS,
    REPO_ROOT,
    SIGN_ALLOCATIONS,
    SIGNS_YAML,
    TEST_IDS,
    TRAIN_IDS,
)
from traffic_bench.scene_collection.sign_scenes.filter.selection import is_reserved_scene_dir

KEEP_FILES = ("map.net.xml", "meta.json", "custom_cropped.png")
SKIP_NAMES = {"moscow_pool.json", "scene_selection.json", "center.json"}
EXAMPLE_SIGNS = (
    "yield",
    "stop",
    "roundabout",
    "crosswalk",
    "no_entry",
    "detour_right",
    "speed_limit",
    "one_way_right",
)
DEFAULT_STAGING = REPO_ROOT / "dist" / "hf-traffic-sign-bench"
HF_REPO = "emb-ai/traffic-sign-bench"
GITHUB = "https://github.com/emb-ai/traffic-rule-bench"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_index(allocations: dict) -> Dict[str, Dict[str, str]]:
    """pdd_code → {scene_id: train|test}."""
    out: Dict[str, Dict[str, str]] = {}
    for pdd, block in (allocations.get("signs") or {}).items():
        half: Dict[str, str] = {}
        for split in ("train", "test"):
            for sid in (block.get(split) or {}).get("scene_ids") or []:
                half[str(sid)] = split
        out[str(pdd)] = half
    return out


def _iter_scene_dirs(src: Path) -> Iterable[Path]:
    if not src.is_dir():
        return
    for child in sorted(src.iterdir()):
        if child.name in SKIP_NAMES:
            continue
        if child.is_dir() and is_reserved_scene_dir(child.name):
            continue
        if child.is_symlink() or child.is_dir():
            if child.is_file() or (child.is_symlink() and child.resolve().is_file()):
                continue
            yield child


def _copy_scene(src: Path, dst: Path) -> bool:
    real = src.resolve() if src.is_symlink() else src
    net = real / "map.net.xml"
    if not net.is_file():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP_FILES:
        path = real / name
        if path.is_file():
            shutil.copy2(path, dst / name)
    leftover = dst / "center.json"
    if leftover.is_file():
        leftover.unlink()
    return (dst / "map.net.xml").is_file()


def _meta(scene_dir: Path) -> dict:
    path = scene_dir / "meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_parquet(rows: Sequence[dict], path: Path) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)
    return True


def _size_category(n: int) -> str:
    if n < 1000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    return "100K<n<1M"


def _sign_table_rows(catalog: Sequence[dict], profiles: Sequence[SignProfile]) -> List[str]:
    counts: Dict[str, Counter] = {}
    kinds: Dict[str, str] = {}
    for row in catalog:
        sid = str(row["sign_id"])
        counts.setdefault(sid, Counter())[str(row.get("split") or "")] += 1
        if row.get("crop_kind"):
            kinds[sid] = str(row["crop_kind"])
    lines = [
        "| Sign | PDD | Family | Train | Test | Total |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    by_id = {p.id: p for p in profiles}
    for sign_id in sorted(counts):
        profile = by_id.get(sign_id)
        pdd = profile.pdd_code if profile else ""
        family = kinds.get(sign_id, "")
        n_train = counts[sign_id].get("train", 0)
        n_test = counts[sign_id].get("test", 0)
        total = sum(counts[sign_id].values())
        lines.append(
            f"| `{sign_id}` | {pdd} | {family} | {n_train} | {n_test} | {total} |"
        )
    return lines


def _gallery_md(example_rel: Sequence[str]) -> str:
    if not example_rel:
        return ""
    cells = []
    for rel in example_rel:
        stem = Path(rel).stem
        cells.append(f'<img src="{rel}" alt="{stem}" width="220"/>')
    return "<p align=\"center\">\n  " + "\n  ".join(cells) + "\n</p>\n"


def render_dataset_card(
    *,
    n_scenes: int,
    n_signs: int,
    catalog_file: str,
    sign_table: Sequence[str],
    gallery: str,
) -> str:
    size = _size_category(n_scenes)
    table = "\n".join(sign_table)
    return f"""---
pretty_name: Traffic Sign Bench
license: odbl
task_categories:
  - other
language:
  - en
tags:
  - autonomous-driving
  - traffic-signs
  - sumo
  - simulation
  - osm
size_categories:
  - {size}
configs:
  - config_name: catalog
    default: true
    data_files: metadata/{catalog_file}
---

# Traffic Sign Bench

Official per-sign SUMO maps for [TrafficRuleBench]({GITHUB}): real Moscow OSM
layouts, **{n_signs} signs**, **{n_scenes} maps**. Protocol size is
**80 train + 20 test** maps per sign.

Road geometry is derived from [OpenStreetMap](https://www.openstreetmap.org/copyright)
© OpenStreetMap contributors and is released under **ODbL 1.0**.

{gallery}

## Download

All scenes land under `data/scenes/<sign>/<scene_id>/`, which is what eval
expects:

```bash
huggingface-cli download {HF_REPO} \\
    --repo-type dataset \\
    --local-dir data
```

Code and runners: [{GITHUB}]({GITHUB}).

```python
from datasets import load_dataset
ds = load_dataset("{HF_REPO}", split="train")
```

That loads the **catalog** (one row per scene, with a path to the preview and
the net). The `.net.xml` files themselves are next to the catalog in `scenes/`.

## Layout

```
scenes/<sign>/<scene_id>/
  map.net.xml
  meta.json
  custom_cropped.png
metadata/
  signs.yaml
  sign_allocations.json
  train_ids.json
  test_ids.json
  catalog.jsonl
  catalog.parquet
```

`sign` is the eval id (`yield`, not `2.4`). Train/test is in
`sign_allocations.json` and the catalog, not in the folder name.

## Signs

{table}

## How it was built

Moscow OSM → city SUMO net → junction / dual-path / segment crops → each sign
**queries** that shared pool (`signs.yaml`) → materialize into
`data/scenes/<sign>/`. Pedestrian crossings (PDD 5.19) get a mid-block zebra
injected after copy.

Split is stamped on **place identity** before allocation:

- junctions and dual-path: `junction_id`
- segments: `osm_way_id` (one street does not appear in both halves)

Signs may reuse the same physical place within a split when allowed by the tiered
assign policy (same behavioral family, or same semantic group).

## Limitations

- Geography is Moscow only.
- On some 5.19 segments SUMO omits the `crossing` edge; the split node and
  sidewalks are still there.
- These folders are maps, not closed-loop eval manifests.

## Citation

```bibtex
@misc{{traffic-sign-bench,
  title  = {{Traffic Sign Bench}},
  author = {{EMB AI}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{HF_REPO}}}
}}
```
"""


def pack_hf_dataset(
    out: Path,
    *,
    scenes_root: Optional[Path] = None,
    allocations_path: Path = SIGN_ALLOCATIONS,
    signs_yaml: Path = SIGNS_YAML,
) -> Dict[str, Any]:
    """Write a self-contained HF dataset folder. Returns pack stats."""
    out = out.expanduser().resolve()
    scenes_out = out / "scenes"
    meta_out = out / "metadata"
    assets_out = out / "assets" / "examples"
    if out.exists():
        shutil.rmtree(out)
    scenes_out.mkdir(parents=True)
    meta_out.mkdir(parents=True)
    assets_out.mkdir(parents=True)

    allocations = _load_json(allocations_path) if allocations_path.is_file() else {}
    splits = _split_index(allocations)
    _copy_if_exists(signs_yaml, meta_out / "signs.yaml")
    _copy_if_exists(allocations_path, meta_out / "sign_allocations.json")
    _copy_if_exists(TRAIN_IDS, meta_out / "train_ids.json")
    _copy_if_exists(TEST_IDS, meta_out / "test_ids.json")
    inventory = ANALYSIS / "inventory.png"
    _copy_if_exists(inventory, out / "assets" / "inventory.png")

    catalog: List[dict] = []
    packed_signs: List[str] = []
    examples: List[str] = []

    for profile in list_profiles():
        src = (scenes_root / profile.data_subdir) if scenes_root else profile_scenes_dir(profile)
        if not src.is_dir():
            continue
        dest_sign = scenes_out / profile.id
        n_here = 0
        half = splits.get(profile.pdd_code, {})
        alloc_kind = str((allocations.get("signs") or {}).get(profile.pdd_code, {}).get("crop_kind") or "")
        for scene_src in _iter_scene_dirs(src):
            scene_id = scene_src.name
            dest = dest_sign / scene_id
            if not _copy_scene(scene_src, dest):
                if dest.exists():
                    shutil.rmtree(dest)
                continue
            meta = _meta(dest)
            split = half.get(scene_id, "")
            catalog.append(
                {
                    "sign_id": profile.id,
                    "pdd_code": profile.pdd_code,
                    "split": split,
                    "scene_id": scene_id,
                    "crop_kind": alloc_kind or str(meta.get("crop_kind") or meta.get("scene_kind") or ""),
                    "shape": str(meta.get("shape") or ""),
                    "scene_kind": str(meta.get("scene_kind") or ""),
                    "latitude": meta.get("latitude"),
                    "longitude": meta.get("longitude"),
                    "road_id": str(meta.get("road_id") or ""),
                    "preview": f"scenes/{profile.id}/{scene_id}/custom_cropped.png",
                    "net_path": f"scenes/{profile.id}/{scene_id}/map.net.xml",
                }
            )
            n_here += 1
        if n_here:
            packed_signs.append(profile.id)
            print(f"[pack] {profile.id}: {n_here} → {dest_sign}")
            if profile.id in EXAMPLE_SIGNS:
                png = next(dest_sign.glob("*/custom_cropped.png"), None)
                if png and png.is_file():
                    rel = f"assets/examples/{profile.id}.png"
                    shutil.copy2(png, out / rel)
                    examples.append(rel)

    catalog_jsonl = meta_out / "catalog.jsonl"
    with catalog_jsonl.open("w", encoding="utf-8") as handle:
        for row in catalog:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    wrote_parquet = _write_parquet(catalog, meta_out / "catalog.parquet")
    catalog_file = "catalog.parquet" if wrote_parquet else "catalog.jsonl"

    card = render_dataset_card(
        n_scenes=len(catalog),
        n_signs=len(packed_signs),
        catalog_file=catalog_file,
        sign_table=_sign_table_rows(catalog, list_profiles()),
        gallery=_gallery_md(examples),
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    stats = {
        "out": str(out),
        "signs": packed_signs,
        "n_scenes": len(catalog),
        "n_signs": len(packed_signs),
        "catalog": catalog_file,
        "parquet": wrote_parquet,
    }
    print(
        f"[pack] HF layout: {len(catalog)} scenes / {len(packed_signs)} signs → {out}"
    )
    return stats


def pack_one_sign(sign: str, out: Path, *, scenes_dir: Optional[Path] = None) -> int:
    """Legacy standalone folder (whole scene dirs, symlinks followed)."""
    profile = get_profile(sign)
    src = scenes_dir or profile_scenes_dir(profile)
    if not src.is_dir():
        print(f"ERROR: scenes dir not found: {src}", file=sys.stderr)
        return 1
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for child in sorted(src.iterdir()):
        name = child.name
        if child.is_dir() and is_reserved_scene_dir(name):
            continue
        dest = out / name
        if child.is_dir() or child.is_symlink():
            if name in SKIP_NAMES:
                continue
            if child.is_file() or (child.is_symlink() and child.resolve().is_file()):
                shutil.copy2(child, dest)
            else:
                if dest.exists():
                    shutil.rmtree(dest)
                real = child.resolve() if child.is_symlink() else child
                shutil.copytree(real, dest, symlinks=False)
                leftover = dest / "center.json"
                if leftover.is_file():
                    leftover.unlink()
                n += 1
        elif child.is_file() and name not in SKIP_NAMES:
            shutil.copy2(child, dest)
    print(f"[pack] {profile.id}: {n} scenes → {out}")
    print("This folder is self-contained (symlinks followed).")
    return 0
