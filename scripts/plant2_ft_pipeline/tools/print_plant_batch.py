#!/usr/bin/env python3
"""Instantiate PlanTDataset + DataLoader and print a batch (no training).

Examples::

  /home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python \\
    scripts/plant2_ft_pipeline/tools/print_plant_batch.py

  # same dump, explicit path (parent or .../data both work)
  .../python print_plant_batch.py \\
    --ds .../plant2_stop_pipeline_debug400/plant2_l1_stop_train \\
    --batch-size 2 --num-batches 1 --max-samples 8

  # first samples of the first iterdir route are often n_objs=1; find convoy frames:
  .../python print_plant_batch.py --min-objs 2 --batch-size 2

  # frames 0–4 are skipped by PlanTDataset; guarantee a 2.5 row in x_objs:
  .../python print_plant_batch.py --require-class 2.5 --batch-size 1
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PLAN_T = _ROOT.parents[1] / "plant2" / "PlanT"
if str(_PLAN_T) not in sys.path:
    sys.path.insert(0, str(_PLAN_T))

import argparse
import gzip
import json
import os
from collections import Counter
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from lib.env import plan_t, trb_root
from util.sign_id import SIGN_CODES

DEFAULT_DS = (
    trb_root() / "plant2_stop_pipeline_debug400" / "plant2_l1_stop_train"
)

_TYPE_NAMES: dict[float, str] = {
    0.0: "pad",
    1.0: "car",
    2.0: "walker",
    3.0: "static",
    4.0: "stop_sign",
    5.0: "traffic_light",
    6.0: "emergency",
}
for _i, _code in enumerate(SIGN_CODES):
    _TYPE_NAMES[float(7 + _i)] = _code

_XOBJ_COLS = ("type", "x", "y", "yaw_deg", "speed_kmh", "ext_y", "ext_x")
_SIGN_LIKE = set(SIGN_CODES) | {"stop_sign"}
_SIGN_RANGE_M = 30.0


def type_name(t: float) -> str:
    return _TYPE_NAMES.get(float(t), f"type_{t:g}")


def n_objs_from_boxes_file(path: str | Path) -> int:
    """PlanT n_objs ≈ dump objects minus ego at boxes[0]."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        boxes = json.load(f)
    if not isinstance(boxes, list):
        return 0
    return max(0, len(boxes) - 1)


def _label_path_str(label_path: Any) -> str:
    if isinstance(label_path, bytes):
        return label_path.decode()
    return str(label_path)


def boxes_has_class(path: str | Path, class_name: str) -> bool:
    """True if class_name would survive PlanTDataset filters into x_objs.

    Signs (PDD codes / stop_sign) need affects_ego and a 30 m xy radius.
    Scans the current-frame boxes file (labels[i][0] at seq_len=1).
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        boxes = json.load(f)
    if not isinstance(boxes, list) or len(boxes) < 2:
        return False
    want = str(class_name)
    sign_like = want in _SIGN_LIKE
    for obj in boxes[1:]:
        if str(obj.get("class")) != want:
            continue
        if sign_like:
            pos = obj.get("position") or [0.0, 0.0, 0.0]
            px, py = float(pos[0]), float(pos[1])
            pz = float(pos[2]) if len(pos) > 2 else 0.0
            if px * px + py * py > _SIGN_RANGE_M ** 2 or abs(pz) > _SIGN_RANGE_M:
                continue
            if not obj.get("affects_ego"):
                continue
        return True
    return False


def sign_id_label(sid: int) -> str:
    if sid <= 0:
        return "unknown"
    if 1 <= sid <= len(SIGN_CODES):
        return SIGN_CODES[sid - 1]
    return f"?{sid}"


def _has_routes(path: Path) -> bool:
    try:
        return any(p.is_dir() and (p / "boxes").is_dir() for p in path.iterdir())
    except OSError:
        return False


def resolve_data_root(ds: Path) -> Path:
    """Accept dump parent (.../plant2_l1_stop_train) or .../data."""
    ds = ds.expanduser().resolve()
    if not ds.exists():
        raise SystemExit(f"ERROR: --ds does not exist: {ds}")
    if (ds / "data").is_dir() and _has_routes(ds / "data"):
        return ds / "data"
    if _has_routes(ds):
        return ds
    raise SystemExit(
        f"ERROR: no route dirs with boxes/ under {ds} or {ds / 'data'}"
    )


def load_cfg(*, augment: bool, filter_routes: bool):
    from omegaconf import OmegaConf, open_dict

    pt = plan_t()
    cfg = OmegaConf.load(pt / "config" / "config.yaml")
    cfg = OmegaConf.merge(
        cfg,
        {
            "user": OmegaConf.load(pt / "config" / "user" / "arbelyaev.yaml"),
            "model": OmegaConf.load(pt / "config" / "model" / "PlanT.yaml"),
        },
    )
    with open_dict(cfg):
        cfg.use_caching = False
        cfg.model.training.augment = bool(augment)
        cfg.model.training.augment_parked = False
        cfg.model.training.filter_routes = bool(filter_routes)
    return cfg


def _preview_tensor(key: str, v: torch.Tensor, *, max_elems: int = 8) -> str:
    parts = [f"Tensor {tuple(v.shape)} {v.dtype}"]
    if v.numel() == 0:
        return parts[0] + " (empty)"
    if v.numel() == 1:
        return parts[0] + f" value={v.item()!r}"
    if key == "BEV":
        return (
            parts[0]
            + f" min={v.min().item():.4f} max={v.max().item():.4f} "
            f"mean={v.mean().item():.4f}"
        )
    if v.dim() == 1 and v.numel() <= max_elems:
        return parts[0] + f" values={v.tolist()}"
    if v.dim() == 1:
        return parts[0] + f" [:{max_elems}]={v[:max_elems].tolist()}"
    first = v.reshape(v.shape[0], -1)[0, :max_elems].tolist()
    return parts[0] + f" [0,:{max_elems}]={first}"


def _preview_non_tensor(v: Any, *, max_chars: int = 240) -> str:
    if isinstance(v, (str, bytes)):
        s = v.decode() if isinstance(v, bytes) else v
        return f"str {s[:max_chars]!r}" + ("..." if len(s) > max_chars else "")
    if isinstance(v, (int, float, bool)) or v is None:
        return f"{type(v).__name__} {v!r}"
    if isinstance(v, (list, tuple)):
        n = len(v)
        head = v[:3]
        return f"{type(v).__name__} len={n} head={head!r}"[:max_chars]
    return f"{type(v).__name__} {str(v)[:max_chars]}"


def _fmt_xobj_row(row) -> str:
    vals = [float(x) for x in row]
    name = type_name(vals[0])
    return (
        f"{name:12s} type={vals[0]:g}  xy=({vals[1]:7.2f},{vals[2]:7.2f})  "
        f"yaw={vals[3]:7.2f}  spd={vals[4]:6.2f}  ext=({vals[5]:.2f},{vals[6]:.2f})"
    )


def print_xobjs(x_objs: torch.Tensor, idxs: torch.Tensor, *, max_rows: int = 8) -> None:
    """Print per-sample object counts + a few rows (skip padding at x_objs[0])."""
    print(f"  x_objs {tuple(x_objs.shape)} {x_objs.dtype}  (row 0 is padding)")
    print(f"  idxs   {tuple(idxs.shape)} {idxs.dtype}")
    B, maxseq = idxs.shape
    for b in range(B):
        ptrs = idxs[b]
        n = int((ptrs > 0).sum().item()) if maxseq else 0
        rows = x_objs[ptrs[:n].long()] if n else x_objs[:0]
        types = [type_name(float(t)) for t in rows[:, 0].tolist()] if n else []
        counts = Counter(types)
        hist = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
        print(f"  sample[{b}] n_objs={n}  {hist}")
        for row in rows[:max_rows].tolist():
            print(f"    {_fmt_xobj_row(row)}")
        if n > max_rows:
            print(f"    ... {n - max_rows} more")


def print_batch(batch: dict, *, batch_i: int) -> None:
    print(f"\n=== batch {batch_i}  keys={list(batch.keys())} ===")
    for key in batch:
        v = batch[key]
        if key in ("x_objs", "idxs"):
            continue
        if torch.is_tensor(v):
            print(f"  {key:24s} {_preview_tensor(key, v)}")
        else:
            print(f"  {key:24s} {_preview_non_tensor(v)}")

    if "sign_id" in batch and torch.is_tensor(batch["sign_id"]):
        sids = [int(x) for x in batch["sign_id"].flatten().tolist()]
        labels = [f"{sid}:{sign_id_label(sid)}" for sid in sids]
        print(f"  sign_id labels          {labels}")

    if "x_objs" in batch and "idxs" in batch:
        print_xobjs(batch["x_objs"], batch["idxs"])


def print_raw_sample(sample: dict, *, index: int, label_path: str) -> None:
    print(f"\n=== raw sample index={index} ===")
    print(f"  label[0]  {label_path}")
    sid = int(sample.get("sign_id", 0))
    print(f"  sign_id   {sid} ({sign_id_label(sid)})")
    print(f"  keys      {sorted(sample.keys())}")
    inp = sample.get("input") or []
    print(f"  input     {len(inp)} objects")
    for row in inp[:8]:
        print(f"    {_fmt_xobj_row(row)}")
    if len(inp) > 8:
        print(f"    ... {len(inp) - 8} more")
    for key in ("target_speed", "ego_speed", "speed_limit", "ego_pos", "ego_rot"):
        if key in sample:
            print(f"  {key:10s} {sample[key]!r}")
    if "BEV" in sample and torch.is_tensor(sample["BEV"]):
        print(f"  BEV       {_preview_tensor('BEV', sample['BEV'])}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load PlanTDataset and print one/few DataLoader batches (no training).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ds",
        type=Path,
        default=DEFAULT_DS,
        help=(
            "Dump root (contains data/) or the data/ directory itself. "
            f"Default: {DEFAULT_DS}"
        ),
    )
    p.add_argument(
        "--ds-local",
        type=Path,
        default=None,
        help="Optional diskcache dir (omit to read the raw dump).",
    )
    p.add_argument("--batch-size", type=int, default=2, help="DataLoader batch size")
    p.add_argument("--num-batches", type=int, default=1, help="How many batches to print")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap dataset length via Subset (still indexes all routes at init)",
    )
    p.add_argument(
        "--min-objs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Keep samples with n_objs>=N (boxes minus ego). Scans labels in "
            "iterdir order and stops after enough hits for --num-batches. "
            "Without this, the first route is often n_objs=1."
        ),
    )
    p.add_argument(
        "--require-class",
        type=str,
        default=None,
        metavar="CLS",
        help=(
            "Keep samples whose current-frame boxes contain CLS in x_objs "
            "(for PDD signs: affects_ego and within 30 m). Example: 2.5. "
            "Needed because PlanTDataset skips frames 0–4, where signs often live."
        ),
    )
    p.add_argument(
        "--route",
        type=str,
        default=None,
        help=(
            "Restrict to one route (substring match on the boxes path). "
            "Does not by itself guarantee a sign in the training window."
        ),
    )
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--shuffle", action="store_true", help="Shuffle the loader")
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable geometric augment (default: off for deterministic debug)",
    )
    p.add_argument(
        "--filter-routes",
        action="store_true",
        help="Enable PlanTDataset results/slurm filtering (needs slurm logs)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    data_root = resolve_data_root(args.ds)
    pt = plan_t()
    os.chdir(pt)
    if str(pt) not in sys.path:
        sys.path.insert(0, str(pt))

    from dataset import PlanTDataset, generate_batch

    cfg = load_cfg(augment=args.augment, filter_routes=args.filter_routes)

    cache = None
    if args.ds_local is not None:
        from diskcache import Cache

        args.ds_local.mkdir(parents=True, exist_ok=True)
        cache = Cache(directory=str(args.ds_local), size_limit=int(50 * 1024**3))
        print(f"diskcache: {args.ds_local}")

    print(f"PlanT cwd:  {pt}")
    print(f"--ds:       {args.ds}")
    print(f"data root:  {data_root}")
    print(
        f"cfg:        filter_routes={cfg.model.training.filter_routes}  "
        f"augment={cfg.model.training.augment}  "
        f"augment_parked={cfg.model.training.augment_parked}  "
        f"input_bev={cfg.model.training.input_bev}"
    )

    dataset = PlanTDataset(str(data_root), cfg, shared_dict=cache)
    n = len(dataset)
    print(f"dataset len: {n}")
    if n == 0:
        print("ERROR: empty dataset (check --ds / --filter-routes)", file=sys.stderr)
        return 1

    candidates = list(range(n))
    if args.route is not None:
        key = str(args.route).rstrip("/")
        route_hits = [
            i
            for i in candidates
            if key in _label_path_str(dataset.labels[i][0])
        ]
        print(
            f"route:       {key!r}  hits={len(route_hits)}/{n}  "
            f"first_index={route_hits[0] if route_hits else None}"
        )
        if not route_hits:
            print(f"ERROR: no samples matching --route {key!r}", file=sys.stderr)
            return 1
        candidates = route_hits

    need_scan = args.min_objs is not None or args.require_class is not None
    if need_scan:
        need = max(1, int(args.batch_size) * int(args.num_batches))
        if args.max_samples is not None:
            need = max(1, min(need, int(args.max_samples)))
        hits: list[int] = []
        scanned = 0
        for i in candidates:
            scanned += 1
            label_path = _label_path_str(dataset.labels[i][0])
            if args.min_objs is not None:
                if n_objs_from_boxes_file(label_path) < int(args.min_objs):
                    continue
            if args.require_class is not None:
                if not boxes_has_class(label_path, args.require_class):
                    continue
            hits.append(i)
            if len(hits) >= need:
                break
        filters = []
        if args.min_objs is not None:
            filters.append(f"n_objs>={args.min_objs}")
        if args.require_class is not None:
            filters.append(f"class={args.require_class}")
        print(
            f"filter:      {' AND '.join(filters)}  hits={len(hits)}/{need}  "
            f"scanned={scanned}/{len(candidates)}  "
            f"first_index={hits[0] if hits else None}"
        )
        if hits:
            print(f"             first hit {dataset.labels[hits[0]][0]}")
        if not hits:
            print(
                f"ERROR: no samples matching {' AND '.join(filters)} "
                f"(scanned {scanned} labels)",
                file=sys.stderr,
            )
            return 1
        dataset = Subset(dataset, hits)
    elif args.route is not None:
        if args.max_samples is not None:
            candidates = candidates[: max(0, int(args.max_samples))]
        dataset = Subset(dataset, candidates)
        print(f"subset:      {len(candidates)} samples from --route")
    elif args.max_samples is not None:
        n_use = max(0, min(int(args.max_samples), n))
        dataset = Subset(dataset, list(range(n_use)))
        print(f"subset:      first {n_use} samples")

    raw_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    idx0 = dataset.indices[0] if isinstance(dataset, Subset) else 0
    label0 = raw_ds.labels[idx0][0]
    if isinstance(label0, bytes):
        label0 = label0.decode()
    print_raw_sample(dataset[0], index=idx0, label_path=label0)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=generate_batch,
        pin_memory=False,
    )
    print(
        f"\nDataLoader batch_size={args.batch_size}  "
        f"num_workers={args.num_workers}  shuffle={args.shuffle}  "
        f"len={len(loader)}"
    )

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        print_batch(batch, batch_i=i)

    if cache is not None:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
