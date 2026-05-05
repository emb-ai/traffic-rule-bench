#!/usr/bin/env python3
"""
Flatten per-episode Plant2 .pt files produced by repack_expert_picks_jsonl.py
into fixed-size shards for fast DataLoader reading.

Each shard is a single .pt file containing N_STEPS_PER_SHARD steps with
pre-stacked numpy arrays:

  idxs              (N, maxseq)       int32
  x_objs            (N, pool, 7)      float32  ← includes injected sign types
  route_original    (N, 20, 2)        float32
  speed_limit       (N,)              int64
  BEV               (N, 3, H, W)      float32  (present if episodes contain BEV)
  input_ego_speed   (N, 1, 1)         float32  (optional)
  target_speed      (N, 1)            float32
  actual_wps_ego    (N, 4, 2)         float32  ← stride-3 future waypoints in ego frame
  step_idx          (N,)              float32

Shards are shuffled at the step level before writing (so training needs no
within-shard shuffle, only cross-shard shuffle via DataLoader).

Usage:
  python shard_plant2_pt.py \\
      --input-dir  /path/to/plant2_pt_expert \\
      --output-dir /path/to/plant2_shards_expert \\
      --steps-per-shard 1024 \\
      --shuffle

A companion ShardedPlantDataset class is included at the bottom of this file;
pass shard_dir to it and use it as a drop-in replacement for
CarlPlantTrajectoryDataset in train_plant2_from_carl_trajectories.py (prefer ShardedCarlPlantDataset there).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Step extraction helpers
# ---------------------------------------------------------------------------

def _compute_actual_wps_ego(step: Dict[str, Any]) -> Optional[np.ndarray]:
    """Convert stride-3 world-frame future positions to ego frame (4, 2)."""
    pos_before = step.get("ego_pos_world_before")
    heading = step.get("ego_heading_before")
    pos_future4 = step.get("ego_pos_world_future_4_s3")
    if pos_before is None or heading is None or pos_future4 is None:
        return None
    p0 = np.asarray(pos_before, dtype=np.float64)[:2]
    pts = np.asarray(pos_future4, dtype=np.float64)[:, :2]
    h = float(heading)
    dx, dy = pts[:, 0] - p0[0], pts[:, 1] - p0[1]
    fwd = dx * np.cos(h) + dy * np.sin(h)
    right = dx * np.sin(h) - dy * np.cos(h)
    return np.stack([fwd, right], axis=1).astype(np.float32)


def _extract_step_arrays(step: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Pull all fields needed for training out of a single step dict."""
    pb = step.get("plant2_batch")
    if pb is None:
        return None

    actual_wps = _compute_actual_wps_ego(step)
    if actual_wps is None:
        return None

    def _to_np(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.cpu().numpy()
        return np.asarray(v)

    idxs = _to_np(pb.get("idxs"))
    x_objs = _to_np(pb.get("x_objs"))
    route = _to_np(pb.get("route_original"))
    speed_limit = _to_np(pb.get("speed_limit"))
    target_speed = _to_np(pb.get("target_speed"))
    step_idx = _to_np(step.get("step_idx"))
    bev = _to_np(pb.get("BEV"))
    ego_speed_input = _to_np(pb.get("input_ego_speed"))

    if any(v is None for v in [idxs, x_objs, route, speed_limit, target_speed]):
        return None

    # Normalise shapes to 1-D along the "sample" axis (drop leading batch dim)
    def _squeeze_leading(a, ndim_want):
        while a.ndim > ndim_want and a.shape[0] == 1:
            a = a.squeeze(0)
        return a

    idxs = _squeeze_leading(idxs, 1)           # (maxseq,)
    x_objs = _squeeze_leading(x_objs, 2)        # (pool, 7)
    route = _squeeze_leading(route, 2)           # (20, 2)
    speed_limit = speed_limit.reshape(())        # scalar
    target_speed = target_speed.reshape(1).astype(np.float32)  # (1,)
    step_idx_val = float(step_idx.flat[0]) if step_idx is not None else 0.0

    out: Dict[str, np.ndarray] = {
        "idxs": idxs.astype(np.int32),
        "x_objs": x_objs.astype(np.float32),
        "route_original": route.astype(np.float32),
        "speed_limit": np.int64(int(speed_limit.flat[0])),
        "target_speed": target_speed,
        "actual_wps_ego": actual_wps,
        "step_idx": np.float32(step_idx_val),
    }

    if bev is not None:
        bev = _squeeze_leading(bev, 3)           # (3, H, W)
        out["BEV"] = bev.astype(np.float32)

    if ego_speed_input is not None:
        out["input_ego_speed"] = ego_speed_input.astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def _stack_field(arrays: List[np.ndarray]) -> np.ndarray:
    return np.stack(arrays, axis=0)


def _write_shard(
    steps: List[Dict[str, np.ndarray]],
    out_path: Path,
) -> int:
    """Stack per-step dicts and save as a single .pt file. Returns n steps."""
    if not steps:
        return 0
    keys = list(steps[0].keys())
    shard: Dict[str, Any] = {}
    for k in keys:
        vals = [s[k] for s in steps if k in s]
        if not vals:
            continue
        if isinstance(vals[0], np.ndarray):
            shard[k] = _stack_field(vals)
        else:
            # scalar (int64 / float32)
            shard[k] = np.array(vals)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, out_path)
    return len(steps)


def _iter_steps(input_dir: Optional[Path], file_list: Optional[Path] = None) -> Iterator[Dict[str, np.ndarray]]:
    """Yield extracted step dicts from all .pt files in input_dir or file_list."""
    if file_list is not None:
        pt_files = [Path(l.strip()) for l in file_list.open() if l.strip()]
    else:
        pt_files = sorted(
            f for f in input_dir.glob("*.pt")
            if "__" in f.stem  # skip old-named files ({scene_uid}_plant2.pt)
        )
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found")
    for f in pt_files:
        try:
            ep = torch.load(f, weights_only=False, map_location="cpu")
        except Exception as exc:
            print(f"[warn] cannot load {f.name}: {exc}", file=sys.stderr)
            continue
        for step in ep.get("steps", []):
            arr = _extract_step_arrays(step)
            if arr is not None:
                yield arr


def build_shards(
    input_dir: Path,
    output_dir: Path,
    steps_per_shard: int = 256,
    shuffle: bool = True,
    seed: int = 42,
    file_list: Optional[Path] = None,
    shard_prefix: str = "",
    shuffle_buffer: int = 8192,
) -> None:
    """
    Stream through .pt files and write fixed-size shards without loading
    everything into memory. Uses a shuffle buffer for local randomisation.
    """
    src = str(file_list) if file_list else str(input_dir)
    print(f"Sharding from {src} …", flush=True)
    print(f"  steps_per_shard={steps_per_shard}  shuffle_buffer={shuffle_buffer}  prefix='{shard_prefix}'", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    buffer: List[Dict[str, np.ndarray]] = []
    shard_idx = 0
    total_written = 0

    def flush_shard(steps: List[Dict[str, np.ndarray]]) -> None:
        nonlocal shard_idx, total_written
        out_path = output_dir / f"shard_{shard_prefix}{shard_idx:06d}.pt"
        n = _write_shard(steps, out_path)
        total_written += n
        shard_idx += 1
        if shard_idx % 50 == 0:
            print(f"  [{shard_idx}] {out_path.name}  steps={n}", flush=True)

    for step in _iter_steps(input_dir, file_list):
        buffer.append(step)
        if len(buffer) >= shuffle_buffer:
            if shuffle:
                rng.shuffle(buffer)
            while len(buffer) >= steps_per_shard:
                flush_shard(buffer[:steps_per_shard])
                buffer = buffer[steps_per_shard:]

    # Flush remaining
    if shuffle and buffer:
        rng.shuffle(buffer)
    while len(buffer) >= steps_per_shard:
        flush_shard(buffer[:steps_per_shard])
        buffer = buffer[steps_per_shard:]
    if buffer:
        flush_shard(buffer)

    index = {
        "n_shards": shard_idx,
        "steps_per_shard": steps_per_shard,
        "total_steps": total_written,
        "shuffle": shuffle,
        "seed": seed,
        "prefix": shard_prefix,
    }
    torch.save(index, output_dir / f"index_{shard_prefix or 'all'}.pt")
    print(f"\nDone. {shard_idx} shards  {total_written} steps  → {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# Dataset for training (drop-in for CarlPlantTrajectoryDataset)
# ---------------------------------------------------------------------------

class ShardedPlantDataset(Dataset):
    """
    Fast-loading dataset that reads pre-sharded .pt files produced by
    build_shards(). Each shard is loaded on-demand and kept in a one-shard
    LRU cache to avoid repeated disk reads within a DataLoader epoch.

    Compatible with the collate_plant2_batch function in
    train_plant2_from_carl_trajectories.py.
    """

    def __init__(self, shard_dir: str) -> None:
        self.shard_dir = Path(shard_dir)
        index_path = self.shard_dir / "index.pt"
        if index_path.exists():
            idx = torch.load(index_path, weights_only=False)
            self._total = int(idx["total_steps"])
        else:
            idx = {}

        self._shards = sorted(self.shard_dir.glob("shard_*.pt"))
        if not self._shards:
            raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")

        # Build offset table: shard_offsets[i] = global index of first step in shard i
        self._offsets: List[int] = []
        self._shard_lengths: List[int] = []
        running = 0
        for sp in self._shards:
            try:
                s = torch.load(sp, weights_only=False, map_location="cpu")
                # Length from any array field
                n = int(next(iter(s.values())).shape[0])
            except Exception:
                n = 0
            self._offsets.append(running)
            self._shard_lengths.append(n)
            running += n
        self._total = running

        self._cache_idx: int = -1
        self._cache_shard: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, shard_i: int) -> Dict[str, np.ndarray]:
        if self._cache_idx == shard_i:
            return self._cache_shard  # type: ignore[return-value]
        self._cache_shard = torch.load(
            self._shards[shard_i], weights_only=False, map_location="cpu"
        )
        self._cache_idx = shard_i
        return self._cache_shard  # type: ignore[return-value]

    def _find_shard(self, global_idx: int) -> Tuple[int, int]:
        """Binary search → (shard_index, local_index)."""
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, global_idx - self._offsets[lo]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_i, local_i = self._find_shard(idx)
        shard = self._load_shard(shard_i)

        sample: Dict[str, Any] = {}
        plant2_batch: Dict[str, Any] = {}
        for k, arr in shard.items():
            if k in ("actual_wps_ego", "step_idx"):
                continue
            val = arr[local_i]
            if k in ("idxs", "x_objs", "route_original", "BEV", "input_ego_speed", "target_speed"):
                plant2_batch[k] = val
            elif k == "speed_limit":
                plant2_batch[k] = val

        sample["plant2_batch"] = plant2_batch
        if "actual_wps_ego" in shard:
            sample["actual_wps_ego"] = shard["actual_wps_ego"][local_i]
        if "step_idx" in shard:
            sample["step_idx"] = shard["step_idx"][local_i]
        return sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shard Plant2 episode .pt files into fixed-size shards."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory of per-episode .pt files (output of repack_expert_picks_jsonl.py).",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default=None,
        help="Text file with one .pt path per line (use instead of --input-dir for parallel sharding).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for shard_NNNNNN.pt files.",
    )
    parser.add_argument(
        "--steps-per-shard",
        type=int,
        default=1024,
        help="Number of steps per shard file (default 1024).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle all steps before sharding (default True).",
    )
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-prefix", type=str, default="", help="Prefix for shard filenames, e.g. 'n00_'")
    parser.add_argument("--shuffle-buffer", type=int, default=8192, help="Steps to buffer before shuffling/writing (controls RAM usage)")
    args = parser.parse_args()

    if args.file_list is None and args.input_dir is None:
        parser.error("Either --input-dir or --file-list is required.")

    build_shards(
        input_dir=Path(args.input_dir) if args.input_dir else None,
        output_dir=Path(args.output_dir),
        steps_per_shard=args.steps_per_shard,
        shuffle=args.shuffle,
        seed=args.seed,
        file_list=Path(args.file_list) if args.file_list else None,
        shard_prefix=args.shard_prefix,
        shuffle_buffer=args.shuffle_buffer,
    )


if __name__ == "__main__":
    main()
