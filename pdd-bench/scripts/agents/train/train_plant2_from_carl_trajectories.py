#!/usr/bin/env python3
"""
Supervised fine-tuning of Plant2 (PlanT/HFLM) on CaRL-generated MetaDrive trajectories.

Trajectories are .pt files saved by:
  scripts/agents/train/collect_metadrive_carl_plant2_trajectories.py

Each step must contain:
  - plant2_batch : dict from metadrive_obs_to_plant2_batch

Supervision targets:
  pred_path (B, 20, 2) vs route_original (B, 20, 2)  — planned route (same as original PlanT)
  pred_wps  (B,  N, 2) vs actual_wps_ego (B, 4, 2)   — CaRL expert's ACTUAL future trajectory
                                                         converted from world frame to ego frame
                                                         using ego_pos_world_before + ego_heading_before

Training is done with pytorch-lightning. True batch training is enabled by a one-line fix
to model.py that generalises embedding[batch_idxs] to batched 3-D pools.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_sdc_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "pdd-bench").is_dir() and (parent / "metadrive").is_dir():
            return parent
    raise RuntimeError("Could not locate SDC root (expected pdd-bench and metadrive)")


FILE_PATH = Path(__file__).resolve()
SDC_ROOT = _find_sdc_root(FILE_PATH)
PDD_BENCH_DIR = SDC_ROOT / "pdd-bench"
METADRIVE_DIR = SDC_ROOT / "metadrive"
PLANT2_DIR = SDC_ROOT / "plant2"
PLANT_PLAN_T_DIR = PLANT2_DIR / "PlanT"

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR, PLANT_PLAN_T_DIR, PLANT2_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

TARGET_SPEEDS = np.array([0, 0.025, 0.05472609, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 20.0], dtype=np.float64)


def _mock_carla_modules():
    import unittest.mock as _mock
    for mod_name in ("carla", "agents", "agents.navigation",
                     "agents.navigation.global_route_planner"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mock.MagicMock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_plant_model(checkpoint_path: str, plant_planT_path: str, device: str = "cpu"):
    """Load HFLM from config + checkpoint."""
    import yaml
    _mock_carla_modules()

    model_yaml = os.path.join(plant_planT_path, "config", "model", "PlanT.yaml")
    if not os.path.isfile(model_yaml):
        raise FileNotFoundError(f"PlanT config not found: {model_yaml}")
    with open(model_yaml) as f:
        plnt = yaml.safe_load(f)

    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self.get(name)
            if isinstance(value, dict) and not isinstance(value, DictAsMember):
                return DictAsMember(value)
            return value

    config_all = DictAsMember({"model": plnt})

    if plant_planT_path not in sys.path:
        sys.path.insert(0, plant_planT_path)
    elif sys.path[0] != plant_planT_path:
        sys.path.remove(plant_planT_path)
        sys.path.insert(0, plant_planT_path)
    from model import HFLM  # type: ignore

    net = HFLM(config_all.model.network, config_all)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint contains no state_dict")
    if list(sd.keys())[0].startswith("model."):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    print(net.wp_rep) # path+wps
    print(net.wp_gen) # linear
    net = net.to(device)
    return net, config_all


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardedCarlPlantDataset(Dataset):
    """
    Fast-loading Dataset that reads from pre-sharded .pt files produced by
    scripts/agents/train/shard_plant2_pt.py. Each shard contains stacked numpy arrays for 256 steps.
    Shards are loaded on demand — no full dataset in RAM.

    For DDP, pass rank/world_size so each process owns a distinct slice of shards.
    Sequential access within a rank means the 1-shard cache is always hot.
    """

    def __init__(self, shard_dir: str, steps_per_shard: int = 256,
                 rank: int = 0, world_size: int = 1, val_split: float = 0.1,
                 split: str = "train"):
        self.shard_dir = Path(shard_dir)
        all_shards = sorted(self.shard_dir.glob("shard_*.pt"))
        if not all_shards:
            raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")

        # Drop the last shard of each node prefix — it may be truncated.
        import re
        prefix_last: Dict[str, int] = {}
        for i, p in enumerate(all_shards):
            m = re.match(r"shard_(n\d+)_", p.name)
            prefix = m.group(1) if m else "default"
            prefix_last[prefix] = i
        drop = set(prefix_last.values())
        full_shards = [p for i, p in enumerate(all_shards) if i not in drop]

        # Train/val split — deterministic, based on sorted order.
        n_val = max(1, int(len(full_shards) * val_split))
        if split == "val":
            full_shards = full_shards[-n_val:]
            rank, world_size = 0, 1   # val runs on all ranks but sampler not split
        else:
            full_shards = full_shards[:-n_val]

        # Split shards evenly across DDP ranks so each rank reads a disjoint
        # sequential slice — keeps the 1-shard cache hot and avoids NFS thrashing.
        self._shards = full_shards[rank::world_size]

        self._sps = steps_per_shard
        self._offsets = [i * steps_per_shard for i in range(len(self._shards))]
        self._total = len(self._shards) * steps_per_shard
        self._cache_idx: int = -1
        self._cache_data: Dict[str, Any] = {}
        print(f"[rank {rank}/{world_size}][{split}] ShardedCarlPlantDataset: "
              f"{len(self._shards)} shards  {self._total} steps  "
              f"(dropped {len(drop)} tail shards)", flush=True)

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, shard_idx: int) -> None:
        if self._cache_idx == shard_idx:
            return
        self._cache_data = torch.load(
            self._shards[shard_idx], map_location="cpu", weights_only=False
        )
        self._cache_idx = shard_idx

    def _find_shard(self, global_idx: int):
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, global_idx - self._offsets[lo]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, local_idx = self._find_shard(idx)
        self._load_shard(shard_idx)
        d = self._cache_data

        # Reconstruct plant2_batch dict from stacked keys
        plant2_batch: Dict[str, Any] = {}
        for k, v in d.items():
            if k in ("actual_wps_ego", "step_idx"):
                continue
            plant2_batch[k] = v[local_idx]

        sample = {
            "plant2_batch": plant2_batch,
            "actual_wps_ego": d["actual_wps_ego"][local_idx],
            "step_idx": d.get("step_idx", np.zeros(1))[local_idx],
        }
        return sample


class ShardShuffleSampler(Sampler):
    """
    Shuffles shards between epochs but accesses steps within each shard
    sequentially — keeps the 1-shard cache hot while adding inter-epoch variety.
    """
    def __init__(self, dataset: "ShardedCarlPlantDataset", seed: int = 0):
        self._ds = dataset
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._ds._total

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self._seed + self._epoch)
        shard_order = torch.randperm(len(self._ds._shards), generator=rng).tolist()
        sps = self._ds._sps
        for shard_idx in shard_order:
            base = self._ds._offsets[shard_idx]
            yield from range(base, base + sps)



class CarlPlantTrajectoryDataset(Dataset):
    """
    Iterates all steps from CaRL/Plant2 trajectory .pt files.

    Each sample contains:
      "plant2_batch"    — raw dict from metadrive_obs_to_plant2_batch
      "actual_wps_ego"  — (4, 2) CaRL expert's actual future positions in ego frame,
                          computed from ego_pos_world_future_4 + ego_pos_world_before
                          + ego_heading_before using the same math as
                          get_route_points_ego_frame (plant_policy.py:129-132).
                          This is the real supervision target for pred_wps.
    """

    def __init__(self, data_dir: str):
        self.samples: List[Dict[str, Any]] = []
        data_path = Path(data_dir)
        files = sorted(data_path.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        for f in files:
            ep = torch.load(f, weights_only=False, map_location="cpu")
            for step in ep.get("steps", []):
                plant2_batch = step.get("plant2_batch")

                sample: Dict[str, Any] = {"plant2_batch": plant2_batch, "step_idx": step.get("step_idx")}

                # Convert saved world-frame future positions to ego frame.
                # Prefer stride-3 (~0.3 s ≈ 4 Hz) if available; fall back to consecutive
                # (0.1 s, 10 Hz).  The stride-3 spacing matches the HFLM nav_planner
                # calibration (_desired_speed_from_waypoints * 4 assumes ~0.25 s/wp).
                pos_before  = step.get("ego_pos_world_before")
                heading     = step.get("ego_heading_before")
                pos_future4 = step.get("ego_pos_world_future_4_s3")
                p0  = np.asarray(pos_before,  dtype=np.float64)[:2]
                pts = np.asarray(pos_future4, dtype=np.float64)[:, :2]  # (4, 2)
                h   = float(heading)
                dx, dy  = pts[:, 0] - p0[0], pts[:, 1] - p0[1]
                fwd   =  dx * np.cos(h) + dy * np.sin(h)
                right =  dx * np.sin(h) - dy * np.cos(h)
                sample["actual_wps_ego"] = np.stack([fwd, right], axis=1).astype(np.float32)  # (4, 2)

                self.samples.append(sample)
        if not self.samples:
            raise RuntimeError(f"No usable steps found in {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx].copy()


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_plant2_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack plant2_batch dicts from individual samples into batch tensors.

    Resulting shapes (B = batch size, pool = max_objects+1):
      idxs           (B, maxseq)      — pool indices per sequence position
      x_objs         (B, pool, 7)     — object feature pool
      route_original (B, 20, 2)
      speed_limit    (B,)             — long, 0-3
      BEV            (B, 3, H, W)     — optional
      input_ego_speed (B, 1, 1)       — optional (unused when input_ego_speed=False)
      target_speed    (B, 1)          — ego speed m/s for pred_speed supervision
    """
    plant_batches = [b["plant2_batch"] for b in batch]
    step_idxs = [torch.as_tensor(b["step_idx"]) for b in batch]
    out_batch: Dict[str, Any] = {}
    keys = set().union(*[pb.keys() for pb in plant_batches])
    keys.add("step_idx")
    # Infer pool size from first sample's idxs
    inferred_max = 30
    for pb in plant_batches:
        idxs = pb.get("idxs")
        if idxs is None:
            continue
        a = np.asarray(idxs)
        if a.ndim == 2 and a.shape[0] == 1:
            a = a.squeeze(0)
        if a.ndim == 1 and a.shape[0] > 0:
            inferred_max = int(a.shape[0])
            break
    pool_size = inferred_max + 1  # index 0 is the padding token

    for k in keys:
        vals = [pb.get(k) for pb in plant_batches]
        if all(v is None for v in vals):
            out_batch[k] = None
            continue
        if any(v is None for v in vals):
            raise ValueError(f"Key '{k}' is None for some samples but not others; cannot stack.")

        arrs = []
        for v in vals:
            a = np.asarray(v)
            if k == "idxs":
                # (1, maxseq) → (maxseq,) so torch.stack gives (B, maxseq)
                if a.ndim == 2 and a.shape[0] == 1:
                    a = a.squeeze(0)
            elif k == "x_objs":
                # (pool, 7) — pad/trim to fixed pool_size
                if a.ndim == 3 and a.shape[0] == 1:
                    a = a.squeeze(0)
                if a.ndim != 2 or a.shape[1] != 7:
                    raise ValueError(f"x_objs must be (N, 7); got {a.shape}")
                if a.shape[0] < pool_size:
                    pad = np.zeros((pool_size - a.shape[0], 7), dtype=a.dtype)
                    a = np.concatenate([a, pad], axis=0)
                elif a.shape[0] > pool_size:
                    a = a[:pool_size]
            elif k == "route_original":
                # (1, 20, 2) → (20, 2) so torch.stack gives (B, 20, 2)
                if a.ndim == 3 and a.shape[0] == 1:
                    a = a.squeeze(0)
            elif k == "speed_limit":
                # ensure 1-D (1,) so torch.stack gives (B, 1); _normalize_batch flattens later
                a = a.reshape(1)
            elif k == "BEV":
                # (1, 3, H, W) → (3, H, W) so torch.stack gives (B, 3, H, W)
                if a.ndim == 4 and a.shape[0] == 1:
                    a = a.squeeze(0)
            elif k == "target_speed":
                # (1, 1) or (1,) → (1,) so stack -> (B, 1)
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                if a.size != 1:
                    a = a.reshape(-1)[:1]
            arrs.append(torch.as_tensor(a))
        out_batch[k] = torch.stack(arrs, dim=0)

    # Stack actual_wps_ego if present in all samples
    wps_list = [b.get("actual_wps_ego") for b in batch]
    if all(w is not None for w in wps_list):
        actual_wps = torch.stack([torch.as_tensor(np.asarray(w, dtype=np.float32)) for w in wps_list])
    else:
        actual_wps = None

    return {"plant2_batch": out_batch, "actual_wps_ego": actual_wps, "step_idx": torch.stack(step_idxs, dim=0)}


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


class LitHFLMSupervised(pl.LightningModule):
    """
    PyTorch-Lightning wrapper for supervised fine-tuning of PlanT/HFLM.

    Training objective:
      - L1 on waypoints
      - standard CrossEntropy on pred_speed vs discretized target_speed bins

    Expects batches produced by collate_plant2_batch:
      batch["plant2_batch"]["target_speed"] (B, 1) — optional; ego speed m/s for speed loss
    """

    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-5,
        grad_clip: float = 1.0,
        speed_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.model = net
        self.lr = lr
        self.grad_clip = grad_clip
        self.speed_loss_weight = speed_loss_weight
        self.criterion = nn.L1Loss()
        self.criterion_speed = nn.CrossEntropyLoss(weight=torch.tensor([15, 15, 10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        self._target_speeds = torch.tensor(TARGET_SPEEDS, dtype=torch.float32)

    @staticmethod
    def get_two_hot_encoding(
        target_speed: torch.Tensor,
        config_target_speeds: torch.Tensor,
        brake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Two-hot encoding as used in the original PlanT lit_module.py.
        target_speed:        (B,)  – ego speed in m/s, must be >= 0
        config_target_speeds:(C,)  – sorted bin centres
        brake:               (B,)  – bool, True → force speed=0 bin
        returns:             (B, C) soft label distribution
        """
        if torch.any(target_speed < 0):
            raise ValueError("Target speed value must be non-negative for two-hot encoding.")

        labels = torch.zeros(target_speed.shape[0], len(config_target_speeds), device=target_speed.device)

        diffs = (config_target_speeds > target_speed[:, None]).float()   # (B, C)
        vals, idxs = diffs.max(dim=1)

        upper_ind = idxs
        lower_ind = (idxs - 1)
        upper_val = config_target_speeds[upper_ind]
        lower_val = config_target_speeds[lower_ind]

        denom = (upper_val - lower_val)
        lower_weight = (upper_val - target_speed) / denom
        upper_weight = (target_speed - lower_val) / denom

        labels[torch.arange(target_speed.shape[0]), lower_ind] = lower_weight
        labels[torch.arange(target_speed.shape[0]), upper_ind] = upper_weight

        # Rows where no bin centre exceeds target_speed → clip to last bin
        labels[torch.logical_and(vals == 0, ~brake)] = 0
        labels[torch.logical_and(vals == 0, ~brake), -1] = 1.0

        # Brake rows → all mass on bin 0
        labels[brake] = 0
        labels[brake, 0] = 1.0

        return labels

    def forward(self, batch, step_idx):
        return self.model(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def on_after_backward(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    @staticmethod
    def _normalize_batch(raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix tensor shapes in a stacked plant2_batch dict so HFLM.forward() can
        consume the full batch directly (no per-sample loop required).

        Key shape contracts expected by HFLM:
          idxs           (B, maxseq)   — already correct from collate
          x_objs         (B, pool, 7)  — already correct; model.py handles 3-D pool
          route_original (B, 20, 2)    — model flattens to (B, 40)
          speed_limit    (B,)          — Embedding needs 1-D indices
          BEV            (B, 3, H, W)  — collate may produce (B, 1, 3, H, W)
        """
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if v is None:
                out[k] = None
                continue
            if k == "speed_limit":
                # collate produces (B, 1); Embedding needs (B,)
                out[k] = v.view(-1).long()
            elif k == "idxs":
                out[k] = v.long()
            elif k == "BEV" and v.dim() == 5 and v.shape[1] == 1:
                out[k] = v.squeeze(1)       # (B, 1, 3, H, W) → (B, 3, H, W)
            else:
                out[k] = v
        out["y_objs"] = None                # disable object-forecasting head
        return out

    def training_step(self, batch, batch_idx):
        log_steps = int(getattr(self, "_log_every_n_steps", 100))

        raw_plant = dict(batch["plant2_batch"])
        target_speed_t = raw_plant.pop("target_speed", None)
        plant_input = self._normalize_batch(raw_plant)
        # route = plant_input["route_original"]               # (B, 20, 2) — planned route
        step_idx = batch["step_idx"]
        _, _, pred_plan, _ = self(plant_input, step_idx)
        pred_path, pred_wps, pred_speed = pred_plan

        losses = {}
        if pred_wps is not None:
            losses["loss_wps"] = self.criterion(pred_wps[:, :batch["actual_wps_ego"].shape[1], :], batch["actual_wps_ego"])

        if pred_speed is not None and target_speed_t is not None:
            ps = pred_speed
            if ps.dim() == 3:
                ps = ps.mean(dim=1)
            ts = target_speed_t.to(ps.device).float().view(-1)          # (B,)
            tsv = self._target_speeds.to(ps.device)
            brake = torch.zeros_like(ts, dtype=torch.bool)               # no forced-brake flag
            twohot_targs = self.get_two_hot_encoding(ts, tsv, brake)     # (B, C)
            losses["loss_speed"] = self.criterion_speed(ps, twohot_targs)

        if not losses:
            raise RuntimeError("HFLM produced no usable predictions for loss (pred_wps / pred_speed).")

        loss = losses["loss_wps"] + self.speed_loss_weight * losses.get("loss_speed", 0.0)
        B = step_idx.shape[0]
        # Epoch means (CSV/TB): one row per epoch for these keys.
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        for name, val in losses.items():
            self.log(f"train/{name}", val, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        # Step snapshot (not every step): every `log_steps` batches only.
        if log_steps > 0 and (batch_idx % log_steps == 0):
            self.log(
                "train/loss_step",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=B,
            )
            # stdout (e.g. tail -f train_console.log): loggers alone do not print here
            if self.trainer.is_global_zero:
                print(
                    f"[train] epoch {int(self.current_epoch)} batch {batch_idx} "
                    f"loss={loss.detach().item():.4f}",
                    flush=True,
                )
        return loss

    def validation_step(self, batch, batch_idx):
        raw_plant = dict(batch["plant2_batch"])
        target_speed_t = raw_plant.pop("target_speed", None)
        plant_input = self._normalize_batch(raw_plant)
        step_idx = batch["step_idx"]
        _, _, pred_plan, _ = self(plant_input, step_idx)
        pred_path, pred_wps, pred_speed = pred_plan

        losses = {}
        if pred_wps is not None:
            losses["loss_wps"] = self.criterion(pred_wps[:, :batch["actual_wps_ego"].shape[1], :], batch["actual_wps_ego"])
        if pred_speed is not None and target_speed_t is not None:
            ps = pred_speed
            if ps.dim() == 3:
                ps = ps.mean(dim=1)
            ts = target_speed_t.to(ps.device).float().view(-1)
            tsv = self._target_speeds.to(ps.device)
            brake = torch.zeros_like(ts, dtype=torch.bool)
            twohot_targs = self.get_two_hot_encoding(ts, tsv, brake)
            losses["loss_speed"] = self.criterion_speed(ps, twohot_targs)

        if not losses:
            return
        loss = losses["loss_wps"] + self.speed_loss_weight * losses.get("loss_speed", 0.0)
        B = step_idx.shape[0]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=B)
        return loss

    def on_validation_epoch_end(self):
        cm = self.trainer.callback_metrics
        if "val/loss" in cm and self.trainer.is_global_zero:
            val = cm["val/loss"]
            val = val.item() if hasattr(val, "item") else float(val)
            print(f"[val]   epoch {int(self.current_epoch)} loss={val:.4f}", flush=True)


        cm = self.trainer.callback_metrics
        for key in ("train/loss_epoch", "train/loss"):
            if key in cm:
                v = cm[key]
                val = v.item() if hasattr(v, "item") else float(v)
                if self.trainer.is_global_zero:
                    print(
                        f"[train] epoch {int(self.current_epoch)} end  {key}={val:.4f}",
                        flush=True,
                    )
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Supervised Plant2 training on CaRL MetaDrive trajectories (L1 loss)."
    )
    parser.add_argument("--data-dir", type=str,
                        default=str(PDD_BENCH_DIR / "data"),
                        help="Directory with .pt trajectory files or shards")
    parser.add_argument("--sharded", action="store_true",
                        help="Use ShardedCarlPlantDataset (shard_*.pt files) instead of per-episode .pt files")
    parser.add_argument("--checkpoint_file", type=str, required=True,
                        help="Initial Plant2 (PlanT) checkpoint (.pt or lightning .ckpt)")
    parser.add_argument("--plant_planT_path", type=str, default=str(PLANT_PLAN_T_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated CUDA indices for Lightning Trainer devices (e.g. 0,1,2,3). "
        "Default 1. When --sharded runs multi-GPU DDP, all visible GPUs are used instead.",
    )
    parser.add_argument("--speed-loss-weight", type=float, default=1.0, help="Weight for pred_speed vs target_speed (two-hot CE)")
    parser.add_argument(
        "--no-csv-log",
        action="store_true",
        help="Disable CSV metrics (metrics.csv) under output_dir",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logs under output_dir",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log step metrics (CSV/TensorBoard) at most every N training steps; "
        "epoch means are still logged once per epoch.",
    )
    parser.add_argument(
        "--ckpt-every-n-epochs",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs (default 1)",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show Lightning tqdm (very noisy when stdout is tee'd to a log file; default off)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or str(PDD_BENCH_DIR / "outputs" / "plant2_supervised_20epochs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Supervised Plant2 training — Lightning / true batch")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Checkpoint: {args.checkpoint_file}")
    print(f"  Output dir: {out_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {device}")
    print(f"  CSV log:    {not args.no_csv_log}  -> {out_dir}/metrics_csv/")
    print(f"  TB log:     {not args.no_tensorboard}  -> {out_dir}/tensorboard/")
    print(
        f"  Logging:    train/loss_step every {args.log_every_n_steps} batches; "
        "train/loss (epoch mean) each epoch"
    )
    print("=" * 70)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_ddp = args.sharded and n_gpus > 1
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = n_gpus if use_ddp else 1

    if use_ddp:
        n_trainer_devices = n_gpus
    elif args.devices:
        n_trainer_devices = max(
            1, len([p for p in args.devices.split(",") if p.strip() != ""])
        )
    else:
        n_trainer_devices = 1
    print(f"  Lightning devices: {n_trainer_devices}  (sharded DDP={use_ddp})", flush=True)

    sharded_dataset = (
        ShardedCarlPlantDataset(args.data_dir, rank=rank, world_size=world_size, split="train")
        if args.sharded
        else None
    )
    dataset = sharded_dataset if args.sharded else CarlPlantTrajectoryDataset(args.data_dir)
    shard_sampler = ShardShuffleSampler(sharded_dataset, seed=42) if args.sharded else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=shard_sampler,
        num_workers=1,         # prefetch next batch while GPU computes
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_plant2_batch,
    )

    val_dataset = (
        ShardedCarlPlantDataset(args.data_dir, rank=0, world_size=1, split="val")
        if args.sharded
        else None
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            drop_last=False,
            collate_fn=collate_plant2_batch,
        )
        if val_dataset is not None
        else None
    )


    net, _cfg = load_plant_model(args.checkpoint_file, args.plant_planT_path, device="cpu")
    lit_model = LitHFLMSupervised(
        net, lr=args.lr, speed_loss_weight=args.speed_loss_weight
    )
    lit_model._log_every_n_steps = max(1, args.log_every_n_steps)

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="plant2_2nd_ep{epoch:02d}_valloss{val/loss:.4f}",
        monitor="val/loss" if val_loader is not None else None,
        mode="min",
        save_last=True,
        save_top_k=3,
        every_n_epochs=1,
    )

    class _SamplerEpochCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if shard_sampler is not None:
                shard_sampler.set_epoch(trainer.current_epoch)

    callbacks = [ckpt_cb, _SamplerEpochCallback()]
    loggers = []
    if not args.no_csv_log:
        loggers.append(CSVLogger(save_dir=out_dir, name="metrics_csv"))
    if not args.no_tensorboard:
        loggers.append(TensorBoardLogger(save_dir=out_dir, name="tensorboard"))
    if not loggers:
        loggers = None  # type: ignore

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=n_trainer_devices,
        strategy="ddp_find_unused_parameters_true" if use_ddp else "auto",
        use_distributed_sampler=False,   # we partition shards by rank manually
        gradient_clip_val=1.0,
        default_root_dir=out_dir,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=args.progress_bar,
        log_every_n_steps=max(1, args.log_every_n_steps),
    )

    trainer.fit(lit_model, loader, val_dataloaders=val_loader)

    # Save final model only from master rank to avoid duplicate writes
    if trainer.is_global_zero:
        final_path = os.path.join(out_dir, "plant2_supervised_2nd_final.pt")
        torch.save({"model_state_dict": lit_model.model.state_dict()}, final_path)
        print(f"\nDone. Final model: {final_path}")

    # Plot epoch-level loss from trainer logs
    losses = [
        v.item()
        for k, v in trainer.callback_metrics.items()
        if "loss_epoch" in k or k == "train/loss"
    ]
    if losses:
        plt.figure()
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("L1 loss")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
