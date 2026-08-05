"""PlanT2 fine-tune launch helpers."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from _env import default_ckpt0, hydra_escape, plan_t, resolve_python, shim_path


@dataclass
class FinetuneConfig:
    split: Path
    learning_rate: str
    checkpoint_addon: str
    cuda_device: str = "0"
    ds: Path | None = None
    ds_val: Path | None = None
    ds_local: Path | None = None
    cache_size_gb: int = 641
    batch_size: int = 1536
    num_workers: int = 4
    max_epochs: int = 30
    ckpt_every_n_epochs: int = 5
    lr_scheduler: str = "cosine_warmup"
    warmup_ratio: float = 0.1
    seed: int = 1
    resume_ckpt: Path | None = None
    augment: bool = False
    augment_parked: bool = False
    filter_routes: bool = True
    stop_speed_loss_weight: float | None = None
    extra_hydra: list[str] = field(default_factory=list)
    hydra_run_dir: Path | None = None
    python: Path | None = None
    shim: Path | None = None
    wandb_mode: str = "offline"

    def __post_init__(self) -> None:
        self.split = Path(self.split)
        self.ds = Path(self.ds or self.split / "train")
        self.ds_val = Path(self.ds_val or self.split / "val")
        self.ds_local = Path(self.ds_local or Path(f"/tmp/plant2_ds_cache_{self.seed}"))
        self.resume_ckpt = Path(self.resume_ckpt or default_ckpt0())
        self.python = Path(self.python or resolve_python())
        self.shim = Path(self.shim or shim_path())

    def validate(self) -> None:
        if "parallel300" in str(self.split) and os.environ.get("ALLOW_PARALLEL300") != "1":
            raise SystemExit(
                f"WARNING: SPLIT points at parallel300 tree: {self.split}\n"
                "Set ALLOW_PARALLEL300=1 to proceed."
            )
        for p, label in (
            (self.resume_ckpt, "CKPT"),
            (self.ds / "data", "DS/data"),
            (self.ds_val / "data", "DS_VAL/data"),
            (self.split / "split_meta.json", "split_meta.json"),
            (self.shim, "SHIM"),
        ):
            if not p.exists():
                raise SystemExit(f"ERROR: missing {label}: {p}")


def build_finetune_cmd(cfg: FinetuneConfig) -> list[str]:
    pt = plan_t()
    cfg.ds_local.mkdir(parents=True, exist_ok=True)
    (pt / "log").mkdir(parents=True, exist_ok=True)
    (pt / "checkpoints_ft").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ["SEED"] = str(cfg.seed)
    os.environ["CHECKPOINT_ADDON"] = cfg.checkpoint_addon
    os.environ["DS"] = str(cfg.ds)
    os.environ["DS_VAL"] = str(cfg.ds_val)
    os.environ["DS_LOCAL"] = str(cfg.ds_local)
    os.environ["CACHE_SIZE_GB"] = str(cfg.cache_size_gb)
    os.environ["CKPT_EVERY_N_EPOCHS"] = str(cfg.ckpt_every_n_epochs)
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device

    cmd = [
        str(cfg.python),
        "-u",
        str(cfg.shim),
        "resume=True",
        f"resume_path={hydra_escape(cfg.resume_ckpt)}",
        "gpus=1",
        "use_caching=True",
        f"lr_scheduler={cfg.lr_scheduler}",
        f"warmup_ratio={cfg.warmup_ratio}",
        f"model.training.learning_rate={cfg.learning_rate}",
        f"model.training.max_epochs={cfg.max_epochs}",
        f"model.training.batch_size={cfg.batch_size}",
        f"model.training.num_workers={cfg.num_workers}",
        f"model.training.augment={'True' if cfg.augment else 'False'}",
        f"model.training.augment_parked={'True' if cfg.augment_parked else 'False'}",
    ]
    if not cfg.filter_routes:
        cmd.append("+model.training.filter_routes=False")
    if cfg.stop_speed_loss_weight is not None:
        cmd.append(f"model.training.stop_speed_loss_weight={cfg.stop_speed_loss_weight}")
    cmd.extend(cfg.extra_hydra)
    cmd.extend(
        [
            f"model.training.log_path={hydra_escape(pt / f'log/ft_{cfg.checkpoint_addon}_{cfg.seed}')}",
            f"expname=ft_{cfg.checkpoint_addon}",
            f"wandb_name=ft_{cfg.checkpoint_addon}_{cfg.seed}",
        ]
    )
    if cfg.hydra_run_dir is not None:
        cmd.append(f"hydra.run.dir={hydra_escape(cfg.hydra_run_dir)}")
    return cmd


def run_finetune(cfg: FinetuneConfig, *, cwd: Path | None = None, log_path: Path | None = None) -> int:
    cfg.validate()
    cmd = build_finetune_cmd(cfg)
    pt = plan_t()
    print("=" * 60)
    print("PlanT2 fine-tune")
    print(f"  SPLIT    = {cfg.split}")
    print(f"  CKPT     = {cfg.resume_ckpt}")
    print(f"  DS_LOCAL = {cfg.ds_local}")
    print(f"  GPU      = {cfg.cuda_device}")
    print(f"  LR       = {cfg.learning_rate}")
    print(f"  ADDON    = {cfg.checkpoint_addon}")
    print("=" * 60)

    stdout = open(log_path, "a") if log_path else None
    try:
        proc = subprocess.run(cmd, cwd=str(cwd or pt), stdout=stdout, stderr=subprocess.STDOUT)
        return proc.returncode
    finally:
        if stdout is not None:
            stdout.close()


def lr_tag(lr: str) -> str:
    return lr.replace("-", "")
