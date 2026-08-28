#!/usr/bin/env python3
"""Train + eval PlanT2 on a single 2.5 trajectory with fixed hyperparameters from YAML/JSON.

All tunables live in the config file (default: tools/configs/overfit_1traj.yaml).
This script only orchestrates train → eval → log.

Examples:
  python tools/overfit_1traj_sweep.py --config tools/configs/overfit_1traj.yaml --gpu 0
  python tools/overfit_1traj_sweep.py --config my_run.yaml --force-train
  python tools/overfit_1traj_sweep.py --config tools/configs/overfit_1traj.yaml --eval-only
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from lib.env import plan_t, resolve_python, shepelev, signs_dir
from lib.eval_core import SignsEvalConfig, run_signs_eval, setup_metrics_tag
from lib.finetune import FinetuneConfig, run_finetune

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "overfit_1traj.yaml"


@dataclass
class OverfitConfig:
    trajectory: str
    split: Path
    ds_local: Path
    metrics_root: Path
    log_path: Path
    pretrain_ckpts: dict[str, Path]
    checkpoint_addon: str
    eval_tag: str
    train: dict[str, Any]
    eval: dict[str, Any]
    success: dict[str, Any]
    python: Path = field(default_factory=resolve_python)


def _load_raw_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SystemExit(
                "PyYAML required for .yaml configs: pip install pyyaml (or use .json)"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Config root must be a mapping: {path}")
    return data


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return base / p


def load_config(path: Path, *, python: Path | None = None) -> OverfitConfig:
    raw = _load_raw_config(path)
    sh = shepelev()
    paths = raw.get("paths") or {}
    run = raw.get("run") or {}
    train = raw.get("train") or {}
    ev = raw.get("eval") or {}
    success = raw.get("success") or {}

    pretrain_raw = paths.get("pretrain_ckpts") or {}
    pretrain_ckpts = {
        str(k): _resolve_path(v, base=sh)
        for k, v in pretrain_raw.items()
    }
    checkpoint_addon = str(run.get("checkpoint_addon") or "").strip()
    if not checkpoint_addon:
        raise SystemExit("config run.checkpoint_addon is required")

    eval_tag = str(run.get("eval_tag") or f"{checkpoint_addon}_eval").strip()

    return OverfitConfig(
        trajectory=str(paths.get("trajectory") or "").strip(),
        split=_resolve_path(paths.get("split", "plant2_l1_fv_experts_split_signs_2.5_1traj"), base=sh),
        ds_local=Path(paths.get("ds_local", "/tmp/plant2_ds_cache_2p5_tsfix")),
        metrics_root=_resolve_path(
            paths.get("metrics_root", "plant2_ft_metrics/overfit_2p5_1traj_bins10_cw_eval"),
            base=sh,
        ),
        log_path=_resolve_path(
            paths.get("log_path", "plant2_ft_metrics/overfit_2p5_1traj_experiment_log.txt"),
            base=sh,
        ),
        pretrain_ckpts=pretrain_ckpts,
        checkpoint_addon=checkpoint_addon,
        eval_tag=eval_tag,
        train=train,
        eval=ev,
        success=success,
        python=python or resolve_python(),
    )


def cw_str(weights: list[float]) -> str:
    inner = ",".join(str(int(w) if w == int(w) else w) for w in weights)
    return f"[{inner}]"


def pick_ckpt(addon_dir: Path, slot: str) -> Path | None:
    if slot == "best":
        hits = sorted(addon_dir.glob("best_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0] if hits else None
    hits = sorted(addon_dir.glob("last_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if hits:
        return hits[0]
    hits = sorted(addon_dir.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def train_log_dir(addon: str) -> Path:
    return plan_t() / "log" / f"ft_{addon}_1"


def eval_out_root(eval_tag: str) -> Path:
    return signs_dir() / "output" / eval_tag / "2_5" / "eval_out"


def parse_train_log(log_dir: Path) -> dict[str, float]:
    csv_path = log_dir / "CSVLogger/version_0/metrics.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    max_epoch = max(int(r["epoch"]) for r in rows if r.get("epoch", "").strip().isdigit())
    ep_rows = [r for r in rows if r.get("epoch", "").strip() == str(max_epoch)]
    if not ep_rows:
        return {}
    last = ep_rows[-1]
    out: dict[str, float] = {}
    for key in (
        "train/loss_all",
        "train/loss_wp",
        "train/loss_egospeed",
        "train/loss_speed",
        "train/loss_path",
    ):
        val = last.get(key, "")
        if val not in ("", None):
            try:
                out[key.replace("train/loss_", "")] = float(val)
            except ValueError:
                pass
    out["epoch"] = float(max_epoch)
    return out


def parse_predictions(predictions_path: Path) -> dict[str, Any]:
    if not predictions_path.is_file():
        return {}
    from collections import Counter

    hist_all: Counter[int] = Counter()
    hist_near: Counter[int] = Counter()
    p0_vals: list[float] = []
    p0_near: list[float] = []
    desired_near: list[float] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "step" not in row:
            continue
        probs = row.get("speed_probs")
        if not probs:
            continue
        argmax = max(range(len(probs)), key=lambda i: probs[i])
        hist_all[argmax] += 1
        p0_vals.append(float(row.get("speed_p0", probs[0])))
        dist = row.get("dist_to_stop_line_m")
        if dist is not None and float(dist) < 30.0:
            hist_near[argmax] += 1
            p0_near.append(float(row.get("speed_p0", probs[0])))
            if row.get("desired_speed_mps") is not None:
                desired_near.append(float(row["desired_speed_mps"]))
    if not hist_all:
        return {}
    return {
        "argmax_bins": dict(sorted(hist_all.items())),
        "argmax_bins_near_sign": dict(sorted(hist_near.items())),
        "p0_mean": sum(p0_vals) / len(p0_vals),
        "near_sign_p0_mean": sum(p0_near) / len(p0_near) if p0_near else None,
        "near_sign_desired_speed_mean": sum(desired_near) / len(desired_near) if desired_near else None,
        "n_steps": len(p0_vals),
    }


def parse_eval(
    cumulative_path: Path,
    episodes_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if cumulative_path.is_file():
        cum = json.loads(cumulative_path.read_text(encoding="utf-8"))
        base = cum.get("per_baseline", {}).get("plant2_default", {})
        out.update(
            {
                "sign_sr": base.get("sign_compliance_sr"),
                "dest_rate": base.get("dest_rate"),
                "success_rate": base.get("success_rate"),
                "steps": base.get("avg_steps"),
                "dist_m": base.get("avg_distance_travelled_m"),
            }
        )
    if episodes_path and episodes_path.is_file():
        line = episodes_path.read_text(encoding="utf-8").strip().splitlines()[0]
        ep = json.loads(line)
        out.update(
            {
                "violations": ep.get("sign_violations", ep.get("violations")),
                "reached_dest": ep.get("reached_dest"),
                "success": ep.get("success"),
            }
        )
    if predictions_path:
        out["speed_pred"] = parse_predictions(predictions_path)
    return out


def is_success(metrics: dict[str, Any], cfg: OverfitConfig) -> bool:
    steps_max = float(cfg.success.get("steps_max", 400.0))
    sign_min = float(cfg.success.get("sign_sr_min", 1.0))
    dest_min = float(cfg.success.get("dest_rate_min", 1.0))

    steps = metrics.get("steps")
    if steps is not None and float(steps) >= steps_max:
        return False
    sign_sr = metrics.get("sign_sr")
    dest = metrics.get("dest_rate")
    if sign_sr is not None and dest is not None:
        return float(sign_sr) >= sign_min and float(dest) >= dest_min
    violations = metrics.get("violations")
    reached = metrics.get("reached_dest")
    if violations is not None and reached is not None:
        return int(violations) == 0 and bool(reached)
    return False


def build_hydra_overrides(cfg: OverfitConfig) -> list[str]:
    t = cfg.train
    overrides = [
        f"model.waypoints.path_weight={int(t.get('path_weight', 0))}",
        f"model.waypoints.speed_weight={float(t.get('speed_weight', 5.0))}",
    ]
    cw = [float(x) for x in t.get("speed_class_weights", [])]
    if cw:
        overrides.append(f"model.training.speed_class_weights={cw_str(cw)}")
    flw = float(t.get("forecast_loss_weight", 1.0))
    if flw != 1.0:
        overrides.append(f"model.pre_training.forecastLoss_weight={flw}")
    wd = float(t.get("weight_decay", 0.1))
    if wd != 0.1:
        overrides.append(f"model.training.weight_decay={wd}")
    if bool(t.get("input_ego_speed", False)):
        overrides.append("model.training.input_ego_speed=True")
    return overrides


def hydra_run_dir_for(addon: str) -> Path:
    d = plan_t() / "outputs" / "hydra_sweep" / addon
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class RunResult:
    status: str
    train_rc: int = 0
    eval_rc: int = 0
    ckpt_path: Path | None = None
    gif_path: Path | None = None
    train_losses: dict[str, float] = field(default_factory=dict)
    eval_metrics: dict[str, Any] = field(default_factory=dict)
    speed_pred: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def format_log_entry(exp_id: str, cfg: OverfitConfig, result: RunResult) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t = cfg.train
    ev = cfg.eval
    m = result.eval_metrics
    sp = result.speed_pred or m.get("speed_pred", {})
    tl = result.train_losses
    ep = int(tl.get("epoch", int(t.get("max_epochs", 20)) - 1))
    cw = t.get("speed_class_weights", [])
    if tl:
        loss_line = (
            f"train_loss_ep{ep}: all={tl.get('all', float('nan')):.4f} "
            f"wp={tl.get('wp', float('nan')):.4f} "
            f"egospeed={tl.get('egospeed', float('nan')):.4f} "
            f"speed={tl.get('speed', float('nan')):.4f} "
            f"path={tl.get('path', float('nan')):.4f}"
        )
    else:
        loss_line = "train_loss_ep?: (missing)"
    lines = [
        f"=== {exp_id} | {ts} ===",
        f"pretrain: {t.get('pretrain')} | addon: {cfg.checkpoint_addon} | status: {result.status}",
        (
            f"train: lr={t.get('learning_rate')} ep={t.get('max_epochs')} "
            f"sw={t.get('speed_weight')} pw={t.get('path_weight')} "
            f"aug={int(bool(t.get('augment', False)))} wd={t.get('weight_decay')} "
            f"input_ego_speed={int(bool(t.get('input_ego_speed', False)))} "
            f"stop_w={t.get('stop_speed_loss_weight')} cw={cw_str([float(x) for x in cw])} "
            f"flw={t.get('forecast_loss_weight')} eval_slot={ev.get('slot', 'last')}"
        ),
        loss_line,
        (
            f"eval: sign_sr={m.get('sign_sr')} dest_rate={m.get('dest_rate')} success={m.get('success')} "
            f"violations={m.get('violations')} steps={m.get('steps')} dist_m={m.get('dist_m')}"
        ),
        (
            f"speed_pred: argmax_bins={sp.get('argmax_bins', {})} "
            f"p0_mean={sp.get('p0_mean', '?')} near_sign_p0_mean={sp.get('near_sign_p0_mean', '?')} "
            f"near_sign_desired={sp.get('near_sign_desired_speed_mean', '?')}"
        ),
        f"ckpt: {result.ckpt_path}",
        f"gif: {result.gif_path}",
        f"notes: {result.notes}",
    ]
    return "\n".join(lines) + "\n"


def append_log(cfg: OverfitConfig, entry: str) -> None:
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.log_path.open("a", encoding="utf-8") as f:
        f.write(entry)


def next_exp_id(cfg: OverfitConfig) -> str:
    if not cfg.log_path.is_file():
        return "EXP-001"
    ids = [int(m.group(1)) for m in re.finditer(r"=== EXP-(\d+)", cfg.log_path.read_text())]
    return f"EXP-{max(ids, default=0) + 1:03d}"


def validate_config(cfg: OverfitConfig) -> None:
    if not cfg.trajectory:
        raise SystemExit("config paths.trajectory is required")
    pretrain_key = str(cfg.train.get("pretrain", "final_1"))
    if pretrain_key not in cfg.pretrain_ckpts:
        raise SystemExit(
            f"train.pretrain={pretrain_key!r} not in paths.pretrain_ckpts keys "
            f"{list(cfg.pretrain_ckpts)}"
        )
    resume = cfg.pretrain_ckpts[pretrain_key]
    if not resume.is_file():
        raise SystemExit(f"pretrain checkpoint missing: {resume}")
    if not cfg.split.is_dir():
        raise SystemExit(f"split missing: {cfg.split}")
    if bool(cfg.train.get("input_ego_speed", False)):
        print(
            "WARN: input_ego_speed=True — 1traj cache may lack this feature (KeyError in forward)"
        )


def run_train(cfg: OverfitConfig, *, gpu: str, force: bool) -> tuple[int, Path | None]:
    addon = cfg.checkpoint_addon
    addon_dir = plan_t() / "checkpoints_ft" / addon
    slot = str(cfg.eval.get("slot", "last"))
    ckpt = pick_ckpt(addon_dir, slot)
    if ckpt is not None and not force:
        print(f"TRAIN SKIP existing ckpt slot={slot}: {ckpt}")
        return 0, ckpt

    pretrain_key = str(cfg.train.get("pretrain", "final_1"))
    resume_ckpt = cfg.pretrain_ckpts[pretrain_key]
    t = cfg.train
    train_log_file = plan_t() / "log" / f"sweep_{addon}.log"

    ft_cfg = FinetuneConfig(
        split=cfg.split,
        learning_rate=str(t.get("learning_rate", "5e-4")),
        checkpoint_addon=addon,
        cuda_device=gpu,
        ds_local=cfg.ds_local,
        batch_size=int(t.get("batch_size", 32)),
        num_workers=int(t.get("num_workers", 4)),
        max_epochs=int(t.get("max_epochs", 20)),
        ckpt_every_n_epochs=int(t.get("ckpt_every_n_epochs", 5)),
        lr_scheduler=str(t.get("lr_scheduler", "cosine_warmup")),
        warmup_ratio=float(t.get("warmup_ratio", 0.1)),
        resume_ckpt=resume_ckpt,
        augment=bool(t.get("augment", False)),
        augment_parked=bool(t.get("augment_parked", False)),
        filter_routes=bool(t.get("filter_routes", False)),
        stop_speed_loss_weight=float(t.get("stop_speed_loss_weight", 1.0)),
        extra_hydra=build_hydra_overrides(cfg),
        hydra_run_dir=hydra_run_dir_for(addon),
        python=cfg.python,
        seed=int(t.get("seed", 1)),
    )
    print(
        f"TRAIN {addon} pretrain={pretrain_key} lr={ft_cfg.learning_rate} "
        f"sw={t.get('speed_weight')} ep={ft_cfg.max_epochs}"
    )
    rc = run_finetune(ft_cfg, cwd=plan_t(), log_path=train_log_file)
    ckpt = pick_ckpt(addon_dir, slot)
    return rc, ckpt


def run_eval(cfg: OverfitConfig, *, gpu: str, ckpt: Path) -> RunResult:
    result = RunResult(status="fail")
    eval_tag = cfg.eval_tag
    setup_metrics_tag(cfg.metrics_root, eval_tag, ckpt)
    pred_path = cfg.metrics_root / eval_tag / f"{cfg.trajectory}_predictions.jsonl"
    ev = cfg.eval
    eval_cfg = SignsEvalConfig(
        ckpt=ckpt,
        tag=eval_tag,
        gpu=gpu,
        metrics_root=cfg.metrics_root,
        trajectory=cfg.trajectory,
        save_gifs=bool(ev.get("save_gifs", True)),
        save_predictions=bool(ev.get("save_predictions", True)),
        force_rerun=bool(ev.get("force_rerun", True)),
        python=cfg.python,
    )
    print(f"EVAL {eval_tag} ckpt={ckpt.name}")
    result.eval_rc = run_signs_eval(eval_cfg)
    result.ckpt_path = ckpt
    result.train_losses = parse_train_log(train_log_dir(cfg.checkpoint_addon))
    cum = eval_out_root(eval_tag) / "reports/cumulative.json"
    eps = eval_out_root(eval_tag) / "benchmark/full/policy_eval/plant2_default/episodes_plant2.jsonl"
    result.eval_metrics = parse_eval(cum, eps, pred_path)
    result.speed_pred = result.eval_metrics.pop("speed_pred", {})
    gif = cfg.metrics_root / eval_tag / f"{cfg.trajectory}.gif"
    result.gif_path = gif if gif.is_file() else None

    if result.eval_rc != 0 and not cum.is_file():
        result.status = "eval_fail"
    elif is_success(result.eval_metrics, cfg):
        result.status = "success"
    else:
        result.status = "done"
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML/JSON config with fixed hyperparameters (default: {DEFAULT_CONFIG.name})",
    )
    p.add_argument("--gpu", default="0")
    p.add_argument("--force-train", action="store_true", help="Retrain even if checkpoint exists")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    p.add_argument("--python", dest="python_exe", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, python=resolve_python(args.python_exe))
    validate_config(cfg)

    if args.dry_run:
        print(json.dumps(
            {
                "config": str(args.config.resolve()),
                "trajectory": cfg.trajectory,
                "split": str(cfg.split),
                "checkpoint_addon": cfg.checkpoint_addon,
                "eval_tag": cfg.eval_tag,
                "train": cfg.train,
                "eval": cfg.eval,
                "success": cfg.success,
            },
            indent=2,
            default=str,
        ))
        return 0

    addon_dir = plan_t() / "checkpoints_ft" / cfg.checkpoint_addon
    slot = str(cfg.eval.get("slot", "last"))
    ckpt: Path | None = None

    if not args.eval_only:
        train_rc, ckpt = run_train(cfg, gpu=args.gpu, force=args.force_train)
        if train_rc != 0 or ckpt is None:
            result = RunResult(
                status="train_fail",
                train_rc=train_rc,
                notes=f"train_rc={train_rc} ckpt_missing={ckpt is None}",
            )
            append_log(cfg, format_log_entry(next_exp_id(cfg), cfg, result))
            return 1

    if args.train_only:
        print(f"Train-only done: {ckpt}")
        return 0

    if ckpt is None:
        ckpt = pick_ckpt(addon_dir, slot)
    if ckpt is None:
        raise SystemExit(f"No checkpoint for addon={cfg.checkpoint_addon} slot={slot}")

    result = run_eval(cfg, gpu=args.gpu, ckpt=ckpt)
    append_log(cfg, format_log_entry(next_exp_id(cfg), cfg, result))

    steps = result.eval_metrics.get("steps")
    print(
        f"status={result.status} sign_sr={result.eval_metrics.get('sign_sr')} "
        f"dest={result.eval_metrics.get('dest_rate')} steps={steps}"
    )
    if result.gif_path:
        print(f"gif: {result.gif_path}")

    return 0 if result.status == "success" else (0 if result.status == "done" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
