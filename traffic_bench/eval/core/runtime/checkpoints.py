"""NN checkpoint defaults for priority_bench policies.

``plant2`` / ``plant2_rule`` → pretrained ``checkpoints/plant2_pretrain``.
``plant2_ft`` → newest ``.ckpt``/``.pt``/``.pth`` under ``checkpoints/plant2_finetuned``.
"""
from __future__ import annotations

from pathlib import Path

_CKPT_SUFFIXES = {".ckpt", ".pt", ".pth"}

# traffic_bench/eval/core/runtime/checkpoints.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
SIGN_BENCH_DIR = Path(__file__).resolve().parents[2]
PDD_BENCH_DIR = SIGN_BENCH_DIR.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
PLANT2_FT_DIR = CHECKPOINTS_DIR / "plant2_finetuned"

NN_NEED_CHECKPOINT = {
    "carl",
    "carl_rule",
    "plant2",
    "plant2_rule",
    "plant2_ft",
}
PLAIN_PLANT2_POLICIES = frozenset({"plant2", "plant2_ft"})
PLANT2_POLICIES = frozenset({"plant2", "plant2_rule", "plant2_ft"})

DEFAULT_MODEL_PATHS: dict[str, Path] = {
    "carl": CHECKPOINTS_DIR / "carl" / "nuplan_51479_1B" / "model_best.pth",
    "carl_rule": CHECKPOINTS_DIR / "carl" / "nuplan_51479_1B" / "model_best.pth",
    "plant2": CHECKPOINTS_DIR / "plant2_pretrain" / "epoch=029_final_3.ckpt",
    "plant2_rule": CHECKPOINTS_DIR / "plant2_pretrain" / "epoch=029_final_3.ckpt",
    "plant2_ft": PLANT2_FT_DIR,
}


def resolve_ckpt_path(path: Path | str | None) -> Path | None:
    """Return a checkpoint file, or the newest weights file in a directory."""
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    cands = [
        f
        for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in _CKPT_SUFFIXES
    ]
    if not cands:
        return None
    cands.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return cands[0]


def resolve_nn_checkpoint(policy: str, model_path: str | None) -> str | None:
    """Use an explicit path, else the policy default (file or finetune dir)."""
    if model_path:
        resolved = resolve_ckpt_path(model_path)
        return str(resolved) if resolved is not None else str(model_path)
    if policy not in NN_NEED_CHECKPOINT:
        return None
    default = DEFAULT_MODEL_PATHS.get(policy)
    resolved = resolve_ckpt_path(default)
    if resolved is None:
        return None
    print(f"Using default checkpoint for {policy}: {resolved}")
    return str(resolved)
