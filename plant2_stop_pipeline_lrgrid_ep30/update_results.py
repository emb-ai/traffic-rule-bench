#!/usr/bin/env python3
"""Rebuild RESULTS.md from per-(lr, ckpt_kind) eval cumulative.json files.

Also records train-side metadata (best epoch, ckpt paths, final losses)
when present under train_meta/<tag>.json.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

WORK = Path(os.environ.get("WORK", Path(__file__).resolve().parent))
EVAL_ROOT = WORK / "eval"
TRAIN_META = WORK / "train_meta"
OUT = WORK / "RESULTS.md"

# Baseline from prior 20-ep @ 3e-4 last_ft eval (signfix pipeline).
BASELINE = {
    "tag": "baseline_ep20_lr3e4_last",
    "lr": "3e-4",
    "epochs": 20,
    "ckpt_kind": "last",
    "n_episodes": 42,
    "success": 0.714,
    "sign_compliance": 0.548,
    "efficiency": 78.109,
    "driving_score": 0.000,
    "note": "plant2_stop_pipeline_signfix/eval_test/full",
}

LR_ORDER = ("lr1e4", "lr3e4", "lr1e3", "lr5e4")
KIND_ORDER = ("last", "best")


def _parse_report(eval_dir: Path) -> dict | None:
    cum = eval_dir / "reports" / "cumulative.json"
    md = eval_dir / "reports" / "report_cumulative.md"
    if not cum.is_file():
        return None
    data = json.loads(cum.read_text())
    pb = data.get("per_baseline") or {}
    # Policy key is usually plant2_default
    row = next(iter(pb.values()), None)
    if row is None:
        return None
    return {
        "n_episodes": int(row.get("n") or 0),
        "success": float(row.get("success_rate") or 0.0),
        "sign_compliance": float(row.get("sign_compliance_sr") or 0.0),
        "efficiency": float(row.get("avg_efficiency") or 0.0),
        "driving_score": float(row.get("avg_driving_score") or 0.0),
        "report": str(md) if md.is_file() else str(cum),
    }


def _train_meta(tag: str) -> dict:
    p = TRAIN_META / f"{tag}.json"
    if p.is_file():
        return json.loads(p.read_text())
    return {}


def _fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main() -> None:
    rows = []
    # Discover completed evals
    if EVAL_ROOT.is_dir():
        for d in sorted(EVAL_ROOT.iterdir()):
            if not d.is_dir():
                continue
            m = re.fullmatch(r"(lr[0-9e]+)_(last|best)", d.name)
            if not m:
                continue
            lr_tag, kind = m.group(1), m.group(2)
            parsed = _parse_report(d)
            if not parsed:
                continue
            meta = _train_meta(lr_tag)
            rows.append(
                {
                    "tag": f"{lr_tag}_{kind}",
                    "lr_tag": lr_tag,
                    "lr": meta.get("lr", ""),
                    "epochs": meta.get("max_epochs", 30),
                    "ckpt_kind": kind,
                    "best_epoch": meta.get("best_epoch"),
                    "ckpt": meta.get(f"ckpt_{kind}", ""),
                    **parsed,
                }
            )

    # Stable sort
    def sort_key(r):
        try:
            li = LR_ORDER.index(r["lr_tag"])
        except ValueError:
            li = 99
        try:
            ki = KIND_ORDER.index(r["ckpt_kind"])
        except ValueError:
            ki = 99
        return (li, ki)

    rows.sort(key=sort_key)

    lines = [
        "# PlanT STOP LR grid — 30 epochs",
        "",
        "Train dump/split: `plant2_stop_pipeline_signfix/plant2_l1_stop_split/` (294/50).",
        "Test: `stop_data/output/ts_test/real_manifest.jsonl` + `stop_data/scenes/`.",
        "Resume base: `stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`.",
        "",
        "Driver updates this file after each `(lr, ckpt_kind)` eval completes.",
        "",
        "Parallel mode: trains run concurrently (one GPU each); evals start as each",
        "train finishes (eval GPU pool; lean `--jobs` when multiple evals overlap).",
        "",
        "## Baseline (prior, not re-run)",
        "",
        "| tag | lr | epochs | ckpt | n | success | sign_compliance | efficiency | driving_score |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        (
            f"| `{BASELINE['tag']}` | {BASELINE['lr']} | {BASELINE['epochs']} | "
            f"{BASELINE['ckpt_kind']} | {BASELINE['n_episodes']} | "
            f"{BASELINE['success']:.3f} | {BASELINE['sign_compliance']:.3f} | "
            f"{BASELINE['efficiency']:.3f} | {BASELINE['driving_score']:.3f} |"
        ),
        "",
        "## LR grid results",
        "",
        "| tag | lr | epochs | ckpt_kind | best_epoch | n | success | sign_compliance | efficiency | driving_score | ckpt |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    if not rows:
        lines.append(
            "| *(pending)* | — | 30 | — | — | — | — | — | — | — | — |"
        )
    else:
        for r in rows:
            ckpt_bn = Path(r.get("ckpt") or "").name or "—"
            lines.append(
                "| `{tag}` | {lr} | {epochs} | {kind} | {be} | {n} | {s} | {sc} | {e} | {ds} | `{ckpt}` |".format(
                    tag=r["tag"],
                    lr=r.get("lr") or r["lr_tag"],
                    epochs=r.get("epochs") or 30,
                    kind=r["ckpt_kind"],
                    be=_fmt(r.get("best_epoch"), 0),
                    n=r["n_episodes"],
                    s=_fmt(r["success"]),
                    sc=_fmt(r["sign_compliance"]),
                    e=_fmt(r["efficiency"]),
                    ds=_fmt(r["driving_score"]),
                    ckpt=ckpt_bn,
                )
            )

    lines += [
        "",
        "## Artifact layout",
        "",
        "- Train ckpts: `plant2/PlanT/checkpoints_ft/stop_signfix_<lr_tag>_ep30/`",
        "  - `last_ft_stop_signfix_<lr_tag>_ep30_1.ckpt`",
        "  - `best_NNN_stop_signfix_<lr_tag>_ep30_1.ckpt`",
        "- Eval out-dirs: `plant2_stop_pipeline_lrgrid_ep30/eval/<lr_tag>_{last,best}/`",
        "- Train meta: `plant2_stop_pipeline_lrgrid_ep30/train_meta/<lr_tag>.json`",
        "- Logs: `plant2_stop_pipeline_lrgrid_ep30/logs/`",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
