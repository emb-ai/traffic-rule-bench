#!/usr/bin/env python3
"""Merge ep10 + ep20/ep30 + ep40 class_emb results into the fixed RESULTS.md.

Reads:
  - plant2_stop_pipeline_lrgrid_ep10_classemb/eval/<lr>_{last,best}
  - plant2_stop_pipeline_lrgrid_ep20_30_classemb/eval/lr3e4_ep{20,30}_last
  - plant2_stop_pipeline_lrgrid_ep40_classemb/eval/lr3e4_ep40_last

Writes (fixed path):
  plant2_stop_pipeline_lrgrid_ep10_classemb/RESULTS.md

Does not delete or overwrite completed prior metric rows — regenerates the
table from all available cumulative.json reports across workdirs.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

TRB = Path("/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench")
EP10_WORK = TRB / "plant2_stop_pipeline_lrgrid_ep10_classemb"
EP20_30_WORK = Path(
    os.environ.get(
        "EP20_30_WORK", TRB / "plant2_stop_pipeline_lrgrid_ep20_30_classemb"
    )
)
EP40_WORK = Path(os.environ.get("WORK", Path(__file__).resolve().parent))
OUT = Path(os.environ.get("RESULTS_MD", EP10_WORK / "RESULTS.md"))

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
}

EP10_LR_ORDER = ("lr1e4", "lr3e4", "lr1e3", "lr5e4")
KIND_ORDER = ("last", "best")
LONG_TAG_ORDER = ("lr3e4_ep20", "lr3e4_ep30", "lr3e4_ep40")


def _parse_report(eval_dir: Path) -> dict | None:
    cum = eval_dir / "reports" / "cumulative.json"
    md = eval_dir / "reports" / "report_cumulative.md"
    if not cum.is_file():
        return None
    data = json.loads(cum.read_text())
    pb = data.get("per_baseline") or {}
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


def _train_meta(work: Path, tag: str) -> dict:
    p = work / "train_meta" / f"{tag}.json"
    if p.is_file():
        return json.loads(p.read_text())
    return {}


def _fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _ckpt_basename(path: str | None) -> str:
    if not path:
        return "—"
    return Path(path).name or "—"


def collect_ep10_rows() -> list[dict]:
    rows: list[dict] = []
    eval_root = EP10_WORK / "eval"
    if not eval_root.is_dir():
        return rows
    for d in sorted(eval_root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(lr[0-9e]+)_(last|best)", d.name)
        if not m:
            continue
        lr_tag, kind = m.group(1), m.group(2)
        parsed = _parse_report(d)
        if not parsed:
            continue
        meta = _train_meta(EP10_WORK, lr_tag)
        rows.append(
            {
                "tag": f"{lr_tag}_{kind}",
                "group": "ep10",
                "lr_tag": lr_tag,
                "lr": meta.get("lr", ""),
                "epochs": meta.get("max_epochs", 10),
                "ckpt_kind": kind,
                "best_epoch": meta.get("best_epoch"),
                "ckpt": meta.get(f"ckpt_{kind}", ""),
                **parsed,
            }
        )

    def sort_key(r):
        try:
            li = EP10_LR_ORDER.index(r["lr_tag"])
        except ValueError:
            li = 99
        try:
            ki = KIND_ORDER.index(r["ckpt_kind"])
        except ValueError:
            ki = 99
        return (li, ki)

    rows.sort(key=sort_key)
    return rows


def _collect_long_from_work(work: Path, tag_re: str) -> list[dict]:
    rows: list[dict] = []
    eval_root = work / "eval"
    if not eval_root.is_dir():
        return rows
    for d in sorted(eval_root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(tag_re, d.name)
        if not m:
            continue
        tag_base, kind = m.group(1), m.group(2)
        parsed = _parse_report(d)
        if not parsed:
            continue
        meta = _train_meta(work, tag_base)
        epochs = meta.get("max_epochs")
        if epochs is None:
            if "ep40" in tag_base:
                epochs = 40
            elif "ep30" in tag_base:
                epochs = 30
            else:
                epochs = 20
        rows.append(
            {
                "tag": f"{tag_base}_{kind}",
                "group": "long",
                "lr_tag": tag_base,
                "lr": meta.get("lr", "3e-4"),
                "epochs": epochs,
                "ckpt_kind": kind,
                "best_epoch": meta.get("best_epoch"),
                "ckpt": meta.get(f"ckpt_{kind}", ""),
                **parsed,
            }
        )
    return rows


def collect_long_rows() -> list[dict]:
    rows: list[dict] = []
    rows.extend(
        _collect_long_from_work(
            EP20_30_WORK, r"(lr3e4_ep(?:20|30))_(last|best)"
        )
    )
    rows.extend(
        _collect_long_from_work(EP40_WORK, r"(lr3e4_ep40)_(last|best)")
    )

    def sort_key(r):
        try:
            li = LONG_TAG_ORDER.index(r["lr_tag"])
        except ValueError:
            li = 99
        try:
            ki = KIND_ORDER.index(r["ckpt_kind"])
        except ValueError:
            ki = 99
        return (li, ki)

    rows.sort(key=sort_key)
    return rows


def _row_line(r: dict) -> str:
    return (
        "| `{tag}` | {lr} | {epochs} | {kind} | {be} | {n} | {s} | {sc} | {e} | {ds} | `{ckpt}` |".format(
            tag=r["tag"],
            lr=r.get("lr") or r["lr_tag"],
            epochs=r.get("epochs") or "—",
            kind=r["ckpt_kind"],
            be=_fmt(r.get("best_epoch"), 0),
            n=r["n_episodes"],
            s=_fmt(r["success"]),
            sc=_fmt(r["sign_compliance"]),
            e=_fmt(r["efficiency"]),
            ds=_fmt(r["driving_score"]),
            ckpt=_ckpt_basename(r.get("ckpt")),
        )
    )


def main() -> None:
    ep10 = collect_ep10_rows()
    long_rows = collect_long_rows()

    lines = [
        "# PlanT STOP LR grid — class_emb (ep10 + lr3e4 ep20/ep30/ep40)",
        "",
        "Architecture: shared `class_emb` + `attr_emb` (no `tok_emb` / `sign_emb`).",
        "Warm-start: pretrain `CKPT0` has `tok_emb`; `lit_finetune` loads `strict=False`.",
        "",
        "Train dump/split: `plant2_stop_pipeline_signfix/plant2_l1_stop_split/` (294/50).",
        "Test: `stop_data/output/ts_test/real_manifest.jsonl` + `stop_data/scenes/`.",
        "Resume base: `stop_data/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt`.",
        "",
        "Drivers update this file after each eval completes.",
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
        "## LR grid results (ep10)",
        "",
        "| tag | lr | epochs | ckpt_kind | best_epoch | n | success | sign_compliance | efficiency | driving_score | ckpt |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    if not ep10:
        lines.append(
            "| *(pending)* | — | 10 | — | — | — | — | — | — | — | — |"
        )
    else:
        for r in ep10:
            lines.append(_row_line(r))

    lines += [
        "",
        "## Longer train @ lr=3e-4 (last ckpt)",
        "",
        "| tag | lr | epochs | ckpt_kind | best_epoch | n | success | sign_compliance | efficiency | driving_score | ckpt |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    if not long_rows:
        lines.append(
            "| *(pending)* | 3e-4 | — | last | — | — | — | — | — | — | — |"
        )
    else:
        for r in long_rows:
            lines.append(_row_line(r))
        # Placeholder until ep40 eval lands (keeps schema visible).
        if not any(r["tag"] == "lr3e4_ep40_last" for r in long_rows):
            lines.append(
                "| `lr3e4_ep40_last` | 3e-4 | 40 | last | — | — | — | — | — | — | *(pending)* |"
            )

    lines += [
        "",
        "## Artifact layout",
        "",
        "- Ep10 train ckpts: `plant2/PlanT/checkpoints_ft/stop_classemb_<lr_tag>_ep10/`",
        "- Ep10 eval: `plant2_stop_pipeline_lrgrid_ep10_classemb/eval/<lr_tag>_{last,best}/`",
        "- Ep20/30 train ckpts: `plant2/PlanT/checkpoints_ft/stop_classemb_lr3e4_ep{20,30}/`",
        "- Ep20/30 eval: `plant2_stop_pipeline_lrgrid_ep20_30_classemb/eval/lr3e4_ep{20,30}_last/`",
        "- Ep20/30 train meta: `plant2_stop_pipeline_lrgrid_ep20_30_classemb/train_meta/`",
        "- Ep40 train ckpts: `plant2/PlanT/checkpoints_ft/stop_classemb_lr3e4_ep40/`",
        "- Ep40 eval: `plant2_stop_pipeline_lrgrid_ep40_classemb/eval/lr3e4_ep40_last/`",
        "- Ep40 train meta: `plant2_stop_pipeline_lrgrid_ep40_classemb/train_meta/`",
        "- Logs: respective `logs/` under each workdir",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ep10_rows={len(ep10)} long_rows={len(long_rows)}")


if __name__ == "__main__":
    main()
