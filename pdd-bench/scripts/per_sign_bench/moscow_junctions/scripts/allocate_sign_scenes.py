#!/usr/bin/env python3
"""Allocate ~n_train maps per sign from the shared global train/test split.

Shared pool: signs sample independently from the same train_ids / test_ids
(a junction may be assigned to several signs). Within one sign, scene_ids are
unique.

For signs with shapes [T, X], ``x_share`` is the fraction of X maps
(e.g. 0.5 → 50/50 X/T). That is the former informal name ``x_frac``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNS_YAML = ROOT / "splits" / "signs.yaml"
DEFAULT_TRAIN = ROOT / "splits" / "train_ids.json"
DEFAULT_TEST = ROOT / "splits" / "test_ids.json"
DEFAULT_OUT = ROOT / "splits" / "sign_allocations.json"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        # Minimal subset parser for our flat signs.yaml (no nested lists of maps).
        return _load_yaml_stdlib(path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_yaml_stdlib(path: Path) -> dict:
    """Tiny YAML loader for this config shape only."""
    text = path.read_text(encoding="utf-8")
    # Prefer PyYAML when present; fallback uses json via a rewrite is fragile.
    # Require PyYAML for robustness.
    raise ImportError(
        f"PyYAML is required to read {path}. "
        "Install with: pip install pyyaml"
    )


def _counts_for_sign(
    *,
    shapes: List[str],
    n_total: int,
    x_share: Optional[float],
) -> Dict[str, int]:
    shapes = [s.upper() for s in shapes]
    if shapes == ["O"]:
        return {"O": n_total}
    if shapes == ["T"]:
        return {"T": n_total}
    if set(shapes) == {"T", "X"}:
        share = 0.5 if x_share is None else float(x_share)
        n_x = int(round(n_total * share))
        n_x = min(max(n_x, 0), n_total)
        return {"X": n_x, "T": n_total - n_x}
    # Generic: split evenly across requested shapes
    base, rem = divmod(n_total, len(shapes))
    out = {s: base for s in shapes}
    for s in shapes[:rem]:
        out[s] += 1
    return out


def _sample(
    pool: List[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    if k <= 0:
        return []
    if k > len(pool):
        # Not enough unique maps: take all (caller logs shortfall).
        return sorted(pool)
    return sorted(rng.sample(pool, k))


def allocate(
    *,
    signs_cfg: dict,
    train_by_shape: Dict[str, List[str]],
    test_by_shape: Dict[str, List[str]],
) -> dict:
    seed = int(signs_cfg.get("seed", 42))
    n_train = int(signs_cfg.get("n_train", 115))
    test_frac = float(signs_cfg.get("test_frac", 0.2))
    default_n_test = int(signs_cfg.get("n_test") or max(1, round(n_train * test_frac / (1.0 - test_frac))))
    signs = signs_cfg.get("signs") or {}

    allocations: Dict[str, Any] = {}
    shortfalls: List[str] = []

    for sign_code, spec in sorted(signs.items(), key=lambda kv: str(kv[0])):
        spec = spec or {}
        shapes = list(spec.get("shapes") or ["T", "X"])
        x_share = spec.get("x_share", spec.get("x_frac"))
        sign_n_train = int(spec.get("n_train", n_train))
        sign_n_test = int(spec.get("n_test", default_n_test))

        train_need = _counts_for_sign(
            shapes=shapes, n_total=sign_n_train, x_share=x_share
        )
        test_need = _counts_for_sign(
            shapes=shapes, n_total=sign_n_test, x_share=x_share
        )

        rng_train = random.Random(f"{seed}|{sign_code}|train")
        rng_test = random.Random(f"{seed}|{sign_code}|test")

        train_ids: List[str] = []
        test_ids: List[str] = []
        got_train: Dict[str, int] = {}
        got_test: Dict[str, int] = {}

        for shape, need in train_need.items():
            pool = list(train_by_shape.get(shape, []))
            picked = _sample(pool, need, rng_train)
            got_train[shape] = len(picked)
            if len(picked) < need:
                shortfalls.append(
                    f"{sign_code} train {shape}: want {need}, got {len(picked)}"
                )
            train_ids.extend(picked)

        for shape, need in test_need.items():
            pool = list(test_by_shape.get(shape, []))
            picked = _sample(pool, need, rng_test)
            got_test[shape] = len(picked)
            if len(picked) < need:
                shortfalls.append(
                    f"{sign_code} test {shape}: want {need}, got {len(picked)}"
                )
            test_ids.extend(picked)

        allocations[str(sign_code)] = {
            "shapes": shapes,
            "x_share": x_share,
            "train": {
                "scene_ids": sorted(set(train_ids)),
                "by_shape": got_train,
                "n": len(set(train_ids)),
            },
            "test": {
                "scene_ids": sorted(set(test_ids)),
                "by_shape": got_test,
                "n": len(set(test_ids)),
            },
        }

    return {
        "seed": seed,
        "n_train_target": n_train,
        "n_test_target": default_n_test,
        "test_frac": test_frac,
        "shared_pool": True,
        "shortfalls": shortfalls,
        "signs": allocations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signs-yaml", type=Path, default=DEFAULT_SIGNS_YAML)
    ap.add_argument("--train-ids", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--test-ids", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in (args.signs_yaml, args.train_ids, args.test_ids):
        if not p.is_file():
            sys.exit(f"ERROR: missing {p}")

    try:
        signs_cfg = _load_yaml(args.signs_yaml)
    except ImportError:
        # Fallback: try json sibling or pip hint
        sys.exit(
            "ERROR: PyYAML required. pip install pyyaml\n"
            f"(config: {args.signs_yaml})"
        )

    train_doc = json.loads(args.train_ids.read_text(encoding="utf-8"))
    test_doc = json.loads(args.test_ids.read_text(encoding="utf-8"))
    result = allocate(
        signs_cfg=signs_cfg,
        train_by_shape=train_doc.get("by_shape") or {},
        test_by_shape=test_doc.get("by_shape") or {},
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[allocate] signs={len(result['signs'])} → {args.out}")
    for code, block in result["signs"].items():
        print(
            f"  {code}: train={block['train']['n']} {block['train']['by_shape']}  "
            f"test={block['test']['n']} {block['test']['by_shape']}"
        )
    if result["shortfalls"]:
        print("[allocate] shortfalls:")
        for line in result["shortfalls"]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
