#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


BACKEND_FILES = {
    "sumo": ["sumo/sumo_manifest.jsonl", "real_manifest.jsonl"],
    "pgmap": ["pgmap_materialized.jsonl"],
    "paired": ["paired_materialized.jsonl"],
    "citymap": ["citymap_materialized.jsonl"],
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _pick_manifest(sign_dir: Path, backend: str) -> Path | None:
    for rel in BACKEND_FILES[backend]:
        p = sign_dir / rel
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _base_group_key(row: dict, backend: str, key_mode: str) -> str:
    if key_mode == "net_path":
        if backend == "sumo":
            v = row.get("net_path")
            if v:
                return f"net:{v}"
        v = row.get("scene_id")
        if v:
            return f"scene:{v}"
        return "unknown"

    # key_mode == scene_id
    v = row.get("scene_id")
    if v:
        return f"scene:{v}"
    if backend == "sumo":
        nv = row.get("net_path")
        if nv:
            return f"net:{nv}"
    return "unknown"


def _intra_sign_round_robin(rows: list[tuple[str, str, dict]], key_mode: str) -> list[tuple[str, str, dict]]:
    by_group: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for sign_slug, backend, row in rows:
        k = _base_group_key(row, backend, key_mode)
        by_group[k].append((sign_slug, backend, row))

    ordered_groups = sorted(by_group.keys())
    if not ordered_groups:
        return []

    max_len = max(len(by_group[g]) for g in ordered_groups)
    out: list[tuple[str, str, dict]] = []
    for i in range(max_len):
        for g in ordered_groups:
            if i < len(by_group[g]):
                out.append(by_group[g][i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--chunk-start", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--backends", default="sumo,pgmap,paired,citymap")
    ap.add_argument(
        "--intra-sign-balance-key",
        default="net_path",
        choices=["net_path", "scene_id"],
        help="How to group similar scenes inside one sign before round-robin selection",
    )
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]

    by_sign_rows: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for sign_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
        sign_slug = sign_dir.name
        for backend in backends:
            manifest = _pick_manifest(sign_dir, backend)
            if manifest is None:
                continue
            for row in _read_jsonl(manifest):
                if "valid" in row and not row["valid"]:
                    continue
                by_sign_rows[sign_slug].append((sign_slug, backend, row))

    signs = sorted(by_sign_rows.keys())
    if not signs:
        raise RuntimeError(f"No valid rows found under {src_root}")

    # Reorder rows inside each sign: first diversify base scenes, then variants.
    for s in signs:
        by_sign_rows[s] = _intra_sign_round_robin(by_sign_rows[s], args.intra_sign_balance_key)

    # Balanced order by sign: round-robin 1 row per sign each cycle.
    balanced: list[tuple[str, str, dict]] = []
    max_len = max(len(by_sign_rows[s]) for s in signs)
    for i in range(max_len):
        for s in signs:
            rows_s = by_sign_rows[s]
            if i < len(rows_s):
                balanced.append(rows_s[i])

    sel = balanced[args.chunk_start : args.chunk_start + args.chunk_size]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sign_slug, backend, row in sel:
        grouped[(sign_slug, backend)].append(row)

    for (sign_slug, backend), rows in grouped.items():
        sign_dir = out_root / sign_slug
        if backend == "sumo":
            dst = sign_dir / "sumo" / "sumo_manifest.jsonl"
        elif backend == "pgmap":
            dst = sign_dir / "pgmap_materialized.jsonl"
        elif backend == "paired":
            dst = sign_dir / "paired_materialized.jsonl"
        else:
            dst = sign_dir / "citymap_materialized.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    meta = {
        "src_root": str(src_root),
        "chunk_start": args.chunk_start,
        "chunk_size": args.chunk_size,
        "selected": len(sel),
        "total_available": len(balanced),
        "backends": backends,
        "n_signs": len(signs),
        "selection": "balanced_round_robin_by_sign",
        "intra_sign_balance_key": args.intra_sign_balance_key,
    }
    (out_root / "chunk_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
