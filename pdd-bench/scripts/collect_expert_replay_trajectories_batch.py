#!/usr/bin/env python3
"""
Run smirnova per_sign_bench/expert_replay_inenv.py on up to N replays per sign folder
under benchmark_output/<preset>/<sign>/expert/replays/, save stdout JSON per scene.

Example:
  export SDL_VIDEODRIVER=dummy
  python collect_expert_replay_trajectories_batch.py \\
    --benchmark-root .../per_sign_bench/benchmark_output/mini \\
    --expert-replay-script .../per_sign_bench/expert_replay_inenv.py \\
    --output-dir .../trajectories \\
    --per-sign-limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _list_replay_pairs(replays_dir: Path) -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for pkl in sorted(replays_dir.glob("sd_*.pkl")):
        scene_id = pkl.name[len("sd_") : -len(".pkl")]
        sidecar = replays_dir / f"{scene_id}.meta.json"
        if sidecar.is_file():
            out.append((pkl, sidecar))
    return out


def main() -> None:
    # When stdout is a pipe/file (e.g. nohup >> log), default block buffering hides lines from `tail -f`.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass

    p = argparse.ArgumentParser(description="Batch expert_replay_inenv per sign, save JSON.")
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--expert-replay-script", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--per-sign-limit", type=int, default=100)
    p.add_argument("--python", type=Path, default=None, help="Python exe (default: current interpreter)")
    p.add_argument("--max-steps", type=int, default=500)
    args = p.parse_args()

    bench = args.benchmark_root.resolve()
    script = args.expert_replay_script.resolve()
    out_root = args.output_dir.resolve()
    py = args.python or Path(sys.executable)

    if not bench.is_dir():
        print(f"benchmark-root not found: {bench}", file=sys.stderr, flush=True)
        sys.exit(2)
    if not script.is_file():
        print(f"expert-replay-script not found: {script}", file=sys.stderr, flush=True)
        sys.exit(2)

    out_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")

    manifest = []
    sign_dirs = sorted(d for d in bench.iterdir() if d.is_dir() and not d.name.startswith("."))
    for sign_dir in sign_dirs:
        replays = sign_dir / "expert" / "replays"
        if not replays.is_dir():
            continue
        pairs = _list_replay_pairs(replays)[: max(0, args.per_sign_limit)]
        if not pairs:
            continue
        sign_out = out_root / sign_dir.name
        sign_out.mkdir(parents=True, exist_ok=True)
        print(f"[sign {sign_dir.name}] {len(pairs)} replay(s)", flush=True)
        for pkl, sidecar in pairs:
            scene_id = pkl.name[len("sd_") : -len(".pkl")]
            json_path = sign_out / f"{scene_id}.json"
            cmd = [
                str(py),
                str(script),
                "--pkl",
                str(pkl),
                "--sidecar",
                str(sidecar),
                "--ego-mode",
                "recorded",
                "--npc-mode",
                "recorded",
                "--max-steps",
                str(args.max_steps),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(script.parent))
            text = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
            rec = {
                "sign_folder": sign_dir.name,
                "scene_id": scene_id,
                "pkl": str(pkl),
                "sidecar": str(sidecar),
                "returncode": r.returncode,
                "stdout": r.stdout or "",
                "stderr": r.stderr or "",
            }
            try:
                lines = (r.stdout or "").strip().splitlines()
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith("{"):
                        rec["parsed"] = json.loads(lines[i])
                        break
            except Exception:
                pass
            json_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest.append(
                {
                    "sign_folder": sign_dir.name,
                    "scene_id": scene_id,
                    "json": str(json_path),
                    "ok": r.returncode == 0,
                }
            )
            tag = "ok" if r.returncode == 0 else "FAIL"
            print(f"  [{tag}] {sign_dir.name}/{scene_id} rc={r.returncode}", flush=True)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ok = sum(1 for m in manifest if m["ok"])
    print(f"Done. runs={len(manifest)} ok={ok} fail={len(manifest) - ok} -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
