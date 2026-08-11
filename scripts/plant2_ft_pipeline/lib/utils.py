"""Shared helpers: parallel workers, dataset counts, FV expert prep."""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

# PlanT.yaml: wps_len=8, seq_len=1; range(5, n - wps - seq - 2) → n-16 samples
_PLANT2_SAMPLE_OVERHEAD = 16

_FV_PATH_KEYS = ("pkl_path", "sidecar_path", "gif_path", "winning_pkl", "winning_sidecar")
_FV_OLD_PREFIX = "/home/jovyan/shares/SR006.nfs2/smirnova/"
_FV_NEW_PREFIX = "/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova/"


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def default_prefill_max_workers(nproc: int | None = None) -> int:
    n = nproc or os.cpu_count() or 8
    return max(8, min(32, int(0.50 * n / 4)))


def default_dump_max_workers(nproc: int | None = None) -> int:
    n = nproc or os.cpu_count() or 8
    return max(8, min(64, int(0.80 * n / 3)))


def _count_route_samples(route: Path) -> int:
    boxes = route / "boxes"
    try:
        n = sum(1 for _ in boxes.iterdir())
    except OSError:
        return 0
    return max(0, n - _PLANT2_SAMPLE_OVERHEAD)


def count_plant2_samples(ds_root: Path, max_workers: int | None = None) -> int:
    """Fast sample count mirroring PlanTDataset index without results/slurm I/O."""
    data = Path(ds_root).rstrip("/") if isinstance(ds_root, str) else ds_root
    data = data / "data" if (data / "data").is_dir() else data
    routes = [p for p in data.iterdir() if p.is_dir() and (p / "boxes").is_dir()]
    if not routes:
        return 0
    workers = max_workers or min(32, max(4, (os.cpu_count() or 8) // 4))
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_count_route_samples, r) for r in routes]
        for fut in as_completed(futs):
            total += int(fut.result())
    return total


def _remap_fv_path(value: str) -> str:
    if value.startswith(_FV_OLD_PREFIX):
        return _FV_NEW_PREFIX + value[len(_FV_OLD_PREFIX) :]
    return value


def prepare_fv_experts(
    *,
    src: Path,
    out_dir: Path,
    signs: Sequence[str],
    node_filter: str = "nodeA",
) -> dict[str, int]:
    """Filter FV experts (nodeA), remap NFS paths, write per-sign jsonl files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_sign: dict[str, list[dict]] = {s: [] for s in signs}
    other = miss_pkl = n_in = 0

    if not src.is_file():
        print(f"WARN: FV experts missing: {src}")
        return {s: 0 for s in signs}

    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pkl = row.get("pkl_path") or ""
        if node_filter and node_filter not in pkl:
            continue
        n_in += 1
        for key in _FV_PATH_KEYS:
            if key in row and row[key]:
                row[key] = _remap_fv_path(row[key])
        sign = str(row.get("sign") or "")
        if sign not in by_sign:
            other += 1
            continue
        if not Path(row.get("pkl_path") or "").exists():
            miss_pkl += 1
            continue
        by_sign[sign].append(row)

    counts: dict[str, int] = {}
    for sign, rows in by_sign.items():
        path = out_dir / f"experts_{sign.replace('.', '_')}_top1.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[sign] = len(rows)
        print(f"wrote {path.name} n={len(rows)}")

    print(
        f"FILTERED node={node_filter} kept={n_in} other_sign={other} "
        f"miss_pkl={miss_pkl} counts={counts}"
    )
    return counts


@dataclass
class ParallelTask:
    key: str
    run: Callable[[], int]


class WorkerPool:
    """Cap concurrent subprocess/worker tasks (replaces bash wait_slot loops)."""

    def __init__(self, max_workers: int, stagger_sec: float = 0.0) -> None:
        self.max_workers = max(1, max_workers)
        self.stagger_sec = stagger_sec
        self._procs: list[tuple[str, subprocess.Popen]] = []

    def _reap_done(self) -> None:
        alive: list[tuple[str, subprocess.Popen]] = []
        for key, proc in self._procs:
            rc = proc.poll()
            if rc is None:
                alive.append((key, proc))
        self._procs = alive

    def wait_slot(self) -> None:
        while len(self._procs) >= self.max_workers:
            self._reap_done()
            if len(self._procs) >= self.max_workers:
                time.sleep(2)

    def spawn(
        self,
        key: str,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.wait_slot()
        stdout = open(log_path, "w") if log_path else subprocess.DEVNULL  # noqa: SIM115
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
        self._procs.append((key, proc))
        if self.stagger_sec > 0:
            time.sleep(self.stagger_sec)

    def wait_all(self) -> dict[str, int]:
        rc: dict[str, int] = {}
        for key, proc in self._procs:
            rc[key] = proc.wait()
        self._procs.clear()
        return rc
