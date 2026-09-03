#!/usr/bin/env python3
"""Train/val hardlink split for *_signs dumps (PDD signs in boxes).

Sources:
  plant2_l1_traj_fv_nodeA_signs
  plant2_l1_from_experts_signs
  plant2_l1_lane_signs

OUT: plant2_l1_fv_experts_split_signs/
Same split rules as make_train_val_split_fv_experts.py (SEED=42, fixed50).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gzip
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Nodes without the uid->sign jsonls resolve priority-sign routes from the
# dumped boxes instead (the PDD sign is stored there with its code).
_PDD_RE = re.compile(r"^\d\.\d+(\.\d+)?$")


def sniff_sign(route_dir: Path) -> str | None:
    boxes = route_dir / "boxes"
    if not boxes.is_dir():
        return None
    frames = sorted(boxes.iterdir())
    if not frames:
        return None
    step = max(1, len(frames) // 12)
    for f in frames[::step]:
        try:
            entries = json.load(gzip.open(f))
        except Exception:  # noqa: BLE001
            continue
        for e in entries:
            code = e.get("pdd_code") or e.get("class") or ""
            if _PDD_RE.match(str(code)):
                return str(code)
    return None



def _srcs_from_env():
    """SPLIT_SRCS overrides the source list: `tag=/abs/path` items, ';'-separated.

    Naming the dumps explicitly is what lets a re-dump supersede part of an
    older one: mixing both would put two different conventions for the same
    sign into one training set, and nothing downstream would notice.
    """
    raw = os.environ.get("SPLIT_SRCS", "").strip()
    if not raw:
        raise SystemExit(
            "SPLIT_SRCS is required: ';'-separated `tag=/abs/path` items naming the "
            "dumps to split. It used to default to a fixed trio under one person's "
            "share, which silently split someone else's data or nothing at all."
        )
    out = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        tag, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"SPLIT_SRCS item must be tag=/abs/path, got {item!r}")
        out.append((tag.strip(), Path(path.strip())))
    return out


SRCS = _srcs_from_env()
_OUT_RAW = os.environ.get("SPLIT_OUT", "").strip()
if not _OUT_RAW:
    raise SystemExit("SPLIT_OUT is required: the directory to build the split in.")
OUT = Path(_OUT_RAW)
SEED = 42
# Hardlinking is I/O, not CPU: threads keep the external `cp` calls busy
# without the fork-time deadlock a process pool hits on a few thousand
# queued jobs (it hung here for 11 h with every worker in futex_wait).
WORKERS = int(os.environ.get("SPLIT_WORKERS", min(32, (os.cpu_count() or 8))))
REQUIRED_DIRS = ("measurements", "boxes", "bev_no_car_semantics")


def sign_of(name: str) -> str | None:
    if "_rb_" in name or re.match(r"sign_\d+_rb_", name):
        return "4.3"
    m = re.match(
        r"sumo_(2\.1|2\.3\.[123]|2\.4|2\.5|3\.24|4\.2\.[123]|4\.3|4\.6|5\.15\.1|5\.21|5\.31)_",
        name,
    )
    if m:
        return m.group(1)
    m = re.match(r"sumo_(3\.24|4\.6|5\.21|5\.31|4\.2\.[123])_", name)
    if m:
        return m.group(1)
    return None


def _oracle_jsonls() -> list[Path]:
    """Expert lists of a collection run: ORACLE_ROOT/<family>/experts/*_top1.jsonl.

    Without this the sign of a route is sniffed from ~12 sampled frames of its
    `boxes`; a route whose sign never entered the radius in those frames counts
    as `unknown` and is dropped from the split without a word. The expert list
    names the sign for every scene it selected, so the mapping is exact.
    """
    raw = os.environ.get("ORACLE_ROOT", "").strip()
    if not raw:
        return []
    out: list[Path] = []
    for root in raw.split(";"):
        root = root.strip()
        if not root:
            continue
        out.extend(sorted(Path(root).glob("*/experts/experts_scene_uid_top1.jsonl")))
    return out


def load_priority_uid_sign() -> dict[str, str]:
    mapping: dict[str, str] = {}
    files = _oracle_jsonls()
    for path in files:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                o = json.loads(line)
                uid = o.get("scene_uid")
                sign = str(o.get("sign") or "")
                if uid and sign:
                    mapping[str(uid)] = sign
    print(f"uid->sign map: {len(mapping)} entries from {len(files)} expert lists", flush=True)
    return mapping


def route_sign(name: str, uid2sign: dict[str, str]) -> str | None:
    s = sign_of(name)
    if s:
        return s
    for var in ("default", "s1", "s2", "s3", "s4"):
        suf = "_" + var
        if name.endswith(suf):
            uid = name[: -len(suf)]
            if uid in uid2sign:
                return uid2sign[uid]
    return uid2sign.get(name)


def hardlink_one(pair: tuple[str, str]) -> str:
    src_s, dst_s = pair
    dst = Path(dst_s)
    if dst.exists():
        return "skip"
    src = Path(src_s)
    r = subprocess.run(["cp", "-al", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode == 0:
        return "ok"
    r = subprocess.run(["cp", "-a", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"copy failed {src} -> {dst}: {r.stderr}")
    return "copied"


def _readable(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:  # noqa: BLE001
        return False


def verify_route(route_dir: Path) -> str | None:
    """Read every frame of a route back; return the first unreadable path.

    A dump worker killed mid-write leaves a .json.gz of the right name and size
    whose contents are not gzip. Every cheaper check passes it -- the directory
    listing, results.json.gz, the split itself -- and the run dies with
    BadGzipFile inside a DataLoader worker minutes into epoch 0, taking the
    whole training with it. The only check that sees it is reading the files.
    """
    for sub in ("measurements", "boxes"):
        d = route_dir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == ".gz" and not _readable(f):
                return str(f)
    return None


def route_is_ok(route_dir: Path) -> bool:
    for d in REQUIRED_DIRS:
        if not (route_dir / d).is_dir():
            return False
    results_path = route_dir / "results.json.gz"
    if not results_path.is_file():
        return False
    try:
        with gzip.open(results_path, "rt", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:  # noqa: BLE001
        return False
    status = results.get("status")
    if status in (
        "Failed",
        "Failed - Agent couldn't be set up",
        "Failed - Simulation crashed",
        "Failed - Agent crashed",
    ):
        return False
    scores = results.get("scores") or {}
    if "score_composed" not in scores:
        return False
    infra = results.get("infractions") or {}
    if "min_speed_infractions" not in infra:
        return False
    score = float(scores["score_composed"])
    n_inf = int(results.get("num_infractions") or 0)
    n_min = len(infra.get("min_speed_infractions") or [])
    if score < 100.0 and not (n_inf == n_min):
        return False
    return True


def split_counts(n: int) -> tuple[int, int, str]:
    if n >= 50:
        return n - 50, 50, "fixed50"
    n_val = max(1, int(round(0.2 * n)))
    n_train = n - n_val
    if n_train == 0:
        return n, 0, "ratio80_20"
    return n_train, n_val, "ratio80_20"


def main() -> None:
    rng = random.Random(SEED)
    uid2sign = load_priority_uid_sign()
    by: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    unknown = skipped_no_results = skipped_bad = 0

    for tag, src in SRCS:
        data = src / "data"
        if not data.is_dir():
            # A node that only carries some of the source trees can still build
            # a (smaller) split; the per-sign counts printed below make any
            # missing source obvious.
            if os.environ.get("ALLOW_MISSING_SRCS", "").strip() in ("1", "true"):
                print(f"skipping missing source {data}", flush=True)
                continue
            raise SystemExit(f"missing {data} — run dump first (or ALLOW_MISSING_SRCS=1)")
        print(f"scanning {tag}: {data}", flush=True)
        for p in sorted(data.iterdir()):
            if not p.is_dir():
                continue
            # Some dump generations nest the route inside a directory of its own
            # name, leaving empty boxes/measurements at the top and the real
            # frames one level down. dataset.py's recursive glob finds those, so
            # they train fine once split — but enumerating only the top level
            # dropped every one of them into skipped_no_results without a word.
            # Linking from the inner directory also flattens the route in the
            # split, so the glob stops counting each of them twice.
            route = p if (p / "results.json.gz").exists() else p / p.name
            if not (route / "results.json.gz").exists():
                skipped_no_results += 1
                continue
            if not route_is_ok(route):
                skipped_bad += 1
                continue
            # The OUTER name carries the sign code and names the route in the
            # split; only the source path moves inward.
            # The boxes decide admission, the uid map only refines WHICH sign.
            # With the order reversed, a route whose plate was never placed (the
            # geometry checks reject some) still entered on the uid map alone:
            # 60 such routes reached one split, and the dataset then resolved
            # them BY NAME, handing the model a route-level sign token with no
            # sign anywhere in its frames.
            sniffed = sniff_sign(route)
            if sniffed is None:
                unknown += 1
                continue
            sign = route_sign(p.name, uid2sign) or sniffed
            by[sign].append((route, p.name, tag))

    bare_tags: dict[str, set[str]] = defaultdict(set)
    # Key on the name the route will carry in the split, not on the source
    # directory's: with the nested layout the two can differ.
    for items in by.values():
        for _p, out_name, tag in items:
            bare_tags[out_name].add(tag)
    collide = {n for n, tags in bare_tags.items() if len(tags) > 1}
    if collide:
        print(f"name collisions: {len(collide)} — applying tag prefixes", flush=True)
        new_by: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
        for sign, items in by.items():
            for p, out_name, tag in items:
                if out_name in collide:
                    out_name = f"{tag}__{out_name}"
                new_by[sign].append((p, out_name, tag))
        by = new_by

    if os.environ.get("VERIFY_GZ", "1").strip() not in ("0", "false"):
        routes = [(sign, item) for sign, items in by.items() for item in items]
        print(f"verifying {len(routes)} routes are readable …", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            bad_first = list(ex.map(lambda r: verify_route(r[1][0]), routes))
        broken = {id(item) for (_s, item), b in zip(routes, bad_first) if b}
        if broken:
            for (_s, item), b in zip(routes, bad_first):
                if b:
                    print(f"  unreadable, dropping route: {b}", flush=True)
            by = {
                sign: [it for it in items if id(it) not in broken]
                for sign, items in by.items()
            }
        print(f"verified: {len(routes) - len(broken)} ok, {len(broken)} dropped", flush=True)

    print("signs:", {k: len(v) for k, v in sorted(by.items())}, flush=True)
    print(
        f"unknown={unknown} skipped_no_results={skipped_no_results} skipped_bad={skipped_bad}",
        flush=True,
    )

    train_data = OUT / "train" / "data"
    val_data = OUT / "val" / "data"
    for d in (train_data, val_data):
        d.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        slurm = OUT / split / "slurm"
        if slurm.exists():
            continue
        linked = False
        for _tag, src in SRCS:
            src_slurm = src / "slurm"
            if not src_slurm.exists():
                continue
            r = subprocess.run(
                ["cp", "-al", str(src_slurm), str(slurm)],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                subprocess.run(["cp", "-a", str(src_slurm), str(slurm)], check=True)
            linked = True
            break
        if not linked:
            # create dummy slurm like plant2_frames.ensure_slurm_dummy
            log_dir = slurm / "run_files" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "qsub_out2025_07.log").write_text("# dummy log\n", encoding="utf-8")

    split_meta: dict = {
        "seed": SEED,
        "sources": [{"tag": t, "path": str(p)} for t, p in SRCS],
        "out": str(OUT),
        "per_sign": {},
        "val": {},
        "train_counts": {},
    }

    jobs: list[tuple[str, str]] = []
    for sign, routes in sorted(by.items()):
        routes = list(routes)
        rng.shuffle(routes)
        n = len(routes)
        n_train, n_val, mode = split_counts(n)
        val_routes = routes[:n_val]
        train_routes = routes[n_val : n_val + n_train]
        if n_val == 0:
            train_routes = routes
            val_routes = []
        split_meta["per_sign"][sign] = {
            "N": n,
            "n_train": len(train_routes),
            "n_val": len(val_routes),
            "mode": mode,
        }
        split_meta["val"][sign] = [out_name for _, out_name, _ in val_routes]
        split_meta["train_counts"][sign] = len(train_routes)
        for p, out_name, _tag in val_routes:
            jobs.append((str(p), str(val_data / out_name)))
        for p, out_name, _tag in train_routes:
            jobs.append((str(p), str(train_data / out_name)))
        print(
            f"{sign}: N={n} train={len(train_routes)} val={len(val_routes)} mode={mode}",
            flush=True,
        )

    print(f"hardlinking {len(jobs)} routes with workers={WORKERS} …", flush=True)
    done = skipped = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(hardlink_one, j) for j in jobs]
        for fut in as_completed(futs):
            st = fut.result()
            done += 1
            if st == "skip":
                skipped += 1
            if done % 500 == 0 or done == len(futs):
                print(f"  linked {done}/{len(futs)} (skipped_existing={skipped})", flush=True)

    meta_path = OUT / "split_meta.json"
    meta_path.write_text(json.dumps(split_meta, indent=2))
    n_train = sum(1 for p in train_data.iterdir() if p.is_dir())
    n_val = sum(1 for p in val_data.iterdir() if p.is_dir())
    print(f"DONE train_routes={n_train} val_routes={n_val} meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
