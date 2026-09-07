"""The six stages: plan -> dump -> split -> prefill -> train -> eval.

Each is a plain function taking an Experiment. Each prints the command it runs
and fails loudly. None of them reads anything from the ambient environment
except what `config.Experiment.env()` sets explicitly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import config as cfg
from . import metrics
from .shell import announce, require, run, run_background, wait_for


def _prepare(exp: cfg.Experiment) -> None:
    """Create the run's directories, so a stage is runnable on its own and not
    only through the CLI."""
    exp.log_dir.mkdir(parents=True, exist_ok=True)
    exp.work.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- 1. plan ---

def plan(exp: cfg.Experiment) -> Path:
    """Decide, once, which scenes train and which are held out for the metrics.

    Split by SCENE, per family. A scene contributes many routes (several ego
    variants and seeds), so splitting by route would put the same road in both
    halves — the earlier pipeline did exactly that, and `val/loss_path` looked
    good while meaning nothing.

    Writes, per family:
      plan/<family>/experts_train.jsonl   rows the dump replays
      plan/<family>/test_manifest.jsonl   catalog rows the eval scores
    """
    _prepare(exp)
    announce(f"[1/6] plan  ->  {exp.plan_dir}")

    experts = [json.loads(line) for line
               in cfg.EXPERTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_family: dict[str, list[dict]] = {}
    for row in experts:
        by_family.setdefault(row["family"], []).append(row)

    summary = []
    for name in exp.families:
        family = cfg.FAMILIES[name]
        catalog = cfg.load_catalog(family)
        scenes = sorted({row["scene_id"] for row in catalog})
        rng = random.Random(f"{exp.split_seed}:{name}")
        rng.shuffle(scenes)
        n_test = max(1, round(exp.test_fraction * len(scenes)))
        test = set(scenes[:n_test])

        out = exp.plan_dir / name
        out.mkdir(parents=True, exist_ok=True)

        train_rows = [r for r in by_family.get(name, [])
                      if r["scene_id"] not in test]
        test_rows = [_normalise_row(r) for r in catalog if r["scene_id"] in test]
        _write_jsonl(out / "experts_train.jsonl", train_rows)
        _write_jsonl(out / "test_manifest.jsonl", test_rows)

        leaked = {r["scene_id"] for r in train_rows} & test
        if leaked:
            raise SystemExit(f"{name}: scenes in both halves: {sorted(leaked)[:3]}")

        summary.append((name, family.sign, len(scenes), len(test),
                        len(train_rows), len(test_rows)))

    print(f"  {'family':18s} {'sign':7s} {'scenes':>7} {'test':>6} "
          f"{'train runs':>11} {'test runs':>10}")
    for row in summary:
        print(f"  {row[0]:18s} {row[1]:7s} {row[2]:7d} {row[3]:6d} "
              f"{row[4]:11d} {row[5]:10d}")
    return exp.plan_dir


def _normalise_row(row: dict) -> dict:
    """Make one catalog row usable by the benchmark runner.

    The oracle catalogs put the family NAME in ``sign_id`` ("stop"), while
    ``sumo_runner`` reads that field as an integer routing seed
    (``int(catalog_row.get("sign_id", 0))``) and dies on a string. The dump
    never hit this because the experts rows have no ``sign_id`` at all, so it
    took the 0 default — using 0 here keeps eval and dump on the same env seed,
    ``(sign_id + var_idx) % 100000``.
    """
    row = dict(row)
    if not isinstance(row.get("sign_id"), int):
        row["sign_id"] = 0
    return row


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


# ----------------------------------------------------------------- 2. dump ---

def dump(exp: cfg.Experiment, workers: int = 24) -> Path:
    """Replay each recorded expert run and record what PlanT2 trains on:
    object boxes, measurements and a semantic BEV per step.

    Replays the stored `replay.pkl`, so the trajectory is exactly the one the
    oracle picked — nothing is re-simulated and the result is deterministic.
    One directory per family, so any family can be re-dumped alone.
    """
    _prepare(exp)
    announce(f"[2/6] dump  ->  {exp.dump_dir}")
    for name in exp.families:
        family = cfg.FAMILIES[name]
        out = exp.dump_dir / name
        rows = sum(1 for _ in (exp.plan_dir / name / "experts_train.jsonl")
                   .read_text(encoding="utf-8").splitlines() if _.strip())
        _dump_sharded(exp, name, family, out, rows, workers)
        require(out / "data", f"{name} dump")
        print(f"  {name}: {len(list((out / 'data').glob('*')))} of {rows} routes")
    return exp.dump_dir


def _dump_sharded(exp: cfg.Experiment, name: str, family: cfg.Family,
                  out: Path, rows: int, workers: int) -> None:
    """Replay one family in parallel over contiguous row ranges.

    One route takes ~1.5 minutes, so the whole collection is a day and a half
    single-threaded. The work is CPU-bound (the semantic BEV is rendered on
    CPU), and each shard writes its own route directories, so plain process
    parallelism over ``--start/--count`` is enough — no coordination needed.
    """
    shards = min(workers, max(1, rows))
    per = -(-rows // shards)  # ceil
    procs: list = []
    for index in range(shards):
        start = index * per
        if start >= rows:
            break
        procs.append(run_background(
            [cfg.PYTHON, "-u", cfg.BENCH / "expert_replay_for_plant2.py",
             "--experts", exp.plan_dir / name / "experts_train.jsonl",
             "--scenes-root", family.scenes,
             "--save-plant2-dir", out,
             "--backends", "sumo",
             "--start", str(start), "--count", str(min(per, rows - start))],
            env=exp.env(dumping=True), cwd=cfg.BENCH,
            log=exp.log_dir / f"dump_{name}_{index:02d}.log"))
    print(f"  {name}: {len(procs)} shards over {rows} routes")
    wait_for(procs)


# ---------------------------------------------------------------- 3. split ---

def split(exp: cfg.Experiment) -> Path:
    """Build train/val out of the dumped routes, then rebuild the path target
    from the trajectory the expert actually drove.

    The rebuild is not optional. The dumper writes ``get_route()`` into both
    ``route_original`` (the model's INPUT) and ``route`` (its TARGET), so
    copying the input is exactly optimal and that is what the model learns:
    shifting the input route by +3.00 m shifted the prediction by +2.73 m.
    ``fix_route_target.py`` recomputes the target offline from the recorded
    ``pos_global``/``theta`` and symlinks the heavy files, so the source dump is
    never modified.
    """
    _prepare(exp)
    announce(f"[3/6] split  ->  {exp.split_dir}")
    raw = exp.work / "split_raw"
    _link_scene_split(exp.dump_dir, raw, exp.val_fraction, exp.split_seed)

    run([cfg.PYTHON, cfg.FT / "data" / "fix_route_target.py",
         "--src", raw, "--out", exp.split_dir, "--jobs", "24"],
        env=exp.env(), log=exp.log_dir / "split.log")
    require(exp.split_dir / "split_meta.json", "split metadata")
    _report_split(exp.split_dir)
    return exp.split_dir


def _link_scene_split(dump_dir: Path, out: Path, val_fraction: float,
                      seed: int) -> None:
    """Group dumped routes by scene and hold out whole scenes for validation."""
    scenes: dict[tuple[str, str], list[Path]] = {}
    for source in sorted(dump_dir.iterdir()):
        data = source / "data"
        if not data.is_dir():
            continue
        for route in sorted(data.iterdir()):
            if (route / "boxes").is_dir():
                scenes.setdefault((source.name, _scene_of(route.name)), []).append(route)

    if not scenes:
        raise SystemExit(f"No dumped routes under {dump_dir}")

    ids = sorted(scenes)
    random.Random(seed).shuffle(ids)
    val = set(ids[:max(1, int(val_fraction * len(ids)))])

    counts = {"train": 0, "val": 0}
    for scene in ids:
        part = "val" if scene in val else "train"
        target = out / part / "data"
        target.mkdir(parents=True, exist_ok=True)
        for route in scenes[scene]:
            link = target / route.name
            if not link.exists():
                link.symlink_to(route.resolve())
            counts[part] += 1

    # `lib/finetune.py` refuses to start without this file, and it is the only
    # record of how the split was drawn once the symlinks are made.
    per_family: dict[str, dict[str, int]] = {}
    for (family, _scene), routes in scenes.items():
        entry = per_family.setdefault(family, {"n_train": 0, "n_val": 0})
        key = "n_val" if (family, _scene) in val else "n_train"
        entry[key] += len(routes)
    _write_json(out / "split_meta.json", {
        "seed": seed,
        "mode": "scene_holdout",
        "val_fraction": val_fraction,
        "n_scenes": len(ids),
        "n_train_routes": counts["train"],
        "n_val_routes": counts["val"],
        "per_family": per_family,
        "val_scenes": sorted(f"{f}/{s}" for f, s in val),
    })
    print(f"  scenes {len(ids)}   train routes {counts['train']}   "
          f"val routes {counts['val']}")


def _scene_of(route_name: str) -> str:
    """Scene id of a dumped route directory.

    Route names are ``<scene_id>_lane<N>_seed<S>_v<K>_<variant>``, and scene ids
    themselves contain underscores (``junc_1025291468``,
    ``seg_1158657284_l0_td1_v0``), so the scene is everything before the
    ``_lane`` marker rather than a fixed number of fields.
    """
    marker = route_name.find("_lane")
    return route_name[:marker] if marker > 0 else route_name


def _report_split(split_dir: Path) -> None:
    for part in ("train", "val"):
        routes = sorted((split_dir / part / "data").glob("*"))
        print(f"  {part}: {len(routes)} routes")


# -------------------------------------------------------------- 4. prefill ---

def prefill(exp: cfg.Experiment, workers: int = 32) -> Path:
    """Warm the dataset disk cache before training.

    Without this the first epoch is spent decoding every frame off NFS while
    the GPU idles — the dominant cost of a short run. The cache is keyed by
    frame file path (``dataset.py``, ``labels[0].decode()``), not by dataset
    index, so it stays valid across runs with different oversampling factors,
    and several trainings can share one cache.

    It must be rebuilt whenever the *transform* changes, though: entries are
    stored post-augmentation under a ``_aug`` key, so a fix to ``aug_sample``
    leaves stale, wrongly-transformed samples behind that a later run would
    silently reuse.
    """
    _prepare(exp)
    announce(f"[4/6] prefill  ->  {exp.cache_dir}")
    run([cfg.PYTHON, "-u", cfg.FT / "data" / "prefill_diskcache.py", "parallel",
         "--ds", exp.split_dir / "train",
         "--ds-val", exp.split_dir / "val",
         "--ds-local", exp.cache_dir,
         "--cache-size-gb", "200",
         "--augment" if exp.augment else "--no-augment",
         "--max-workers", str(workers),
         "--python", cfg.PYTHON],
        env=exp.env(), cwd=cfg.FT, log=exp.log_dir / "prefill.log")
    require(exp.cache_dir, "dataset cache")
    return exp.cache_dir


# ---------------------------------------------------------------- 5. train ---

def train(exp: cfg.Experiment, gpu: str = "0", wait: bool = True) -> Path:
    """Fine-tune from the CARLA pretrain checkpoint on the prepared split."""
    _prepare(exp)
    announce(f"[5/6] train  ->  {exp.checkpoint_dir}")
    cmd: list[str | Path] = [
        cfg.PYTHON, "-u", cfg.FT / "train" / "run_plant2_finetune.py",
        "--split", exp.split_dir,
        "--learning-rate", exp.learning_rate,
        "--checkpoint-addon", exp.name,
        "--cuda-device", gpu,
        "--ds-local", exp.cache_dir,
        "--cache-size-gb", "200",
        "--batch-size", str(exp.batch_size),
        "--num-workers", str(exp.num_workers),
        "--max-epochs", str(exp.max_epochs),
        "--ckpt-every-n-epochs", "3",
        "--no-filter-routes",
        "--resume-ckpt", cfg.PRETRAIN,
        "--wandb-mode", "offline",
        "--python", cfg.PYTHON,
        "--hydra-run-dir", cfg.PLANT2 / "PlanT" / "outputs" / "PlanT2_train" / exp.name,
        "--log", exp.log_dir / "train.log",
    ]
    cmd.append("--augment" if exp.augment else "--no-augment")
    for override in exp.training_overrides():
        cmd += ["--hydra-override", override]

    proc = run_background(cmd, env=exp.env(), cwd=cfg.FT,
                          log=exp.log_dir / "train.nohup.log")
    (exp.log_dir / "train.pid").write_text(str(proc.pid))
    if wait:
        print("  waiting for training (detached: interrupting this does not "
              "stop it)")
        wait_for([proc])
        require(exp.checkpoint_dir, "checkpoint directory")
    return exp.checkpoint_dir


def last_checkpoint(exp: cfg.Experiment) -> Path:
    """The final epoch's weights.

    Always `last`, never `best`. `best` is selected by ``val/loss_all``, which
    is dominated by the ego-speed term: it rises monotonically while the path
    loss plateaus, so the "best" epoch is an almost-untrained one. Measured:
    `best` checkpoints score stop compliance 1.000 at success 0.000 and
    efficiency 8.8 — a car that barely moves satisfies a stop sign perfectly.
    """
    candidates = sorted(exp.checkpoint_dir.glob("last_ft_*.ckpt"))
    if not candidates:
        raise SystemExit(f"No last_ft_*.ckpt in {exp.checkpoint_dir}")
    return candidates[-1]


# ----------------------------------------------------------------- 6. eval ---

def evaluate(exp: cfg.Experiment, checkpoint: Path | None = None,
             gpu: str = "0", shards: int = 4) -> str:
    """Score the checkpoint on the held-out scenes of every family."""
    checkpoint = checkpoint or last_checkpoint(exp)
    _prepare(exp)
    announce(f"[6/6] eval  {checkpoint.name}  ->  {exp.eval_dir}")

    for name in exp.families:
        family = cfg.FAMILIES[name]
        run([cfg.PYTHON, "-u", cfg.FT / "eval" / "eval_full.py", "fv",
             "--ckpt", checkpoint,
             "--out", exp.eval_dir / name,
             "--manifest", exp.plan_dir / name / "test_manifest.jsonl",
             "--scenes", family.scenes,
             "--gpus", gpu,
             "--nshards", str(shards), "--concurrency", str(shards),
             "--exclude-codes", ""],
            env=exp.env(), log=exp.log_dir / f"eval_{name}.log")

    table = metrics.report(exp)
    exp.eval_dir.mkdir(parents=True, exist_ok=True)
    (exp.eval_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return table
