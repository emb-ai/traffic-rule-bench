"""Everything the pipeline needs to know, in one place.

Two rules hold throughout this package:

1. **No silent fallbacks.** Every path is declared here and checked before a
   stage runs. A missing path stops the run with a message naming it, instead
   of a default that quietly points somewhere else. (The old
   ``lib/env.py:resolve_python`` walked a list of candidate interpreters and
   picked whichever existed — it once selected an unrelated system python and
   the eval ran to completion with wrong results.)

2. **Result-changing environment variables are set here, not by the caller.**
   ``PLANT2_SIGN_RANGE_M``, ``PLANT2_DUMP_SIGN_CLASSES`` and ``PLANT2_YLEFT``
   each silently change what the model sees or how it steers. Forgetting one
   produces plausible-looking numbers that cannot be compared with anything.
   They are derived from the experiment below and printed before every
   subprocess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMIRNOVA = Path("/home/jovyan/shares/SR006.nfs2/smirnova")

PYTHON = Path("/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python")
PLANT2 = ROOT / "plant2"
BENCH = ROOT / "pdd-bench" / "scripts" / "per_sign_bench"
FT = ROOT / "scripts" / "plant2_ft_pipeline"

# --- the oracle trajectory collection -------------------------------------
#
# One winning expert run per scene, picked across four rule-following policies
# (idm_rule / carl_rule / plant2_rule / ppo_rule). Each row carries absolute
# `pkl_path` and `sidecar_path`, so the dump replays a recording rather than
# re-simulating.
ORACLE = SMIRNOVA / "traj_full" / "20260829_004215"
EXPERTS = ORACLE / "experts_all" / "experts_scene_uid_top1.jsonl"
#: Maps live per family: <SCENES>/<family>/<scene_id>/map.net.xml
SCENES = SMIRNOVA / "traffic-rule-bench-main" / "data" / "scenes"

#: CARLA-pretrained weights every fine-tune starts from. On its own this
#: checkpoint scores 0.000 on the detour benchmark and hits the obstacle in
#: 100% of runs, so it is a genuine baseline rather than a warm start.
PRETRAIN = ROOT / "stop_data" / "checkpoints" / "plant2_pretrain" / "epoch=029_final_3.ckpt"

#: Sign visibility radius (m) the MODEL is trained and evaluated with.
#:
#: `dataset.py` re-filters sign boxes at load time using
#: model.training.sign_range_m, so training has always been capped at 30 m
#: whatever the dump contained. Inference, going through the same
#: `collect_boxes`, used whatever PLANT2_SIGN_RANGE_M said — 90 m in the older
#: runs. That mismatch fed the model sign tokens at distances it never saw:
#: measured on the stop dump, 26% of sign boxes sit beyond 30 m.
#: Both sides now take this one value.
SIGN_RANGE_M = 90

#: Radius the DUMP records at. Deliberately wider than SIGN_RANGE_M: training
#: discards the surplus anyway, and keeping it on disk means raising
#: SIGN_RANGE_M later costs a retrain rather than a re-dump.
DUMP_SIGN_RANGE_M = 90

#: Lateral frame. The dump writes route/path/waypoint targets as y=LEFT and
#: object boxes as y=RIGHT; inference must be told so. With this off, route
#: following still works but every obstacle is passed on the wrong side —
#: measured honest compliance 0.000 vs 0.879 on the same checkpoint.
YLEFT = "1"

#: Pose jitter recorded into the dump: the BEV is re-rendered from a viewpoint
#: this far off the expert trajectory, and `aug_sample` shifts the labels by the
#: same amount. Left at 0 the `--augment` flag is a silent no-op — the dumper
#: writes 0.0 and copies the plain BEV, which is what it did for months while
#: every run was nominally "augmented". Non-zero here is what actually teaches
#: recovery from an off-nominal pose; it cut off-road runs from 0.75 to 0.16.
AUG_TRANSLATION_M = 1.0
AUG_ROTATION_DEG = 5.0


@dataclass(frozen=True)
class Family:
    """One sign family of the oracle collection."""

    name: str      #: directory name, both in ORACLE and under SCENES
    sign: str      #: PDD code as it appears in x_objs
    group: str     #: how it is reported: stop / detour / speed

    @property
    def catalog(self) -> Path:
        return ORACLE / self.name / "catalog.jsonl"

    @property
    def scenes(self) -> Path:
        return SCENES / self.name


FAMILIES: dict[str, Family] = {f.name: f for f in (
    Family("stop", "2.5", "stop"),
    Family("detour_right", "4.2.1", "detour"),
    Family("detour_left", "4.2.2", "detour"),
    Family("detour_either", "4.2.3", "detour"),
    Family("speed_limit", "3.24", "speed"),
    Family("min_speed", "4.6", "speed"),
    Family("zone_speed_limit", "5.31", "speed"),
)}

#: Shorthand so `--families detour` expands to the three detour signs.
GROUPS: dict[str, tuple[str, ...]] = {
    "stop": ("stop",),
    "detour": ("detour_right", "detour_left", "detour_either"),
    "speed": ("speed_limit", "min_speed", "zone_speed_limit"),
}


def resolve_families(names: list[str]) -> tuple[str, ...]:
    """Expand group names, keep family names, reject anything else."""
    out: list[str] = []
    for name in names:
        if name in GROUPS:
            out += list(GROUPS[name])
        elif name in FAMILIES:
            out.append(name)
        else:
            raise SystemExit(
                f"Unknown family or group: {name}. "
                f"Groups: {sorted(GROUPS)}. Families: {sorted(FAMILIES)}")
    return tuple(dict.fromkeys(out))


@dataclass
class Experiment:
    """One dump -> split -> train -> eval cycle.

    Every field that changes the result is here. Nothing is read from the
    ambient environment.
    """

    name: str
    families: tuple[str, ...] = tuple(FAMILIES)

    #: Scenes held out of training entirely and used only for the final
    #: metrics. Split per family and by SCENE, so no scene contributes to both.
    test_fraction: float = 0.2
    split_seed: int = 0
    #: Of the remaining scenes, the share used for validation during training.
    val_fraction: float = 0.15

    # --- training ---
    learning_rate: str = "3e-4"
    max_epochs: int = 24
    batch_size: int = 512
    num_workers: int = 4
    augment: bool = True
    #: Duplicate frames whose nearest cone is within [min, max] metres, this
    #: many times. Only detour scenes contain cones, so this is a no-op for the
    #: rest. The lane-change decision happens ~50 m before the cone, so a
    #: window ending at 5 m (the obvious first guess) flags the wrong frames.
    oversample_cone_m: tuple[float, float] | None = None
    #: Same window, measured to the SIGN. This is the one that bites on the
    #: oracle scenes: a cone is visible in 4% of detour frames there, a sign in
    #: 87%, so the cone rule duplicated 725 frames where the older data had
    #: 42050. 40-70 m is where the manoeuvre is decided — the expert starts its
    #: lane change a median 51.5 m before the obstacle.
    oversample_sign_m: tuple[float, float] | None = (40.0, 70.0)
    oversample_factor: int = 8
    hydra_overrides: tuple[str, ...] = ()

    work: Path = field(init=False)

    def __post_init__(self) -> None:
        self.work = ROOT / "plant2_pipeline" / "runs" / self.name

    # --- derived paths ---
    @property
    def plan_dir(self) -> Path:
        return self.work / "plan"

    @property
    def dump_dir(self) -> Path:
        return self.work / "dump"

    @property
    def split_dir(self) -> Path:
        return self.work / "split"

    @property
    def cache_dir(self) -> Path:
        """Dataset disk cache. Local disk, not NFS: 0.09 ms/file vs 0.57 ms."""
        return Path("/tmp") / f"plant2_cache_{self.name}"

    @property
    def checkpoint_dir(self) -> Path:
        return PLANT2 / "PlanT" / "checkpoints_ft" / self.name

    @property
    def eval_dir(self) -> Path:
        return self.work / "eval"

    @property
    def log_dir(self) -> Path:
        return self.work / "logs"

    # --- derived settings ---
    @property
    def sign_classes(self) -> tuple[str, ...]:
        """Every PDD code that may become an x_objs token.

        A code not listed here is invisible to the model. Dumping a joint model
        with only the detour codes gives frames where the stop sign is simply
        absent, and the model then cannot learn it at any amount of training.
        """
        return tuple(dict.fromkeys(FAMILIES[f].sign for f in self.families))

    def env(self, *, dumping: bool = False) -> dict[str, str]:
        """Environment for a child process.

        `dumping=True` records the wider radius; every other stage — training
        and, crucially, inference — uses SIGN_RANGE_M so the two agree.
        """
        return {
            "PLANT2_DUMP_SIGN_CLASSES": ",".join(self.sign_classes),
            "PLANT2_SIGN_RANGE_M": str(DUMP_SIGN_RANGE_M if dumping else SIGN_RANGE_M),
            "PLANT2_YLEFT": YLEFT,
            "PLANT2_AUG_TRANSLATION_M": str(AUG_TRANSLATION_M if self.augment else 0.0),
            "PLANT2_AUG_ROTATION_DEG": str(AUG_ROTATION_DEG if self.augment else 0.0),
            # timm fetches resnet18 weights from HF on model init; the proxy
            # here fails intermittently and has killed a run mid-training.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }

    def training_overrides(self) -> list[str]:
        # `dataset.py` re-filters sign boxes at load time using
        # model.training.sign_range_m, whose default is 30. Without this line
        # the dump keeps signs out to SIGN_RANGE_M, inference shows them out to
        # SIGN_RANGE_M, and training alone throws away everything past 30 m --
        # measured on the stop dump, 26% of all sign boxes. The model would
        # then meet, at inference, sign tokens at distances it never saw.
        out = [f"user.working_dir={PLANT2}",
               f"model.training.sign_range_m={SIGN_RANGE_M}"]
        if self.oversample_cone_m is not None:
            lo, hi = self.oversample_cone_m
            out += [f"model.training.oversample_cone_distance_min_m={lo}",
                    f"model.training.oversample_cone_distance_m={hi}"]
        if self.oversample_sign_m is not None:
            lo, hi = self.oversample_sign_m
            out += [f"model.training.oversample_sign_distance_min_m={lo}",
                    f"model.training.oversample_sign_distance_m={hi}"]
        if self.oversample_cone_m is not None or self.oversample_sign_m is not None:
            out.append(f"model.training.oversample_factor={self.oversample_factor}")
        out += list(self.hydra_overrides)
        return out


def check_inputs(exp: Experiment) -> None:
    """Fail before doing any work if something the run needs is missing."""
    required: dict[str, Path] = {
        "python": PYTHON,
        "pretrain checkpoint": PRETRAIN,
        "PlanT2 repo": PLANT2 / "PlanT",
        "benchmark scripts": BENCH,
        "fine-tune scripts": FT,
        "oracle experts jsonl": EXPERTS,
    }
    for name in exp.families:
        family = FAMILIES[name]
        required[f"{name} catalog"] = family.catalog
        required[f"{name} scenes"] = family.scenes

    missing = [f"  {label}: {path}" for label, path in required.items()
               if not path.exists()]
    if missing:
        raise SystemExit("Missing inputs:\n" + "\n".join(missing))


def describe(exp: Experiment) -> str:
    window = "  ".join(
        [f"cone {exp.oversample_cone_m}"] if exp.oversample_cone_m else []
        + [f"sign {exp.oversample_sign_m}"] if exp.oversample_sign_m else []) or "off"
    window += f"  x{exp.oversample_factor}"
    return "\n".join([
        f"experiment   {exp.name}",
        f"families     {', '.join(exp.families)}",
        f"signs        {', '.join(exp.sign_classes)}",
        f"holdout      test {exp.test_fraction:.0%}   val {exp.val_fraction:.0%}"
        f"   seed {exp.split_seed}",
        f"training     lr {exp.learning_rate}   epochs {exp.max_epochs}   "
        f"batch {exp.batch_size}   augment {exp.augment}",
        f"oversample   {window}",
        f"work dir     {exp.work}",
        "env          " + "  ".join(f"{k}={v}" for k, v in exp.env().items()
                                    if k.startswith("PLANT2")),
    ])


def load_catalog(family: Family) -> list[dict]:
    return [json.loads(line) for line in
            family.catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
