"""Target-sign registry for dataset statistics.

Maps PDD codes to:
  - OSM catalog folders under ``pdd-bench/scenes/``
  - per-sign benchmark packages under ``scripts/per_sign_bench/``
  - optional external speed-sign catalogs (smirnova run_v61_a6)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PDD_BENCH_ROOT = Path(__file__).resolve().parents[2]
SCENES_ROOT = PDD_BENCH_ROOT / "scenes"
PER_SIGN_ROOT = PDD_BENCH_ROOT / "scripts" / "per_sign_bench"

# Speed-sign balanced catalog (A6) from the shared smirnova tree.
# Prefer map-trimmed balanced catalog (~1.2k scenarios / sign) when present;
# fall back to the full catalog.jsonl.
_SPEED_RUN = Path(
    "/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench"
    "/benchmark_output_speed/balanced/run_v61_a6"
)
_SPEED_BALANCED = _SPEED_RUN / "catalog_balanced_1k2.jsonl"
_SPEED_FULL = _SPEED_RUN / "catalog.jsonl"
SPEED_CATALOG_JSONL = _SPEED_BALANCED if _SPEED_BALANCED.exists() else _SPEED_FULL
SPEED_METRICS_CSV = _SPEED_RUN / "eval_fast" / "metrics_per_episode.csv"
SPEED_MAPS_ROOT = Path(
    "/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench/maps"
)
# Observed max final_step in speed eval = 1500.
SPEED_HORIZON_STEPS = 1500

# Detour signs 4.2.1–4.2.3 (smirnova detour_v1).
DETOUR_CATALOG_JSONL = Path(
    "/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench"
    "/benchmark_output/detour_v1/catalog.jsonl"
)
DETOUR_METRICS_CSV = Path(
    "/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench"
    "/benchmark_output/detour_v1/eval_fast/metrics_per_episode.csv"
)
# Observed max final_step in detour eval_fast = 1200.
DETOUR_HORIZON_STEPS = 1200

# MetaDrive control frequency used throughout per_sign_bench (s).
DT_SECONDS = 0.1
# Matches lib/traffic_density_levels.py META_DENSITY_SCALE.
META_DENSITY_SCALE = 80.0


@dataclass(frozen=True)
class SignSpec:
    pdd_code: str
    name: str
    category: str
    package: Optional[str]
    # Folder name under package/scenes/ and package/benchmark_output/
    package_sign_dir: Optional[str]
    # Top-level OSM catalog folder(s) under pdd-bench/scenes/
    catalog_dirs: tuple[str, ...]
    # How agents are configured in manifests
    # aux_convoy | density | density_ped | catalog_only | speed_ego | detour_ego
    agent_mode: str
    # Optional external catalog.jsonl (speed / detour signs)
    external_catalog: Optional[Path] = None
    default_horizon_steps: Optional[int] = None
    metrics_csv: Optional[Path] = None


# Reviewer-facing subset requested by the user (+ speed signs).
TARGET_SIGNS: tuple[SignSpec, ...] = (
    SignSpec("2.1", "Main road", "Priority", "main_sign", "2_1", ("2.1",), "aux_convoy"),
    SignSpec(
        "2.3.1",
        "Secondary road (2.3.1)",
        "Priority",
        "secondary_sign",
        "2_3",
        ("2.3.1",),
        "aux_convoy",
    ),
    SignSpec(
        "2.3.2",
        "Secondary road (2.3.2)",
        "Priority",
        "secondary_sign",
        "2_3",
        ("2.3.2",),
        "aux_convoy",
    ),
    SignSpec(
        "2.3.3",
        "Secondary road (2.3.3)",
        "Priority",
        "secondary_sign",
        "2_3",
        ("2.3.3",),
        "aux_convoy",
    ),
    SignSpec("2.4", "Yield", "Priority", "yield_sign", "2_4", ("2.4",), "aux_convoy"),
    SignSpec("2.5", "Stop", "Priority", "stop_sign", "2_5", ("2.5",), "aux_convoy"),
    SignSpec("3.1", "No entry", "Prohibitory", "no_entry_signs", "3_1", ("3.1",), "density"),
    SignSpec(
        "3.2",
        "Movement prohibited",
        "Prohibitory",
        "no_entry_signs",
        "3_2",
        ("3.2",),
        "density",
    ),
    SignSpec(
        "3.24",
        "Speed limit",
        "Prohibitory",
        None,
        None,
        ("3.24",),
        "speed_ego",
        external_catalog=SPEED_CATALOG_JSONL,
        default_horizon_steps=SPEED_HORIZON_STEPS,
        metrics_csv=SPEED_METRICS_CSV,
    ),
    SignSpec(
        "4.3",
        "Roundabout",
        "Mandatory",
        "roundabout_sign",
        "4_3",
        ("4.3",),
        "aux_convoy",
    ),
    SignSpec(
        "4.2.1",
        "Detour (right)",
        "Mandatory",
        None,
        None,
        ("4.2.1",),
        "detour_ego",
        external_catalog=DETOUR_CATALOG_JSONL,
        default_horizon_steps=DETOUR_HORIZON_STEPS,
        metrics_csv=DETOUR_METRICS_CSV,
    ),
    SignSpec(
        "4.2.2",
        "Detour (left)",
        "Mandatory",
        None,
        None,
        ("4.2.2",),
        "detour_ego",
        external_catalog=DETOUR_CATALOG_JSONL,
        default_horizon_steps=DETOUR_HORIZON_STEPS,
        metrics_csv=DETOUR_METRICS_CSV,
    ),
    SignSpec(
        "4.2.3",
        "Detour (both)",
        "Mandatory",
        None,
        None,
        ("4.2.3",),
        "detour_ego",
        external_catalog=DETOUR_CATALOG_JSONL,
        default_horizon_steps=DETOUR_HORIZON_STEPS,
        metrics_csv=DETOUR_METRICS_CSV,
    ),
    SignSpec(
        "4.6",
        "Min speed limit",
        "Mandatory",
        None,
        None,
        ("4.6",),
        "speed_ego",
        external_catalog=SPEED_CATALOG_JSONL,
        default_horizon_steps=SPEED_HORIZON_STEPS,
        metrics_csv=SPEED_METRICS_CSV,
    ),
    SignSpec(
        "5.7.1",
        "One-way (right)",
        "Special",
        "one_way_signs",
        "5_7_1",
        ("5.7.1",),
        "density",
    ),
    SignSpec(
        "5.7.2",
        "One-way (left)",
        "Special",
        "one_way_signs",
        "5_7_2",
        ("5.7.2",),
        "density",
    ),
    SignSpec(
        "5.15.1",
        "Lane directions (start)",
        "Special",
        "lane_direction_signs",
        "5_15_1",
        (),  # not mirrored under top-level scenes/
        "density",
    ),
    SignSpec(
        "5.19",
        "Pedestrian crossing",
        "Special",
        "crosswalk_sign",
        "5_19",
        ("5.19",),
        "density_ped",
    ),
    SignSpec(
        "5.21",
        "Living zone start",
        "Special",
        None,
        None,
        (),  # not present under local scenes/
        "speed_ego",
        external_catalog=SPEED_CATALOG_JSONL,
        default_horizon_steps=SPEED_HORIZON_STEPS,
        metrics_csv=SPEED_METRICS_CSV,
    ),
    SignSpec(
        "5.31",
        "Zone with speed limit",
        "Special",
        None,
        None,
        ("5.31",),
        "speed_ego",
        external_catalog=SPEED_CATALOG_JSONL,
        default_horizon_steps=SPEED_HORIZON_STEPS,
        metrics_csv=SPEED_METRICS_CSV,
    ),
)


def package_scenes_dir(spec: SignSpec) -> Optional[Path]:
    if not spec.package or not spec.package_sign_dir:
        return None
    return PER_SIGN_ROOT / spec.package / "scenes" / spec.package_sign_dir


def package_final_manifest(spec: SignSpec) -> Optional[Path]:
    if not spec.package or not spec.package_sign_dir:
        return None
    return (
        PER_SIGN_ROOT
        / spec.package
        / "benchmark_output"
        / spec.package_sign_dir
        / "final_metrics_v1"
        / "real_manifest.jsonl"
    )


def _package_eval_base(spec: SignSpec) -> Optional[Path]:
    if not spec.package or not spec.package_sign_dir:
        return None
    return (
        PER_SIGN_ROOT
        / spec.package
        / "benchmark_output"
        / spec.package_sign_dir
        / "final_metrics_v1"
    )


def package_eval_agg(spec: SignSpec) -> Optional[Path]:
    """Prefer per-sign agg, then overall; ``eval_out_all`` then ``eval_out`` / ``eval_out_test``."""
    base = _package_eval_base(spec)
    if base is None:
        return None
    for sub in ("eval_out_all", "eval_out", "eval_out_test"):
        for name in ("agg_per_sign_baseline.csv", "agg_per_baseline.csv"):
            cand = base / sub / "aggregations" / name
            if cand.exists():
                return cand
    return None


def package_metrics_csv(spec: SignSpec) -> Optional[Path]:
    base = _package_eval_base(spec)
    if base is None:
        return None
    for sub in ("eval_out_all", "eval_out", "eval_out_test"):
        cand = base / sub / "metrics_per_episode.csv"
        if cand.exists():
            return cand
    return None
