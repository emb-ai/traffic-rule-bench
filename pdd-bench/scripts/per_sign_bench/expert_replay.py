"""Record policy replays (trajectories) on benchmark scenes.

For each scene in a benchmark manifest, runs the chosen ego policy through the
scene with MetaDrive's native RecordManager on, then:
  * dumps a ScenarioDescription pkl (ego+NPC+map+pedestrians+cyclists+TL)
  * writes a sidecar JSON with the sign, per-step violations, expert actions,
    and the original source row (so the scene can be rebuilt in-env).

The episode itself is SHARED with the corrected eval — env construction comes
from bench.env_builders.build_env_for_row, the ego policy from
bench.policy_factory.make_ego_policy, and the step loop + all metric tracking
from run_benchmark._run_rollout / build_sidecar_metrics. This file only adds
the recording glue (record_episode=True, RecordManager tolerance patch, pkl
dump) around that loop, so recorder metrics are the eval metrics by
construction.

Usage:
    # rule expert with 4 sampled IDM variants on a SUMO manifest:
    python expert_replay.py --manifest <m.jsonl> --code 2.5 --backend sumo \\
        --policy comprehensive_rule_expert --ego-extra-samples 4 \\
        --count 5 --output-dir <run_root>

    # NN policy (needs --model-path):
    python expert_replay.py --manifest <m.jsonl> --backend sumo \\
        --policy plant2_rule --model-path <ckpt> --output-dir <run_root>

    # build oracle manifest from merged all_runs.jsonl:
    python expert_replay.py --build-oracle-manifest --run-dir <merged_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
SCENES_ROOT = PDD_BENCH_DIR / "scenes"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Shared with the corrected eval: bench/ helpers + the run_benchmark rollout.
# bench._paths (imported transitively) completes the sys.path setup.
from bench.util import (_row_seed, _row_sign_code, _seed_everything,  # noqa: E402
                        _unwrap_base_env)
from bench.env_builders import build_env_for_row  # noqa: E402
from bench import env_builders as _env_builders  # noqa: E402
from bench.policy_factory import _load_policy_models, make_ego_policy  # noqa: E402
from run_benchmark import (Rollout, _run_rollout,  # noqa: E402,F401
                           build_sidecar_metrics)

# Legacy-layout fallback: per_sign_bench/benchmark_output/ preferred, but we also
# read from sibling benchmark/benchmark_output/ when per_sign_bench has none.
LEGACY_OUTPUT_DIR = SCRIPTS_DIR / "benchmark"


# Same policy set as run_benchmark.py — resolved via bench.policy_factory.
POLICY_NAMES = ("idm", "comprehensive_rule_expert", "rule_compliant",
                "ppo_lidar", "carl", "carl_rule", "plant2", "plant2_rule")

# Policies whose ego gets sampled s1..sN IDM variants (make_ego_policy).
IDM_VARIANT_POLICIES = {"idm", "comprehensive_rule_expert"}
# Policies whose ego is relocated onto the sign lane by default (mirror of
# run_benchmark.main's --relocate-ego-to-sign-lane=auto).
IDM_FAMILY = {"idm", "comprehensive_rule_expert", "rule_compliant"}

# Old recorder name → corrected-eval name. NOTE: old "carl"/"plant2" meant the
# sign-compliant classes — those are now carl_rule/plant2_rule, while the bare
# names mean the plain (sign-unaware) policies. Only "comprehensive" is an
# unambiguous alias; use the *_rule names explicitly for the old behavior.
LEGACY_POLICY_ALIASES = {"comprehensive": "comprehensive_rule_expert"}


# ---------------------------------------------------------------------------
# RecordManager tolerance patch
# ---------------------------------------------------------------------------
# Between env.reset() finishing (RecordManager.after_reset clears reset_frame
# and current_frames to None) and the first env.step() (before_step creates a
# new current_frames), RecordManager has no active frame — but our
# TrafficSignManager.add_sign and sumo_env.reset both spawn sign objects in
# that window, which crashes on `current_frame.spawn_info` assertion.
#
# Sign objects are static — we don't need them in the recording (they're in
# the sidecar). So we monkey-patch RecordManager.add_spawn_info at module
# load to be a no-op when no frame is active. This is done once per process
# and only affects expert_replay.py / expert_replay_inenv.py processes.

_RM_PATCHED = False

def _patch_record_manager_once():
    global _RM_PATCHED
    if _RM_PATCHED:
        return
    try:
        from metadrive.manager.record_manager import RecordManager
        from metadrive.utils.utils import is_map_related_class
        from metadrive.constants import ObjectState
    except ImportError as exc:
        # MetaDrive not importable yet — try again after env is built.
        print(f"[warn] RecordManager patch deferred: {exc}")
        return

    def _tolerant_add_spawn_info(self, obj, object_class, kwargs):
        if is_map_related_class(object_class) or not self.engine.record_episode:
            return
        # If no frame is currently active (between reset and first step), skip
        # silently — this is the case for sign spawns that happen post-reset.
        if self.reset_frame is None and self.current_frames is None:
            return
        try:
            frame = self.current_frame
        except (TypeError, AttributeError):
            return
        name = obj.name
        if name in frame.spawn_info:
            return  # idempotent
        self._episode_obj_names.add(name)
        frame.spawn_info[name] = {
            ObjectState.CLASS: object_class,
            ObjectState.INIT_KWARGS: kwargs,
            ObjectState.NAME: name,
        }

    RecordManager.add_spawn_info = _tolerant_add_spawn_info

    # collect_objects_states iterates ALL engine.get_objects() on every step.
    # Traffic-sign objects are not map-related (so they're not filtered out),
    # but they have no physics body → `obj.get_state()` → `obj.velocity` →
    # `self._body.hasPythonTag(...)` crashes with 'NoneType has no attribute'.
    # Skip such bodyless objects safely.
    import copy as _copy

    def _tolerant_collect_objects_states(self):
        from metadrive.utils.utils import is_map_related_instance
        policy_mapping = self.engine.get_policies()
        frame = self.current_frame
        for name, obj in self.engine.get_objects().items():
            if is_map_related_instance(obj):
                continue
            if getattr(obj, "_body", None) is None:
                continue  # static / bodyless objects (e.g. traffic signs)
            try:
                frame.step_info[name] = obj.get_state()
            except Exception:
                continue
            if name in policy_mapping:
                try:
                    frame.policy_info[name] = policy_mapping[name].get_state()
                except Exception:
                    pass
        frame.agents = list(self.engine.agents.keys())
        frame._agent_to_object = _copy.deepcopy(self.engine.agent_manager._agent_to_object)
        frame._object_to_agent = _copy.deepcopy(self.engine.agent_manager._object_to_agent)

    RecordManager.collect_objects_states = _tolerant_collect_objects_states

    # add_policy_info: pedestrian_manager (and other managers) call
    # engine.add_policy() during reset (e.g. _spawn_due_tracks → _spawn_pedestrian
    # → add_policy(ScenarioNetLikeReplayPolicy)). At that point reset_frame may
    # be None and current_frames is None — accessing self.current_frame crashes
    # with `'NoneType' object is not subscriptable`. Skip silently when no
    # frame is active (the policy is still registered in engine.get_policies()
    # via add_policy itself; we only skip the recording side-effect).
    try:
        from metadrive.constants import PolicyState
        from metadrive.base_class.base_object import BaseObject
    except ImportError:
        PolicyState = None
        BaseObject = None

    def _tolerant_add_policy_info(self, name, policy_class, *args, **kwargs):
        if not self.engine.record_episode:
            return
        if self.reset_frame is None and self.current_frames is None:
            return
        try:
            frame = self.current_frame
        except (TypeError, AttributeError):
            return
        if name in frame.policy_spawn_info:
            return  # idempotent
        # Filter BaseObject args/kwargs (match original record_manager logic).
        filtered_args = []
        for arg in args:
            if BaseObject is not None and isinstance(arg, BaseObject):
                filtered_args.append(BaseObject)
            else:
                filtered_args.append(arg)
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if BaseObject is not None and isinstance(v, BaseObject):
                filtered_kwargs[k] = BaseObject
            else:
                filtered_kwargs[k] = v
        if PolicyState is not None:
            frame.policy_spawn_info[name] = {
                PolicyState.POLICY_CLASS: policy_class,
                PolicyState.ARGS: filtered_args,
                PolicyState.KWARGS: filtered_kwargs,
                PolicyState.OBJ_NAME: name,
            }

    RecordManager.add_policy_info = _tolerant_add_policy_info
    _RM_PATCHED = True

# Apply at import so everything downstream (including env.reset calls inside
# expert_replay_inenv.py) is covered.
_patch_record_manager_once()


# ---------------------------------------------------------------------------
# Env builder — thin compat wrapper over bench.env_builders.build_env_for_row.
# ---------------------------------------------------------------------------

def _build_env(row: dict, backend: str, max_steps: int,
                record_episode: bool = True, ego_policy_cls=None,
                render: bool = False, scenes_root=None):
    """Build an UNRESET env for one scene via the shared bench builder.

    Kept for expert_replay_inenv.py, which re-uses this signature. Post-reset
    sign placement (pgmap/paired/citymap) is the caller's job — the recorder
    uses build_env_for_row's post_reset directly; in-env replay re-adds signs
    from the sidecar instead.
    """
    env, _env_seed, _post_reset = build_env_for_row(
        row, backend, scenes_root=Path(scenes_root or SCENES_ROOT),
        max_steps=max_steps)
    env.config["record_episode"] = bool(record_episode)
    if render:
        env.config["use_render"] = True
    if ego_policy_cls is not None:
        env.config["agent_policy"] = ego_policy_cls
    return env


# ---------------------------------------------------------------------------
# Record one scene
# ---------------------------------------------------------------------------

def record_expert_replay(
    row: dict, backend: str,
    output_pkl: Path, output_sidecar: Path,
    *,
    policy_type: str, models: dict,
    max_steps: int = 600,
    save_gif: Optional[Path] = None,
    ego_variant: str = "default",
    ego_sample_seed_base: int = 42,
    sample_ego_velocity: bool = False,
    spawn_velocity_override_ms: Optional[float] = None,
    scenes_root: Optional[Path] = None,
    extra_sidecar: Optional[dict] = None,
    plant2_dir: Optional[Path] = None,
) -> dict:
    """Run one policy on one scene with RecordManager on; write pkl + sidecar.

    Mirrors run_benchmark.run_one_episode step for step — same seeding, env
    from bench.env_builders.build_env_for_row, ego from make_ego_policy, step
    loop + metrics from _run_rollout — and only adds the recording glue
    (record_episode=True, post-reset sign spawns hidden from the RecordManager,
    ScenarioDescription pkl dump). Returns a flat row for all_runs.jsonl.

    Args:
        policy_type: one of POLICY_NAMES (run_benchmark --policy choices).
        models: result of bench.policy_factory._load_policy_models.
        ego_variant: "default" or "s1".."sN" — sampled per scene inside
            make_ego_policy (seed = ego_sample_seed_base + seed + k*1000003);
            only applies to IDM_VARIANT_POLICIES.
        sample_ego_velocity: sample ego spawn velocity from the nuPlan
            distribution (ignored for braking_spawn rows — the env
            reconstructs the upstream brake spawn itself).
        extra_sidecar: optional dict merged into the sidecar JSON (e.g.,
            policy/variant identifiers).
        plant2_dir: if set, live-dump PlanT2 boxes/measurements/results under
            ``<plant2_dir>/data/<scene_uid>_<variant>/`` during the rollout.
            ``None`` (default) leaves the code path identical to today.
    """
    from metadrive.scenario.utils import convert_recorded_scenario_exported

    seed = _row_seed(row)
    _seed_everything(seed)

    # Spawn velocity: braking rows get theirs from the env's upstream spawn —
    # never override those. Otherwise the explicit CLI override wins, else the
    # optional nuPlan sample replaces the manifest value.
    if not row.get("braking_spawn"):
        if spawn_velocity_override_ms is not None:
            row = dict(row, spawn_velocity_ms=float(spawn_velocity_override_ms))
        elif sample_ego_velocity:
            from factorized_space.agent_profile_bank import sample_spawn_velocity
            row = dict(row, spawn_velocity_ms=sample_spawn_velocity(seed))

    env, env_seed, post_reset = build_env_for_row(
        row, backend, scenes_root=Path(scenes_root or SCENES_ROOT),
        max_steps=max_steps)
    # RecordManager reads this off global_config on reset — no builder change
    # needed to record.
    env.config["record_episode"] = True

    reason = None

    # Make sure the RecordManager patch is in place — module-load attempt can
    # fail if MetaDrive wasn't yet importable. By now env construction has
    # imported everything, so the patch should stick.
    _patch_record_manager_once()

    try:
        env.reset(seed=env_seed)
        base_env = _unwrap_base_env(env)
        if hasattr(base_env, "engine") and hasattr(base_env.engine, "np_random"):
            base_env.engine.np_random = np.random.RandomState(seed)
        veh = base_env.vehicle

        # Capture ego initial speed right after reset for sidecar verification.
        try:
            initial_speed_mps = float(getattr(veh, "speed", 0.0))
            initial_speed_kmh = float(getattr(veh, "speed_km_h", initial_speed_mps * 3.6))
            print(f"[ego] initial speed: {initial_speed_mps:.2f} m/s "
                  f"({initial_speed_kmh:.1f} km/h)  scene={row.get('scene_id')}")
        except Exception:
            initial_speed_mps = 0.0
            initial_speed_kmh = 0.0

        # Backend-specific post-reset setup (pgmap/paired/citymap sign
        # placement; sumo is a no-op) — run_benchmark parity, but under a
        # RecordManager guard: between env.reset() and the first env.step()
        # RecordManager has no active frame, and sign objects have no physics
        # body, so:
        #   * add_spawn_info is temporarily a no-op during placement, and
        #   * just-spawned bodyless objects are detached from
        #     engine._spawned_objects so collect_objects_states never iterates
        #     them (they stay alive in traffic_sign_manager.signs and in the
        #     sidecar JSON).
        _rm = getattr(base_env.engine, "record_manager", None)
        _rm_original_add_spawn = None
        _signs_pre = set(base_env.engine._spawned_objects.keys())
        if _rm is not None:
            _rm_original_add_spawn = _rm.add_spawn_info
            _rm.add_spawn_info = lambda *a, **kw: None
        try:
            setup_error = post_reset(base_env)
        finally:
            _signs_post = set(base_env.engine._spawned_objects.keys())
            for _sid in _signs_post - _signs_pre:
                obj = base_env.engine._spawned_objects.get(_sid)
                body = getattr(obj, "_body", None) if obj is not None else None
                if obj is not None and body is None:
                    base_env.engine._spawned_objects.pop(_sid, None)
            # Restore add_spawn_info so NPC spawns during stepping get
            # captured into the recording.
            if _rm is not None and _rm_original_add_spawn is not None:
                _rm.add_spawn_info = _rm_original_add_spawn

        if setup_error:
            return _fail_result(row, backend, output_pkl, output_sidecar, setup_error)

        # Ego policy — identical to run_benchmark.run_one_episode: hold-v0 on
        # braking spawns; s* variants sampled per scene inside make_ego_policy.
        # No agent_policy in the env config → the engine's own act() call runs
        # the default EnvInputPolicy, and the manual policy_obj.act() below is
        # the ONLY expert step per frame.
        ego_hold_speed_ms = None
        if row.get("braking_spawn"):
            try:
                ego_hold_speed_ms = float(row.get("spawn_velocity_ms") or 0.0) or None
            except (TypeError, ValueError):
                ego_hold_speed_ms = None
        policy_obj, sampled_ego_params = make_ego_policy(
            policy_type, models, base_env, seed,
            ego_variant=ego_variant, ego_sample_seed_base=ego_sample_seed_base,
            ego_hold_speed_ms=ego_hold_speed_ms)

        # Optional PlanT2 live dump — capture pre-action frames via step_hook.
        plant2_collector = None
        step_hook = None
        plant2_path: Optional[str] = None
        if plant2_dir is not None:
            from bench.plant2_frames import (
                Plant2FrameCollector, ensure_slurm_dummy,
            )
            ensure_slurm_dummy(Path(plant2_dir))
            plant2_collector = Plant2FrameCollector(row)
            step_hook = lambda: plant2_collector.on_step(base_env, row)

        # The shared eval step loop — in-zone tracking, step+event violation
        # counts, dt=0.1 kinematics, TTC, smoothness, driving score etc. all
        # come from here (and from build_sidecar_metrics below).
        r = _run_rollout(env, base_env, policy_obj,
                         max_steps=max_steps, save_gif=save_gif,
                         step_hook=step_hook)

        # Dump episode data.
        # Try ScenarioDescription conversion first (works for PGMap); fallback
        # to raw FrameInfo format (works for any env; replayable via
        # env.config["replay_episode"] = episode_info).
        output_pkl.parent.mkdir(parents=True, exist_ok=True)
        scenario_desc = None
        try:
            raw_frames = base_env.engine.record_manager.episode_info
            scenario_desc = convert_recorded_scenario_exported(raw_frames, to_dict=True)
        except Exception:
            scenario_desc = None

        if scenario_desc is not None:
            with open(output_pkl, "wb") as f:
                pickle.dump(scenario_desc, f)
        else:
            # Fallback: save raw episode_info (FrameInfo list) — replay via
            # env.config["replay_episode"]=pickle.load(f); env.reset().
            try:
                episode_info = base_env.engine.dump_episode(str(output_pkl))
                scenario_desc = episode_info  # non-None → marks valid
            except Exception as exc:
                reason = f"dump_episode: {type(exc).__name__}: {exc}"
                scenario_desc = None

        # GIF (frames were rendered inside _run_rollout when save_gif is set).
        if save_gif:
            try:
                save_gif.parent.mkdir(parents=True, exist_ok=True)
                renderer = getattr(base_env, "top_down_renderer", None)
                if renderer is not None:
                    renderer.generate_gif(str(save_gif), duration=40)
            except Exception:
                pass

        # Identity — EXACT run_benchmark formulas so recorder outputs join
        # against eval episodes on (scene_uid, sign_slug, policy, variant).
        sign_slug = str(_row_sign_code(row) or "").replace(".", "_")
        scene_id_out = row.get("scene_id") or f"scene_{seed}"
        scene_uid = _scene_uid(row)

        if plant2_collector is not None and plant2_dir is not None:
            from bench.plant2_frames import plant2_route_dir
            route_dir = plant2_route_dir(Path(plant2_dir), scene_uid, ego_variant)
            success = bool(r.reached_dest and not r.crashed_flag_raw and not r.out_of_road)
            plant2_collector.flush(route_dir, success=success)
            plant2_path = str(route_dir)

        metrics = build_sidecar_metrics(r)
        # Recorder-only telemetry on top of the shared schema:
        metrics["initial_speed_mps"] = float(initial_speed_mps)
        metrics["initial_speed_kmh"] = float(initial_speed_kmh)

        # Sidecar — run_benchmark.run_one_episode layout + recorder extras
        # (real pkl_path, top-level violations_timeline for expert_replay_inenv).
        sidecar = {
            "scene_id": scene_id_out,
            "scene_uid": scene_uid,
            "backend": backend,
            "pdd_code": (row.get("pdd_code") or row.get("sign_code")
                         or row.get("sign_type")),
            "sign_key": row.get("sign_type"),
            "sign_slug": sign_slug,
            "source_row": row,
            "env_config_summary": {
                "map_name": row.get("net_path"),
                "road_id": row.get("road_id"),
                "spawn_lane_num": row.get("spawn_lane_num"),
                "lane_num": row.get("lane_num"),
                "lane_width": row.get("lane_width"),
                "horizon": max_steps,
                "seed": seed,
            },
            "signs": r.sign_info_snapshot,
            "expert_actions": r.expert_actions,
            "violations_timeline": list(r.violations_timeline),
            "smoothness_step_vars": r.smoothness_step_vars,
            "metrics": metrics,
            "dump_error": reason,
            "ego_idm_params": (sampled_ego_params if sampled_ego_params is not None
                               else "DEFAULT_EGO_PARAMS"),
            "pkl_path": str(output_pkl) if scenario_desc is not None else None,
            "sidecar_path": str(output_sidecar),
            "valid": bool(scenario_desc is not None),
        }
        if plant2_path is not None:
            sidecar["plant2_path"] = plant2_path
        if extra_sidecar:
            sidecar.update(extra_sidecar)
        output_sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(output_sidecar, "w") as f:
            json.dump(sidecar, f, default=str)

        # Flat all_runs.jsonl row: identity + the full metrics block minus the
        # bulky violations_timeline (it stays in the sidecar).
        flat_metrics = {k: v for k, v in metrics.items()
                        if k != "violations_timeline"}
        flat_row = {
            "scene_id": scene_id_out,
            "scene_uid": scene_uid,
            "backend": backend,
            "sign_type": sidecar["sign_key"],
            "sign_code": sidecar["pdd_code"],
            "sign_slug": sign_slug,
            "seed": seed,
            "pkl_path": str(output_pkl) if scenario_desc is not None else None,
            "sidecar_path": str(output_sidecar),
            "gif_path": str(save_gif) if save_gif else None,
            "valid": bool(scenario_desc is not None),
            **flat_metrics,
            "ego_idm_params": (sampled_ego_params if sampled_ego_params is not None
                               else "DEFAULT_EGO_PARAMS"),
            "failure_reason": reason,
        }
        if plant2_path is not None:
            flat_row["plant2_path"] = plant2_path
        return flat_row
    finally:
        try:
            env.close()
        except Exception:
            pass


def _fail_result(row, backend, output_pkl, output_sidecar, reason):
    return {
        "scene_id": row.get("scene_id") or row.get("flat_index"),
        "backend": backend,
        "valid": False,
        "failure_reason": reason,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_global(preset: str = "mini",
                      output_root: Optional[Path] = None) -> dict:
    """Scan benchmark_output/<preset>/*/expert/expert_summary.json and combine."""
    if output_root is None:
        for candidate in (BENCHMARK_DIR / "benchmark_output" / preset,
                           LEGACY_OUTPUT_DIR / "benchmark_output" / preset):
            if candidate.exists():
                output_root = candidate
                break
    if output_root is None or not output_root.exists():
        return {"error": f"no benchmark_output for preset={preset}"}

    per_sign = {}
    totals = Counter()
    violations_global = Counter()
    for code_dir in sorted(output_root.iterdir()):
        if not code_dir.is_dir():
            continue
        summary_path = code_dir / "expert" / "expert_summary.json"
        if not summary_path.exists():
            continue
        s = json.load(open(summary_path))
        per_sign[code_dir.name.replace("_", ".")] = s
        totals["n_episodes"] += s.get("n_episodes", 0)
        totals["n_valid"] += s.get("n_valid", 0)
        for k, v in s.get("violations_by_class", {}).items():
            violations_global[k] += v

    out = {
        "preset": preset,
        "total_signs": len(per_sign),
        "total_episodes": totals["n_episodes"],
        "total_valid": totals["n_valid"],
        "violations_top10": violations_global.most_common(10),
        "per_sign": per_sign,
    }
    with open(output_root / "expert_global_summary.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# ---------------------------------------------------------------------------
# Multi-variant trajectory recording: runs one policy with default + N
# nuPlan-sampled IDM variants, writes the new by_sign/by_scene layout, and
# appends one row per (scene × variant) to all_runs.jsonl. After all policies
# are recorded for a run, --build-oracle-manifest aggregates the jsonl rows
# into oracle_manifest.jsonl (best variant per scene_uid).
# ---------------------------------------------------------------------------


def _scene_uid(row: dict) -> str:
    """Stable per-episode identity — EXACT run_benchmark.run_one_episode formula
    (scene_id_lane<spawn_lane_num>_seed<seed>_v<var_idx>), so recorder outputs
    join against eval episodes/CSV on scene_uid."""
    seed = _row_seed(row)
    sid = row.get("scene_id") or f"scene_{seed}"
    lane = int(row.get("spawn_lane_num", 0) or 0)
    var_idx = int(row.get("var_idx", 0) or 0)
    return f"{sid}_lane{lane}_seed{seed}_v{var_idx}"


def _ego_score_key(record: dict) -> tuple:
    """Lower-is-better key for oracle selection.

    Priority: reached_dest desc → ego-fault crash asc → out_of_road asc →
    any crash asc → violation events asc → total_reward desc.

    Violations are ranked by EVENT count. New rows carry it under
    violations_event_count (their total_violations is per-step); legacy rows
    only have the event-counted total_violations — the fallback keeps both
    ranked on the same semantics.
    """
    reached = bool(record.get("arrived_dest", False))
    ego_crash = 1 if (record.get("crashed", False)
                       and record.get("crash_attribution") == "ego") else 0
    any_crash = 1 if record.get("crashed", False) else 0
    oor = 1 if record.get("out_of_road", False) else 0
    viol = int(record.get("violations_event_count",
                          record.get("total_violations", 0)) or 0)
    reward = float(record.get("total_reward", 0.0))
    return (-int(reached), ego_crash, oor, any_crash, viol, -reward)


def run_batch_multi_variant(
    manifest_path: Path, backend: str, run_root: Path,
    sign_slug: str, policy_type: str, models: dict,
    count: int, start: int, max_steps: int,
    extra_samples: int = 0, sample_seed_base: int = 42,
    save_gifs: bool = False,
    sample_ego_velocity: bool = False,
    skip_variants: Optional[set[str]] = None,
    scenes_root: Optional[Path] = None,
    spawn_velocity_override_ms: Optional[float] = None,
    plant2_dir: Optional[Path] = None,
) -> dict:
    """Record one policy across all scenes in the manifest, with `1 + extra_samples`
    variants per scene for IDM-family policies. Appends rows to <run_root>/all_runs.jsonl.

    Variants are labels only ("default", "s1"..); the actual IDM params are
    sampled per scene inside make_ego_policy (seed = sample_seed_base +
    scene_seed + k*1000003) — the corrected-eval semantics, NOT the old
    one-global-sample-per-k behavior.

    Output layout under run_root:
        by_sign/<sign_slug>/by_scene/<scene_uid>/<variant_id>/replay.pkl
        by_sign/<sign_slug>/by_scene/<scene_uid>/<variant_id>/replay.json
        all_runs.jsonl   (appended; one row per (scene × variant))
    """
    def _episode_already_recorded(pkl_path: Path, sidecar_path: Path,
                                   gif_path: Optional[Path],
                                   require_gif: bool) -> bool:
        if not (pkl_path.exists() and pkl_path.stat().st_size > 0):
            return False
        if not (sidecar_path.exists() and sidecar_path.stat().st_size > 0):
            return False
        if require_gif and gif_path is not None:
            if not (gif_path.exists() and gif_path.stat().st_size > 0):
                return False
        return True


    is_idm_based = policy_type in IDM_VARIANT_POLICIES
    n_extra = extra_samples if is_idm_based else 0

    rows = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if count is not None and len(rows) >= count:
                break
    if not rows:
        return {"error": f"no rows in {manifest_path}", "n": 0}

    sign_root = run_root / "by_sign" / sign_slug / "by_scene"
    sign_root.mkdir(parents=True, exist_ok=True)
    all_runs_path = run_root / "all_runs.jsonl"

    variants: list[str] = ["default"] + [f"s{k}" for k in range(1, n_extra + 1)]

    # Optionally drop variants the caller knows are already valid elsewhere
    # (used by the optim/scene_dispatcher.py to avoid recomputing IDM variants
    # that already have a valid replay for this scene_id at some other var_idx).
    # Accepts either bare ids ("default", "s2") or fully-qualified
    # ("comprehensive_rule_expert_s2"). Hierarchy is preserved because the
    # variant index k determines the sample seed — skipping one variant does
    # not shift the params of the others.
    if skip_variants:
        _skip_ids = set()
        for v in skip_variants:
            _skip_ids.add(v)
            if v.startswith(f"{policy_type}_"):
                _skip_ids.add(v[len(policy_type) + 1:])
        variants = [vid for vid in variants if vid not in _skip_ids]
        if not variants:
            return {"info": "all variants skipped via --skip-variants",
                    "n": 0, "skipped_existing": 0}

    n_total = len(rows) * len(variants)
    t0 = time.time()
    counter = 0
    valid_count = 0
    skipped_existing = 0

    # Resume mode    
    resume_mode = os.environ.get("PDD_BENCH_RESUME", "0") == "1"
    if resume_mode:
        print(f"[resume] PDD_BENCH_RESUME=1 — skipping already-recorded episodes",
              flush=True)

    with open(all_runs_path, "a") as ao:
        for i, row in enumerate(rows):
            scene_uid = _scene_uid(row)
            for variant_id in variants:
                counter += 1
                variant_full = f"{policy_type}_{variant_id}" if is_idm_based else policy_type
                out_dir = sign_root / scene_uid / variant_full
                pkl_path = out_dir / "replay.pkl"
                sidecar_path = out_dir / "replay.json"
                gif_path = (out_dir / "replay.gif") if save_gifs else None

                tag = (f"[{counter}/{n_total}] sign={sign_slug} scene={scene_uid} "
                       f"variant={variant_full}")

                if resume_mode and _episode_already_recorded(pkl_path, sidecar_path,
                                                             gif_path, save_gifs):
                    print(f"{tag}  [skip: already recorded]", flush=True)
                    skipped_existing += 1
                    continue

                print(tag, flush=True)

                try:
                    result = record_expert_replay(
                        row, backend, pkl_path, sidecar_path,
                        policy_type=policy_type, models=models,
                        max_steps=max_steps, save_gif=gif_path,
                        ego_variant=variant_id,
                        ego_sample_seed_base=sample_seed_base,
                        sample_ego_velocity=sample_ego_velocity,
                        spawn_velocity_override_ms=spawn_velocity_override_ms,
                        scenes_root=scenes_root,
                        plant2_dir=plant2_dir,
                        extra_sidecar={
                            "policy": policy_type,
                            "variant": variant_id,
                            "sign_slug": sign_slug,
                            "scene_uid": scene_uid,
                        },
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [ERROR] {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    result = {
                        "scene_id": row.get("scene_id"),
                        "valid": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }

                row_out = dict(result)
                row_out.update({
                    "policy": policy_type,
                    "variant": variant_id,
                    "sign_slug": sign_slug,
                    "scene_uid": scene_uid,
                    "pkl_path": str(pkl_path) if result.get("valid") else None,
                    "sidecar_path": str(sidecar_path),
                })
                ao.write(json.dumps(row_out, default=str) + "\n")
                ao.flush()
                if result.get("valid"):
                    valid_count += 1

    return {
        "sign_slug": sign_slug,
        "policy": policy_type,
        "n_total": n_total,
        "n_valid": valid_count,
        "wall_time_s": round(time.time() - t0, 1),
    }


def build_oracle_manifest(run_root: Path) -> dict:
    """Read all_runs.jsonl, pick best variant per scene_uid, write oracle_manifest.jsonl."""
    runs_path = run_root / "all_runs.jsonl"
    if not runs_path.exists():
        return {"error": f"no all_runs.jsonl under {run_root}"}

    by_scene: dict[str, list[dict]] = {}
    with open(runs_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("valid"):
                continue
            uid = r.get("scene_uid") or _scene_uid(r)
            by_scene.setdefault(uid, []).append(r)

    out_lines = []
    for uid in sorted(by_scene):
        candidates = by_scene[uid]
        keyed = sorted(
            candidates,
            key=lambda c: (_ego_score_key(c), c.get("policy", ""), c.get("variant", "")),
        )
        winner = keyed[0]
        out_lines.append({
            "scene_uid": uid,
            "sign_slug": winner.get("sign_slug"),
            "scene_id": winner.get("scene_id"),
            "policy": winner.get("policy"),
            "variant": winner.get("variant"),
            "winner_id": (f"{winner.get('policy')}_{winner.get('variant')}"
                          if winner.get("policy") in (IDM_VARIANT_POLICIES
                                                      | {"comprehensive"})
                          else winner.get("policy")),
            "winning_pkl": winner.get("pkl_path"),
            "winning_sidecar": winner.get("sidecar_path"),
            "score_key": list(_ego_score_key(winner)),
            "outcome": (
                "reached" if winner.get("arrived_dest") else
                "crash_ego" if (winner.get("crashed") and winner.get("crash_attribution") == "ego") else
                "crash_npc" if winner.get("crashed") else
                "out_of_road" if winner.get("out_of_road") else
                "timeout"
            ),
            "n_candidates": len(candidates),
        })

    manifest_path = run_root / "oracle_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for r in out_lines:
            f.write(json.dumps(r, default=str) + "\n")
    return {
        "n_scenes": len(out_lines),
        "manifest_path": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# Output-dir resolution
# ---------------------------------------------------------------------------

def _resolve_code_dir(code: str, preset: str) -> Optional[Path]:
    code_slug = code.replace(".", "_")
    for candidate in (BENCHMARK_DIR / "benchmark_output" / preset / code_slug,
                       LEGACY_OUTPUT_DIR / "benchmark_output" / preset / code_slug):
        if candidate.exists():
            return candidate
    return None


def _pick_manifest_and_backend(code_dir: Path, backend: str) -> tuple[Path, str]:
    """Prefer PGMap synthetic → paired → CityMap → SUMO catalog, unless user chose one."""
    candidates = [
        ("pgmap",  code_dir / "pgmap_materialized.jsonl"),
        ("pgmap",  code_dir / "synthetic_manifest.jsonl"),
        ("paired", code_dir / "paired_materialized.jsonl"),
        ("citymap",code_dir / "citymap_materialized.jsonl"),
        ("sumo",   code_dir / "sumo" / "sumo_manifest.jsonl"),
        ("sumo",   code_dir / "real_manifest.jsonl"),
    ]
    if backend != "auto":
        for b, p in candidates:
            if b == backend and p.exists() and p.stat().st_size > 0:
                return p, b
        raise FileNotFoundError(f"No {backend} manifest under {code_dir}")
    for b, p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p, b
    raise FileNotFoundError(f"No manifests at all under {code_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Record policy replays for one PDD code")
    parser.add_argument("--manifest", type=str, help="Path to *.jsonl manifest")
    parser.add_argument("--code", type=str, help="PDD code; auto-resolves manifest")
    parser.add_argument("--preset", type=str, default="mini")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "pgmap", "paired", "sumo", "citymap"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-gifs", action="store_true")
    parser.add_argument("--scenes-root", type=str, default=str(SCENES_ROOT),
                        help="Root dir manifest net_path entries resolve against "
                             "(default: pdd-bench/scenes)")
    parser.add_argument("--spawn-velocity-ms", type=float, default=None,
                        help="Override ego initial velocity (m/s). If unset, "
                             "uses row.spawn_velocity_ms from manifest (default 0). "
                             "Ignored for braking_spawn rows.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Only rebuild expert_global_summary.json from existing per-sign summaries")

    # Multi-variant trajectory recording
    parser.add_argument("--policy", type=str, default=None,
                        choices=list(POLICY_NAMES) + sorted(LEGACY_POLICY_ALIASES),
                        help="Ego policy — same choices as run_benchmark.py "
                             "('comprehensive' is a deprecated alias for "
                             "comprehensive_rule_expert). NOTE: the old recorder's "
                             "'carl'/'plant2' correspond to today's carl_rule/"
                             "plant2_rule; the bare names now mean the plain "
                             "(sign-unaware) policies.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Checkpoint path; required for carl/carl_rule/"
                             "plant2/plant2_rule (device auto-selected, control "
                             "GPU via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--plant2-action-mode", type=str, default="pid",
                        choices=["pid", "wps_pure_pursuit"],
                        help="How PlanT2 converts pred_plan -> action "
                             "(mirror of run_benchmark.py)")
    parser.add_argument("--relocate-ego-to-sign-lane", type=str, default="auto",
                        choices=["auto", "true", "false"],
                        help="After sign placement, teleport ego onto the sign-"
                             "topology lane. 'auto' (default) = True for idm/"
                             "comprehensive_rule_expert/rule_compliant, False for "
                             "NN policies — mirror of run_benchmark.py.")
    parser.add_argument("--ego-extra-samples", type=int, default=0,
                        help="N additional rollouts per scene with nuPlan-sampled ego "
                             "IDM params (only for idm/comprehensive_rule_expert). "
                             "Default 0: only the apply_ego_defaults baseline is recorded.")
    parser.add_argument("--ego-sample-seed-base", type=int, default=42,
                        help="Base seed for sampled IDM ego variants "
                             "(default 42 — run_benchmark parity)")
    parser.add_argument("--skip-variants", type=str, default=None,
                        help="Comma-separated list of IDM-family variants to NOT record. "
                             "Accepts bare ids ('default', 's2') or fully qualified "
                             "('comprehensive_rule_expert_s2'). Used by optim/"
                             "scene_dispatcher.py to avoid recomputing IDM variants "
                             "that already have a valid replay elsewhere.")
    parser.add_argument("--sample-ego-spawn-velocity", action="store_true",
                        help="Sample ego initial spawn velocity from nuPlan distribution "
                             "(sample_spawn_velocity(row.seed)) instead of using the "
                             "manifest value. Ignored for braking_spawn rows.")
    parser.add_argument("--save-plant2-dir", type=str, default=None,
                        help="If set, live-dump PlanT2 boxes/measurements/results under "
                             "<dir>/data/<scene_uid>_<variant>/ during recording. "
                             "Without this flag behaviour is unchanged.")
    parser.add_argument("--build-oracle-manifest", action="store_true",
                        help="Read <run-dir>/all_runs.jsonl and write oracle_manifest.jsonl")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run root for --build-oracle-manifest")

    args = parser.parse_args()

    if args.policy in LEGACY_POLICY_ALIASES:
        new_name = LEGACY_POLICY_ALIASES[args.policy]
        print(f"[deprecated] --policy {args.policy} → {new_name}")
        args.policy = new_name

    if args.aggregate:
        out = aggregate_global(preset=args.preset)
        print(json.dumps({"preset": out.get("preset"),
                           "total_signs": out.get("total_signs"),
                           "total_episodes": out.get("total_episodes"),
                           "total_valid": out.get("total_valid")}, indent=2))
        return

    if args.build_oracle_manifest:
        if not args.run_dir:
            print("ERROR: --build-oracle-manifest requires --run-dir", file=sys.stderr)
            sys.exit(1)
        out = build_oracle_manifest(Path(args.run_dir))
        print(json.dumps(out, indent=2))
        return

    backend = args.backend
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.code:
        code_dir = _resolve_code_dir(args.code, args.preset)
        if code_dir is None:
            print(f"ERROR: no benchmark output for code={args.code} preset={args.preset}",
                  file=sys.stderr)
            sys.exit(1)
        manifest_path, backend = _pick_manifest_and_backend(code_dir, backend)
    else:
        print("Need --manifest or --code", file=sys.stderr)
        sys.exit(1)

    if not args.policy:
        print("ERROR: --policy is required for recording", file=sys.stderr)
        sys.exit(1)
    if args.output_dir is None:
        print("ERROR: --policy mode requires --output-dir (run root)", file=sys.stderr)
        sys.exit(1)

    # relocate_ego_to_sign_lane — mirror of run_benchmark.main: NN policies
    # (plant2/carl/ppo_lidar) need ego left on the manifest road_id, not
    # teleported onto the sign lane. Must be set before any env is built.
    if args.relocate_ego_to_sign_lane == "auto":
        _env_builders.RELOCATE_EGO_TO_SIGN_LANE = args.policy in IDM_FAMILY
    else:
        _env_builders.RELOCATE_EGO_TO_SIGN_LANE = (args.relocate_ego_to_sign_lane == "true")
    print(f"relocate_ego_to_sign_lane: {_env_builders.RELOCATE_EGO_TO_SIGN_LANE}")

    logging.getLogger().setLevel(logging.CRITICAL)

    scenes_root = Path(args.scenes_root).resolve()
    if not scenes_root.exists():
        raise ValueError(f"Scenes root not found: {scenes_root}")

    # NN checkpoints load once per process (device auto: CUDA_VISIBLE_DEVICES).
    models = _load_policy_models(
        args.policy, args.model_path, plant2_action_mode=args.plant2_action_mode)

    run_root = Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    # sign_slug used in by_sign/<slug>/ — must match run_benchmark's
    # _row_sign_code-derived slug so outputs join across the two scripts.
    if args.code:
        sign_slug = args.code.replace(".", "_")
    else:
        sign_slug = None
        try:
            with open(manifest_path) as _mf:
                for _line in _mf:
                    _row = json.loads(_line)
                    _code = _row_sign_code(_row)
                    if _code:
                        sign_slug = str(_code).replace(".", "_")
                        break
        except Exception:
            pass
        if not sign_slug:
            sign_slug = manifest_path.parent.name

    print(f"Manifest:    {manifest_path}")
    print(f"Backend:     {backend}")
    print(f"Scenes root: {scenes_root}")
    print(f"Run root:    {run_root}")
    print(f"Policy:      {args.policy}  extra_samples={args.ego_extra_samples}")
    skip_variants = (
        {v.strip() for v in args.skip_variants.split(",") if v.strip()}
        if args.skip_variants else None
    )
    plant2_dir = Path(args.save_plant2_dir).resolve() if args.save_plant2_dir else None
    if plant2_dir is not None:
        plant2_dir.mkdir(parents=True, exist_ok=True)
        print(f"PlanT2 dir:  {plant2_dir}")
    out = run_batch_multi_variant(
        manifest_path, backend, run_root,
        sign_slug=sign_slug,
        policy_type=args.policy, models=models,
        count=args.count, start=args.start,
        max_steps=args.max_steps,
        extra_samples=args.ego_extra_samples,
        sample_seed_base=args.ego_sample_seed_base,
        save_gifs=args.save_gifs,
        sample_ego_velocity=args.sample_ego_spawn_velocity,
        skip_variants=skip_variants,
        scenes_root=scenes_root,
        spawn_velocity_override_ms=args.spawn_velocity_ms,
        plant2_dir=plant2_dir,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
