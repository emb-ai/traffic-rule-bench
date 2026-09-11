"""MetaDrive dual-path route probe: same loop check as eval RouteValidation.

Crop/manifest call this so scenes that would score ``ok=false`` never fill the
200-row cap. Probe both dests: baseline (default ``set_route``) and compliant
(``install_one_way_compliant_nav_route``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from traffic_bench.eval.engine.map.lane_keys import make_lane_key

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PULSE_SERVER", "none")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

_ENV_CACHE: dict[str, Any] = {}
_RESULT_CACHE: dict[tuple, "DualPathRouteProbeResult"] = {}


def metadrive_route_loops(checkpoints, spawn_lane_idx) -> bool:
    """True when nav has no real path (same predicate as eval ``RouteValidation``)."""
    if not checkpoints or spawn_lane_idx is None:
        return True
    ck = list(checkpoints)
    return len(ck) <= 1 or ck[-1] == spawn_lane_idx or ck[0] == ck[-1]


@dataclass(frozen=True)
class DualPathRouteProbeResult:
    ok: bool
    baseline_ok: bool
    compliant_ok: bool
    reason: str = ""
    baseline_detail: str = ""
    compliant_detail: str = ""

    @property
    def failed_modes(self) -> tuple[str, ...]:
        failed: list[str] = []
        if not self.baseline_ok:
            failed.append("baseline")
        if not self.compliant_ok:
            failed.append("compliant")
        return tuple(failed)


def _lane_id(edge_id: str, lane_num: int) -> str:
    raw = str(edge_id)
    if raw.startswith("lane_"):
        return raw
    return make_lane_key(raw, int(lane_num))


def _normalize_lane_id(raw: Optional[str], *, edge_id: str = "", lane_num: int = 0) -> str:
    text = str(raw or "").strip()
    if text:
        return text if text.startswith("lane_") else f"lane_{text}"
    if edge_id:
        return _lane_id(edge_id, lane_num)
    return ""


def _cache_key(
    net_path: Path,
    *,
    road_id: str,
    spawn_lane_num: int,
    baseline_dest: str,
    compliant_dest: str,
    blocked: tuple[str, ...],
) -> tuple:
    return (
        str(Path(net_path).resolve()),
        str(road_id),
        int(spawn_lane_num),
        str(baseline_dest),
        str(compliant_dest),
        blocked,
    )


def _close_env(env) -> None:
    if env is None:
        return
    for name in ("close", "destroy"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def close_probe_envs() -> None:
    """Drop cached MetaDrive envs (call after a crop/manifest batch)."""
    for key in list(_ENV_CACHE):
        _close_env(_ENV_CACHE.pop(key, None))


def invalidate_probe_env(net_path: Path | str) -> None:
    """Drop env + result cache for a net that was rewritten in place."""
    key = str(Path(net_path).resolve())
    _close_env(_ENV_CACHE.pop(key, None))
    for cached_key in list(_RESULT_CACHE):
        if cached_key and cached_key[0] == key:
            _RESULT_CACHE.pop(cached_key, None)


def _probe_row(
    *,
    net_path: Path,
    road_id: str,
    spawn_lane_num: int,
    baseline_dest: str,
    compliant_dest: str,
    dual_path: dict,
    background_excluded_edges: Optional[list[str]] = None,
    sign_family: str = "no_turn",
) -> dict:
    net_path = Path(net_path).resolve()
    excluded = list(
        background_excluded_edges
        if background_excluded_edges is not None
        else (dual_path.get("wrong_dir_edges") or [])
    )
    return {
        "net_path": str(net_path),
        "road_id": str(road_id),
        "spawn_lane_num": int(spawn_lane_num),
        "destination_lane_id": baseline_dest,
        "destination_edge_id": "",
        "baseline_destination_lane_id": baseline_dest,
        "compliant_destination_lane_id": compliant_dest,
        "sign_family": sign_family,
        "sign_type": sign_family,
        "dual_path": dict(dual_path),
        "background_excluded_edges": excluded,
        "traffic_density": 0.0,
        "skip_auto_signs": True,
        "spawn_distance_before_end": 20.0,
        "horizon": 50,
    }


def _ensure_metadrive_patch() -> None:
    from traffic_bench.eval.engine.sim.metadrive_sumo_patch import apply_metadrive_sumo_via_patch

    apply_metadrive_sumo_via_patch()


def _env_for_net(net_path: Path, row: dict):
    from traffic_bench.eval.run.env import _build_sumo_env

    key = str(Path(net_path).resolve())
    env = _ENV_CACHE.get(key)
    if env is not None:
        return env
    # MetaDrive allows one engine per process; drop the previous net first.
    close_probe_envs()
    scenes_root = Path(net_path).resolve().parent
    probe_row = dict(row)
    probe_row["net_path"] = str(Path(net_path).resolve())
    env = _build_sumo_env(probe_row, scenes_root=scenes_root, max_steps=50)
    _ENV_CACHE[key] = env
    return env


def _checkpoints_after_setup(env, row: dict, *, compliant: bool):
    from traffic_bench.eval.run.env import (
        _apply_manifest_ego_destination,
        _apply_manifest_ego_spawn_lane,
    )
    from traffic_bench.eval.signs.dual_path.nav import install_one_way_compliant_nav_route
    from traffic_bench.eval.signs.dual_path.place import resolve_row_for_policy

    env.reset(seed=0)
    policy = "carl_rule" if compliant else "carl"
    bound = resolve_row_for_policy(row, policy)
    if not _apply_manifest_ego_spawn_lane(env, bound):
        return None, None, "spawn_failed"
    _apply_manifest_ego_destination(env, bound)
    if compliant:
        if not install_one_way_compliant_nav_route(env, bound):
            return None, None, "compliant_nav_failed"
    vehicle = getattr(env, "vehicle", None) or getattr(env, "agent", None)
    nav = getattr(vehicle, "navigation", None) if vehicle is not None else None
    spawn_idx = getattr(getattr(vehicle, "lane", None), "index", None)
    checkpoints = list(getattr(nav, "checkpoints", None) or []) if nav is not None else []
    return checkpoints, spawn_idx, ""


def probe_dual_path_geometry(
    net_path: Path | str,
    *,
    road_id: str,
    spawn_lane_num: int,
    baseline_dest_lane_id: str,
    compliant_dest_lane_id: str,
    dual_path: dict,
    background_excluded_edges: Optional[list[str]] = None,
    sign_family: str = "no_turn",
) -> DualPathRouteProbeResult:
    """Load the cropped net once and test baseline + compliant MetaDrive routes."""
    net = Path(net_path)
    baseline_dest = _normalize_lane_id(baseline_dest_lane_id)
    compliant_dest = _normalize_lane_id(compliant_dest_lane_id) or baseline_dest
    blocked = tuple(
        sorted(
            str(e)
            for e in (
                background_excluded_edges
                if background_excluded_edges is not None
                else (dual_path.get("wrong_dir_edges") or [])
            )
            if e
        )
    )
    key = _cache_key(
        net,
        road_id=str(road_id),
        spawn_lane_num=int(spawn_lane_num),
        baseline_dest=baseline_dest,
        compliant_dest=compliant_dest,
        blocked=blocked,
    )
    cached = _RESULT_CACHE.get(key)
    if cached is not None:
        return cached

    _ensure_metadrive_patch()
    row = _probe_row(
        net_path=net,
        road_id=str(road_id),
        spawn_lane_num=int(spawn_lane_num),
        baseline_dest=baseline_dest,
        compliant_dest=compliant_dest,
        dual_path=dict(dual_path or {}),
        background_excluded_edges=list(blocked),
        sign_family=sign_family,
    )
    baseline_ok = False
    compliant_ok = False
    baseline_detail = ""
    compliant_detail = ""
    try:
        env = _env_for_net(net, row)
        ck, spawn, err = _checkpoints_after_setup(env, row, compliant=False)
        if err:
            baseline_detail = err
        elif metadrive_route_loops(ck, spawn):
            baseline_detail = (
                f"loop spawn={spawn} dest={baseline_dest} "
                f"checkpoints={(ck or [])[:3]}"
            )
        else:
            baseline_ok = True
        ck, spawn, err = _checkpoints_after_setup(env, row, compliant=True)
        if err:
            compliant_detail = err
        elif metadrive_route_loops(ck, spawn):
            compliant_detail = (
                f"loop spawn={spawn} dest={compliant_dest} "
                f"checkpoints={(ck or [])[:3]}"
            )
        else:
            compliant_ok = True
    except Exception as exc:
        baseline_detail = baseline_detail or f"probe_error:{type(exc).__name__}: {exc}"
        compliant_detail = compliant_detail or baseline_detail

    ok = baseline_ok and compliant_ok
    parts = []
    if not baseline_ok:
        parts.append(f"baseline:{baseline_detail or 'loop'}")
    if not compliant_ok:
        parts.append(f"compliant:{compliant_detail or 'loop'}")
    result = DualPathRouteProbeResult(
        ok=ok,
        baseline_ok=baseline_ok,
        compliant_ok=compliant_ok,
        reason="; ".join(parts),
        baseline_detail=baseline_detail,
        compliant_detail=compliant_detail,
    )
    _RESULT_CACHE[key] = result
    return result


def probe_cropped_dual_path(
    net_path: Path | str, scenario, *, sign_family: str = "no_turn"
) -> DualPathRouteProbeResult:
    """Crop-time check: shared dest, both nav modes on the rebuilt net."""
    dual = {
        "turn_path": list(getattr(scenario, "turn_path", ()) or getattr(scenario, "baseline_path", ()) or ()),
        "straight_path": list(
            getattr(scenario, "straight_path", ()) or getattr(scenario, "compliant_path", ()) or ()
        ),
        "wrong_dir_edges": list(getattr(scenario, "wrong_dir_edges", ()) or ()),
        "straight_first_exit": getattr(scenario, "straight_first_exit", None)
        or getattr(scenario, "compliant_first_exit", None),
        "turn_first_exit": getattr(scenario, "turn_first_exit", None)
        or getattr(scenario, "baseline_first_exit", None),
    }
    dest = _lane_id(str(scenario.dest_edge_id), int(scenario.dest_lane_num))
    return probe_dual_path_geometry(
        net_path,
        road_id=str(scenario.ego_edge_id),
        spawn_lane_num=int(scenario.ego_lane_num),
        baseline_dest_lane_id=dest,
        compliant_dest_lane_id=dest,
        dual_path=dual,
        background_excluded_edges=list(getattr(scenario, "wrong_dir_edges", ()) or ()),
        sign_family=sign_family,
    )


def probe_manifest_entry(entry: dict, *, net_path: Path | str) -> DualPathRouteProbeResult:
    """Manifest-time check on the row's (possibly truncated) dests."""
    dual = dict(entry.get("dual_path") or {})
    road_id = str(entry.get("road_id") or "")
    spawn_num = int(entry.get("spawn_lane_num", 0) or 0)
    baseline = str(
        entry.get("baseline_destination_lane_id") or entry.get("destination_lane_id") or ""
    )
    compliant = str(
        entry.get("compliant_destination_lane_id") or entry.get("destination_lane_id") or ""
    )
    family = str(entry.get("sign_family") or entry.get("sign_type") or "no_turn")
    return probe_dual_path_geometry(
        net_path,
        road_id=road_id,
        spawn_lane_num=spawn_num,
        baseline_dest_lane_id=baseline,
        compliant_dest_lane_id=compliant,
        dual_path=dual,
        background_excluded_edges=list(entry.get("background_excluded_edges") or []),
        sign_family=family,
    )


def probe_scene_dir(scene_dir: Path, *, pdd_code: str = "3.18.2") -> DualPathRouteProbeResult:
    """Probe crop meta spawn/dest (used to auto-reject live scene folders)."""
    scene_dir = Path(scene_dir)
    meta_path = scene_dir / "meta.json"
    if not meta_path.is_file():
        return DualPathRouteProbeResult(
            ok=False, baseline_ok=False, compliant_ok=False, reason="no_meta"
        )
    import json

    from traffic_bench.eval.signs.dual_path.scene import dual_path_scenario_from_meta
    from traffic_bench.eval.signs.dual_path.spec import get_spec

    try:
        family = get_spec(pdd_code).family
    except ValueError:
        family = "direction"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    net_file = meta.get("net_file") or "map.net.xml"
    net_path = scene_dir / net_file
    if not net_path.is_file():
        return DualPathRouteProbeResult(
            ok=False, baseline_ok=False, compliant_ok=False, reason="no_net"
        )
    scenario = dual_path_scenario_from_meta(meta, pdd_code=pdd_code)
    if scenario is None:
        dest = _normalize_lane_id(
            meta.get("destination_lane_id"),
            edge_id=str(meta.get("destination_edge_id") or ""),
            lane_num=0,
        )
        dual = dict(meta.get("dual_path") or {})
        return probe_dual_path_geometry(
            net_path,
            road_id=str(meta.get("road_id") or ""),
            spawn_lane_num=int(meta.get("spawn_lane_num") or 0),
            baseline_dest_lane_id=dest,
            compliant_dest_lane_id=dest,
            dual_path=dual,
            background_excluded_edges=list(meta.get("background_excluded_edges") or []),
            sign_family=family,
        )
    return probe_cropped_dual_path(net_path, scenario, sign_family=family)


def reject_unroutable_dual_path_scenes(
    scenes_root: Path,
    *,
    pdd_code: str = "3.18.2",
    apply: bool = True,
    skip_keep: bool = True,
    record: bool = True,
) -> list[tuple[str, DualPathRouteProbeResult]]:
    """Probe every live scene; reject (and optionally move) those that loop.

    ``skip_keep``: already-``keep`` dirs are not re-probed (faster refill loops).
    ``record``: write keep/reject into ``scene_selection.json`` (off for dry-run).
    """
    from traffic_bench.scene_collection.sign_scenes.filter.selection import (
        VERDICT_KEEP,
        apply_rejected_scenes,
        is_reserved_scene_dir,
        set_scene_reject,
        set_scene_verdict,
        scene_verdict,
    )

    scenes_root = Path(scenes_root)
    rejected: list[tuple[str, DualPathRouteProbeResult]] = []
    n_ok = 0
    n_skip = 0
    try:
        for entry in sorted(scenes_root.iterdir()):
            if not entry.is_dir() or is_reserved_scene_dir(entry.name):
                continue
            if not (entry / "meta.json").is_file():
                continue
            if skip_keep and scene_verdict(scenes_root, entry.name) == VERDICT_KEEP:
                n_skip += 1
                continue
            print(f"[route-probe] {entry.name} …", flush=True)
            result = probe_scene_dir(entry, pdd_code=pdd_code)
            if result.ok:
                n_ok += 1
                print(f"[route-probe] {entry.name} ok", flush=True)
                if record:
                    set_scene_verdict(scenes_root, entry.name, VERDICT_KEEP)
                continue
            print(f"[route-probe] REJECT {entry.name}: {result.reason}", flush=True)
            if record:
                set_scene_reject(
                    scenes_root,
                    entry.name,
                    reason="metadrive_route_loop",
                    detail=result.reason,
                )
            rejected.append((entry.name, result))
    finally:
        close_probe_envs()
    if n_skip:
        print(f"[route-probe] skipped {n_skip} already-keep scene(s)")
    print(f"[route-probe] ok={n_ok} reject={len(rejected)}")
    if apply and record and rejected:
        moved, n = apply_rejected_scenes(
            scenes_root, only=[name for name, _ in rejected]
        )
        print(f"[route-probe] moved {moved}/{n} rejected scene(s) to _rejected/")
    return rejected


def _split_counts(scenes_root: Path) -> tuple[int, int, int]:
    from traffic_bench.eval.manifest.io import apply_split_filter, discover_scenes

    all_s = discover_scenes(scenes_root)
    train, _ = apply_split_filter(all_s, scenes_dir=scenes_root, split="train")
    test, _ = apply_split_filter(all_s, scenes_dir=scenes_root, split="test")
    return len(all_s), len(train), len(test)


def _run_materialize_refill(sign: str) -> int:
    import subprocess
    import sys

    from traffic_bench.scene_collection.paths import REPO_ROOT

    cmd = [
        sys.executable,
        "-m",
        "traffic_bench.scene_collection",
        "materialize",
        "--sign",
        sign,
        "--refill",
    ]
    print("[route-probe] Running:", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd, cwd=str(REPO_ROOT)))


def fill_dual_path_quota(
    *,
    sign: str,
    scenes_root: Path,
    pdd_code: str,
    max_loops: int = 10,
) -> int:
    """Probe+apply, then materialize --refill, until no MD rejects and quotas hold."""
    from traffic_bench.eval.manifest.io import sign_split_quota

    n_train_q = sign_split_quota(pdd_code, "train") or 0
    n_test_q = sign_split_quota(pdd_code, "test") or 0
    last_rejects = 0
    for i in range(1, max(1, int(max_loops)) + 1):
        live, n_tr, n_te = _split_counts(scenes_root)
        print(
            f"[route-probe] loop {i}/{max_loops}: live={live} "
            f"train={n_tr}/{n_train_q} test={n_te}/{n_test_q}",
            flush=True,
        )
        # First pass probes every live dir (keep from visual review can still loop).
        found = reject_unroutable_dual_path_scenes(
            scenes_root,
            pdd_code=pdd_code,
            apply=True,
            skip_keep=i > 1,
            record=True,
        )
        last_rejects = len(found)
        live, n_tr, n_te = _split_counts(scenes_root)
        at_quota = n_tr >= n_train_q and n_te >= n_test_q
        if last_rejects == 0 and at_quota:
            print("[route-probe] stable: all live maps routable and quotas filled")
            return 0
        rc = _run_materialize_refill(sign)
        if rc != 0:
            print(f"[route-probe] warn: refill exited {rc}")
        if last_rejects == 0 and not at_quota:
            live2, n_tr2, n_te2 = _split_counts(scenes_root)
            if n_tr2 == n_tr and n_te2 == n_te:
                print(
                    "[route-probe] refill added nothing; crop pool may be exhausted"
                )
                return 1
    print(
        f"[route-probe] stopped after {max_loops} loops "
        f"(last_rejects={last_rejects}); pool may be exhausted"
    )
    return 1


_DUAL_PATH_FAMILY_ALIASES = {
    "direction": "direction",
    "no_turn": "no_turn",
    "one_way": "one_way",
}


def resolve_route_probe_targets(sign: str) -> list[tuple[str, Path, str]]:
    """``(sign_id, scenes_root, pdd_code)`` for one eval sign or a dual-path family.

    ``direction`` / ``no_turn`` / ``one_way`` expand to every profile in that family.
    """
    from traffic_bench.eval.sign_registry import (
        get_profile,
        hydra_sign_override,
        list_profiles,
        scenes_dir as profile_scenes_dir,
    )

    raw = str(sign).strip()
    try:
        profile = get_profile(raw)
    except KeyError:
        fam = _DUAL_PATH_FAMILY_ALIASES.get(raw.replace("/", "_").strip("_").lower())
        matched = [
            p
            for p in list_profiles()
            if fam and p.family == "dual_path" and p.sign_type == fam
        ]
        if not matched:
            raise
        profiles = matched
    else:
        profiles = [profile]
    return [
        (hydra_sign_override(p), profile_scenes_dir(p), str(p.pdd_code))
        for p in profiles
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe dual-path scenes and reject MetaDrive loop routes"
    )
    parser.add_argument(
        "scenes_root",
        nargs="?",
        type=Path,
        help="Scenes dir (default: data/scenes/<sign> when --sign is set)",
    )
    parser.add_argument(
        "--sign",
        default=None,
        help=(
            "Eval sign id (no_turn/left, direction/straight, …) or family "
            "(direction, no_turn, one_way)"
        ),
    )
    parser.add_argument("--pdd-code", default=None)
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Dry-run: print rejects, do not write selection or move dirs",
    )
    parser.add_argument(
        "--refill",
        action="store_true",
        help="After apply, top up via materialize --refill (requires --sign)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat probe→apply→refill until stable (implies --refill; requires --sign)",
    )
    parser.add_argument("--max-loops", type=int, default=10)
    args = parser.parse_args()

    targets: list[tuple[str, Path, str]] = []
    if args.sign:
        if args.scenes_root is not None or args.pdd_code is not None:
            try:
                from traffic_bench.eval.sign_registry import get_profile

                get_profile(args.sign)
            except KeyError:
                raise SystemExit(
                    "family --sign (direction / no_turn / one_way) cannot be "
                    "combined with scenes_root or --pdd-code"
                )
        targets = resolve_route_probe_targets(str(args.sign))
        if args.scenes_root is not None:
            targets = [(targets[0][0], Path(args.scenes_root), targets[0][2])]
        if args.pdd_code is not None:
            targets = [(targets[0][0], targets[0][1], str(args.pdd_code))]
    elif args.scenes_root is not None and args.pdd_code is not None:
        targets = [("", Path(args.scenes_root), str(args.pdd_code))]
    else:
        raise SystemExit("need --sign, or scenes_root plus --pdd-code")

    if args.loop:
        if not args.sign:
            raise SystemExit("--loop requires --sign")
        if args.no_apply:
            raise SystemExit("--loop cannot be combined with --no-apply")
        rc = 0
        for sign_id, scenes, pdd in targets:
            print(f"[route-probe] === {sign_id} ({pdd}) ===", flush=True)
            rc |= fill_dual_path_quota(
                sign=str(sign_id),
                scenes_root=scenes,
                pdd_code=str(pdd),
                max_loops=int(args.max_loops),
            )
        raise SystemExit(rc)

    n_reject = 0
    for sign_id, scenes, pdd in targets:
        label = sign_id or str(scenes)
        print(f"[route-probe] === {label} ({pdd}) ===", flush=True)
        found = reject_unroutable_dual_path_scenes(
            scenes,
            pdd_code=str(pdd),
            apply=not args.no_apply,
            skip_keep=False,
            record=not args.no_apply,
        )
        n_reject += len(found)
        if args.refill:
            if not sign_id:
                raise SystemExit("--refill requires --sign")
            if args.no_apply:
                raise SystemExit("--refill cannot be combined with --no-apply")
            rc = _run_materialize_refill(str(sign_id))
            if rc:
                raise SystemExit(rc)
        print(f"[route-probe] {label}: rejected {len(found)} scene(s)")
    raise SystemExit(1 if n_reject else 0)
