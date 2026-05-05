#!/usr/bin/env python3
"""
Diagnostic: run PlanT/HFLM closed-loop on TrafficSignEnv and, at every step
where any traffic sign token is visible in x_objs, print the ego-speed
classifier probability distribution.

Covers all sign types collected by collect_rule_expert_plant2_trajectories.py:
  stop(4)  speed_limit(7)  min_speed(8)  no_entry(9)  no_stopping(10)
  detour(11)  restricted_lane(12)  only_auto(13)

Environment mirrors check_rule_compliant_expert.py / collect_rule_expert_plant2_trajectories.py:
  - TrafficSignEnv  (no agent_policy — PlanT model drives)
  - Maps: X, T, S, O
  - One sign per episode, spawned on the vehicle's route lane

Output
------
  <output_dir>/ep<NNN>_seed<S>_<map>_<sign>.gif
  <output_dir>/speed_vs_dist_<sign_type>.png   (one plot per sign type)
  <output_dir>/top_bin_hist_<sign_type>.png
  <output_dir>/summary.json

Usage
-----
python eval_plant2_rule_sign_speed_probs.py \
    --checkpoint_file /path/to/plant2.ckpt \
    --num-episodes 13 \
    --start-seed 1000
"""

import json
import os
import sys
import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def _find_sdc_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "pdd-bench").is_dir() and (parent / "metadrive").is_dir():
            return parent
    raise RuntimeError("Could not locate SDC root")


FILE_PATH     = Path(__file__).resolve()
SDC_ROOT      = _find_sdc_root(FILE_PATH)
PDD_BENCH_DIR = SDC_ROOT / "pdd-bench"
METADRIVE_DIR = SDC_ROOT / "metadrive"
PLANT2_DIR    = SDC_ROOT / "plant2"
ADAPTER_PATH  = PDD_BENCH_DIR / "agents" / "carl_in_metadrive"

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR,
           PLANT2_DIR / "PlanT", PLANT2_DIR, ADAPTER_PATH):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# ---------------------------------------------------------------------------
# Sign / object-type tables (must stay in sync with metadrive_obs_to_plant2.py)
# ---------------------------------------------------------------------------

OBJ_TYPE_STOP_SIGN        = 4
OBJ_TYPE_TRAFFIC_LIGHT    = 5
OBJ_TYPE_SPEED_LIMIT_SIGN = 7
OBJ_TYPE_MIN_SPEED_SIGN   = 8
OBJ_TYPE_NO_ENTRY_SIGN    = 9
OBJ_TYPE_NO_STOPPING_SIGN = 10
OBJ_TYPE_DETOUR_SIGN      = 11
OBJ_TYPE_RESTRICTED_LANE  = 12
OBJ_TYPE_ONLY_AUTO_SIGN   = 13

# All sign token types we care about (vehicle / pedestrian / static are excluded)
ALL_SIGN_TYPES = (
    OBJ_TYPE_STOP_SIGN,
    OBJ_TYPE_SPEED_LIMIT_SIGN,
    OBJ_TYPE_MIN_SPEED_SIGN,
    OBJ_TYPE_NO_ENTRY_SIGN,
    OBJ_TYPE_NO_STOPPING_SIGN,
    OBJ_TYPE_DETOUR_SIGN,
    OBJ_TYPE_RESTRICTED_LANE,
    OBJ_TYPE_ONLY_AUTO_SIGN,
)

SIGN_TYPE_META: Dict[int, Dict] = {
    OBJ_TYPE_STOP_SIGN:        {"name": "stop_sign",        "color": "crimson",    "spawn_key": "stop"},
    OBJ_TYPE_SPEED_LIMIT_SIGN: {"name": "speed_limit",      "color": "steelblue",  "spawn_key": "speed40"},
    OBJ_TYPE_MIN_SPEED_SIGN:   {"name": "min_speed",        "color": "deepskyblue","spawn_key": "minspeed40"},
    OBJ_TYPE_NO_ENTRY_SIGN:    {"name": "no_entry",         "color": "darkorange", "spawn_key": "no_entry"},
    OBJ_TYPE_NO_STOPPING_SIGN: {"name": "no_stopping",      "color": "mediumpurple","spawn_key": "no_stop"},
    OBJ_TYPE_DETOUR_SIGN:      {"name": "detour",           "color": "seagreen",   "spawn_key": "detour_right"},
    OBJ_TYPE_RESTRICTED_LANE:  {"name": "restricted_lane",  "color": "saddlebrown","spawn_key": "bus_lane_road"},
    OBJ_TYPE_ONLY_AUTO_SIGN:   {"name": "only_auto",        "color": "dimgray",    "spawn_key": "only_auto"},
}

# Episode schedule: one (map, sign_spawn_key) pair per episode index, cycling.
# Mirrors DEFAULT_MAPS × DEFAULT_SIGN_TYPES from the collect script.
DEFAULT_MAPS = ["X", "T", "S", "O"]
DEFAULT_SIGN_TYPES = [
    "stop", "speed40", "zone_speed40", "minspeed40",
    "no_entry", "no_traffic", "no_stop",
    "detour_right", "detour_left",
    "bus_lane_road", "bus_lane",
    "bus_station", "only_auto",
]

# Speed bins — must match training
SPEED_BINS = np.array(
    [0.0, 0.025, 0.05472609, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 20.0],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Sign spawning helpers (identical to collect_rule_expert_plant2_trajectories.py)
# ---------------------------------------------------------------------------

def _get_route_lanes(env) -> List:
    nav = getattr(env.vehicle, "navigation", None)
    if nav is None:
        return []
    checkpoints = getattr(nav, "checkpoints", None)
    if not checkpoints or len(checkpoints) < 2:
        return []
    graph = env.current_map.road_network.graph
    lanes = []
    for s, e in zip(checkpoints[:-1], checkpoints[1:]):
        try:
            lanes.extend(graph[s][e])
        except KeyError:
            pass
    return lanes


def _lane_has_continuation(lane, road_network) -> bool:
    idx = getattr(lane, "index", None)
    if not (isinstance(idx, tuple) and len(idx) >= 2):
        return True
    outgoing = getattr(road_network, "graph", {}).get(idx[1])
    return not (outgoing is None or (isinstance(outgoing, dict) and len(outgoing) == 0))


def _pick_route_lane(route_lanes, min_length=10.0, road_network=None):
    cands = [l for l in route_lanes if l.length >= min_length]
    if road_network:
        cands = [l for l in cands if _lane_has_continuation(l, road_network)]
    return random.choice(cands) if cands else None


def _pick_rightmost_lane(route_lanes, min_length=30.0):
    segs: Dict = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if isinstance(idx, tuple) and len(idx) >= 3:
            segs.setdefault((idx[0], idx[1]), {})[idx[2]] = lane
    cands = []
    for d in segs.values():
        if max(d) + 1 < 2:
            continue
        l = d[max(d)]
        if l.length >= min_length:
            cands.append(l)
    return random.choice(cands) if cands else None


def _pick_detour_lane(route_lanes, sign_class, min_length=30.0):
    dirs = getattr(sign_class, "allowed_directions", set())
    segs: Dict = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if isinstance(idx, tuple) and len(idx) >= 3:
            segs.setdefault((idx[0], idx[1]), {})[idx[2]] = lane
    cands = []
    for d in segs.values():
        n = max(d) + 1
        for num, lane in d.items():
            if lane.length < min_length or num <= 0:
                continue
            if "right" in dirs and num >= n - 1:
                continue
            if "left" in dirs and num <= 0:
                continue
            cands.append(lane)
    return random.choice(cands) if cands else None


def _pick_lane_for_lane_change(route_lanes, vehicle_lane_num=0,
                                min_length=15.0, road_network=None):
    segs: Dict = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if isinstance(idx, tuple) and len(idx) >= 3:
            segs.setdefault((idx[0], idx[1]), {})[idx[2]] = lane
    cands = []
    for lanes_by_num in segs.values():
        if len(lanes_by_num) < 2 or vehicle_lane_num not in lanes_by_num:
            continue
        lane = lanes_by_num[vehicle_lane_num]
        if lane.length < min_length:
            continue
        if road_network and not _lane_has_continuation(lane, road_network):
            continue
        cands.append(lane)
    return random.choice(cands) if cands else None


_DETOUR_KEYS           = {"detour_right", "detour_left", "detour_either"}
_RESTRICTED_BEGIN_KEYS = {"bus_lane_road", "bike_lane_road", "bus_lane", "bike_lane"}
_LANE_CHANGE_KEYS      = {"no_entry", "no_traffic"}

_BEGIN_TO_END: Dict[str, Any] = {}  # filled lazily after sign imports


def _build_begin_to_end():
    # from traffic_signs.restricted_lane_sign import (  # type: ignore
    #     EndBusLaneRoadSign, EndBikeLaneRoadSign, EndBusLaneSign, EndBikeLaneSign,
    # )
    global _BEGIN_TO_END
    # _BEGIN_TO_END = {
    #     "bus_lane_road":  ("end_bus_lane_road",  EndBusLaneRoadSign),
    #     "bike_lane_road": ("end_bike_lane_road", EndBikeLaneRoadSign),
    #     "bus_lane":       ("end_bus_lane",       EndBusLaneSign),
    #     "bike_lane":      ("end_bike_lane",      EndBikeLaneSign),
    # }


def _build_sign_catalogue() -> Dict[str, Any]:
    from traffic_signs.stop_sign import StopSign  # type: ignore
    from traffic_signs.speed_limit_sign import (  # type: ignore
        SpeedLimitSign20, SpeedLimitSign30, SpeedLimitSign40, SpeedLimitSign60,
    )
    # from traffic_signs.zone_signs import (  # type: ignore
    #     ZoneSpeedLimitSign20, ZoneSpeedLimitSign30,
    #     ZoneSpeedLimitSign40, ZoneSpeedLimitSign60,
    # )
    from traffic_signs.min_speed_limit_sign import (  # type: ignore
        MinimumSpeedLimit30, MinimumSpeedLimit40,
        MinimumSpeedLimit50, MinimumSpeedLimit60,
    )
    from traffic_signs.no_entry_sign import NoEntrySign                          # type: ignore
    from traffic_signs.no_traffic_sign import NoTrafficSign                      # type: ignore
    from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign     # type: ignore
    # from traffic_signs.detour_sign import DetourRightSign, DetourLeftSign        # type: ignore
    # from traffic_signs.detour_obstacle import spawn_detour_obstacle              # type: ignore
    # from traffic_signs.restricted_lane_sign import (                             # type: ignore
    #     BusLaneRoadSign, BikeLaneRoadSign, BusLaneSign, BikeLaneSign,
    # )
    from traffic_signs.bus_station_sign import BusStationSign                    # type: ignore
    from traffic_signs.only_auto_sign import OnlyAutoSign                        # type: ignore

    return {
        "stop":          StopSign,
        "speed20":       SpeedLimitSign20,
        "speed30":       SpeedLimitSign30,
        "speed40":       SpeedLimitSign40,
        "speed60":       SpeedLimitSign60,
        # "zone_speed20":  ZoneSpeedLimitSign20,
        # "zone_speed30":  ZoneSpeedLimitSign30,
        # "zone_speed40":  ZoneSpeedLimitSign40,
        # "zone_speed60":  ZoneSpeedLimitSign60,
        "minspeed30":    MinimumSpeedLimit30,
        "minspeed40":    MinimumSpeedLimit40,
        "minspeed50":    MinimumSpeedLimit50,
        "minspeed60":    MinimumSpeedLimit60,
        "no_entry":      NoEntrySign,
        "no_traffic":    NoTrafficSign,
        "no_stop":       NoStoppingAllowedSign,
        # "detour_right":  DetourRightSign,
        # "detour_left":   DetourLeftSign,
        # "bus_lane_road": BusLaneRoadSign,
        # "bike_lane_road":BikeLaneRoadSign,
        # "bus_lane":      BusLaneSign,
        # "bike_lane":     BikeLaneSign,
        "bus_station":   BusStationSign,
        "only_auto":     OnlyAutoSign,
        # keep the detour obstacle spawner accessible
        # "__spawn_detour_obstacle": spawn_detour_obstacle,
    }


def spawn_sign_on_route(env, sign_key: str, sign_catalogue: Dict) -> Optional[Any]:
    sign_class = sign_catalogue.get(sign_key)
    if sign_class is None:
        print(f"[WARN] Unknown sign key: {sign_key}")
        return None

    sign_mgr     = env.engine.traffic_sign_manager
    road_network = env.current_map.road_network
    route_lanes  = _get_route_lanes(env)

    is_detour           = sign_key in _DETOUR_KEYS
    is_restricted_begin = sign_key in _RESTRICTED_BEGIN_KEYS
    is_lane_change      = sign_key in _LANE_CHANGE_KEYS

    try:
        lane = None
        if route_lanes:
            if is_detour:
                lane = _pick_detour_lane(route_lanes, sign_class, 30.0)
            elif is_lane_change:
                veh_idx = getattr(env.vehicle.lane, "index", None)
                veh_lane_num = veh_idx[2] if (veh_idx and len(veh_idx) >= 3) else 0
                lane = _pick_lane_for_lane_change(
                    route_lanes, veh_lane_num, 15.0, road_network)
            elif is_restricted_begin:
                lane = _pick_rightmost_lane(route_lanes, 30.0)
            else:
                lane = _pick_route_lane(route_lanes, 15.0, road_network)

        def _add(cls, lne):
            return (sign_mgr.add_sign(cls, lane=lne, use_random_lane=False)
                    if lne else sign_mgr.add_sign(cls, use_random_lane=True))

        if is_restricted_begin:
            sign = _add(sign_class, lane)
            if sign_key in _BEGIN_TO_END:
                _, end_cls = _BEGIN_TO_END[sign_key]
                sign_mgr.add_sign(end_cls, lane=sign.lane, use_random_lane=False)
        else:
            sign = _add(sign_class, lane)

        if is_detour and sign is not None:
            spawn_det = sign_catalogue.get("__spawn_detour_obstacle")
            if spawn_det:
                spawn_det(env.engine, sign.lane, sign)

        return sign

    except Exception as exc:
        print(f"[WARN] sign spawn failed ({sign_key}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _mock_carla_modules():
    import unittest.mock as _mock
    for mod_name in ("carla", "agents", "agents.navigation",
                     "agents.navigation.global_route_planner"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mock.MagicMock()


def load_plant_model(checkpoint_path: str, device: str = "cpu"):
    import yaml
    _mock_carla_modules()

    model_yaml = str(PLANT2_DIR / "PlanT" / "config" / "model" / "PlanT.yaml")
    if not os.path.isfile(model_yaml):
        raise FileNotFoundError(f"PlanT config not found: {model_yaml}")
    with open(model_yaml) as f:
        plnt = yaml.safe_load(f)

    class DictAsMember(dict):
        def __getattr__(self, name):
            v = self.get(name)
            return DictAsMember(v) if isinstance(v, dict) and not isinstance(v, DictAsMember) else v

    config_all  = DictAsMember({"model": plnt})
    plant_path  = str(PLANT2_DIR / "PlanT")
    if plant_path not in sys.path:
        sys.path.insert(0, plant_path)
    elif sys.path[0] != plant_path:
        sys.path.remove(plant_path)
        sys.path.insert(0, plant_path)

    from model import HFLM  # type: ignore

    net  = HFLM(config_all.model.network, config_all)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd   = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint contains no state_dict")
    if list(sd.keys())[0].startswith("model."):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.to(device).eval()
    return net, config_all


# ---------------------------------------------------------------------------
# Environment factory (no agent_policy — PlanT drives)
# ---------------------------------------------------------------------------

def create_env(map_code: str, seed: int, traffic_density: float, horizon: int):
    from envs.traffic_sign_env import TrafficSignEnv  # type: ignore
    return TrafficSignEnv(dict(
        map=map_code,
        start_seed=seed,
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=50,
        traffic_density=traffic_density,
        horizon=horizon,
        random_lane_width=True,
        random_lane_num=True,
        vehicle_config={"show_lidar": False},
        success_reward=20.0,
        out_of_road_penalty=15.0,
        crash_vehicle_penalty=25.0,
        crash_object_penalty=20.0,
        crash_sidewalk_penalty=5.0,
        driving_reward=0.5,
        speed_reward=0.05,
    ))


# ---------------------------------------------------------------------------
# Batch inspection helpers
# ---------------------------------------------------------------------------

def _signs_present(batch: Dict[str, Any]) -> Dict[int, float]:
    """Return {obj_type: min_dist_m} for every sign type visible in x_objs."""
    x_objs = batch["x_objs"]   # (pool_size, 7); col 0=type, 1=x, 2=y
    found: Dict[int, float] = {}
    for stype in ALL_SIGN_TYPES:
        mask = (x_objs[..., 0] == stype)
        if not mask.any():
            continue
        xy    = x_objs[..., 1:3][mask]
        dist  = float((xy ** 2).sum(dim=-1).sqrt().min().item())
        found[stype] = dist
    return found


def _speed_probs(pred_speed: torch.Tensor) -> np.ndarray:
    logits = pred_speed.detach().float()
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    return torch.softmax(logits, dim=0).cpu().numpy()


def _format_probs(probs: np.ndarray) -> str:
    parts = [f"{b:.3f}→{p:.3f}" for b, p in zip(SPEED_BINS, probs)]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Per-run accumulator
# ---------------------------------------------------------------------------

class SignStats:
    """Accumulates per-sign-type speed-prediction and violation statistics."""

    def __init__(self):
        # {obj_type -> list of (dist_m, ego_speed_mps, expected_speed_mps, top_bin)}
        self._data: Dict[int, List[Tuple]] = defaultdict(list)

        # Violation data (keyed by spawned sign_key string, e.g. "stop", "speed40")
        # total steps where sign._is_violating() is True
        self._viol_steps:   Dict[str, int]       = defaultdict(int)
        # unique violation events via check_all_violations (latched once per episode)
        self._unique_viols: Dict[str, int]        = defaultdict(int)
        # per-episode outcome: True = episode had at least one violation
        self._ep_results:   Dict[str, List[bool]] = defaultdict(list)

    def record(self, obj_type: int, dist: float, ego_speed: float,
               probs: np.ndarray) -> None:
        expected = float((probs * SPEED_BINS).sum())
        top_bin  = int(np.argmax(probs))
        self._data[obj_type].append((dist, ego_speed, expected, top_bin))

    # ------------------------------------------------------------------
    # Violation recording
    # ------------------------------------------------------------------

    def record_violation_step(self, sign_key: str) -> None:
        """Increment counter for every step where sign._is_violating() is True."""
        self._viol_steps[sign_key] += 1

    def record_unique_violation(self, sign_key: str) -> None:
        """Increment counter once per unique violation event (latch fires)."""
        self._unique_viols[sign_key] += 1

    def finalize_episode(self, sign_key: str, had_violation: bool) -> None:
        """Record whether this episode produced any rule violation."""
        self._ep_results[sign_key].append(had_violation)

    def violation_summary(self) -> Dict[str, Any]:
        """Per-sign-key violation metrics aggregated over all episodes."""
        all_keys = set(self._viol_steps) | set(self._unique_viols) | set(self._ep_results)
        out: Dict[str, Any] = {}
        for key in sorted(all_keys):
            ep_list   = self._ep_results.get(key, [])
            n_eps     = len(ep_list)
            n_viol_ep = sum(ep_list)
            n_success = n_eps - n_viol_ep
            viol_rate    = round(n_viol_ep / n_eps, 4) if n_eps else 0.0
            success_rate = round(n_success  / n_eps, 4) if n_eps else 0.0
            out[key] = {
                "n_episodes":             n_eps,
                "violation_episodes":     n_viol_ep,
                "success_episodes":       n_success,
                "violation_rate":         viol_rate,   # eps_with_violation / all_eps
                "success_rate":           success_rate, # eps_without_violation / all_eps
                "total_viol_steps":       self._viol_steps.get(key, 0),
                "unique_viol_events":     self._unique_viols.get(key, 0),
                "mean_viol_steps_per_ep": (
                    round(self._viol_steps.get(key, 0) / n_eps, 2) if n_eps else 0.0
                ),
            }
        return out

    def summary(self) -> Dict[str, Any]:
        out = {}
        for stype, rows in self._data.items():
            name   = SIGN_TYPE_META.get(stype, {}).get("name", str(stype))
            arr    = np.array(rows)  # (N, 4)
            out[name] = {
                "n_steps":            len(rows),
                "mean_dist_m":        float(arr[:, 0].mean()),
                "mean_ego_speed_mps": float(arr[:, 1].mean()),
                "mean_exp_speed_mps": float(arr[:, 2].mean()),
                "top_bin_mode":       int(np.bincount(arr[:, 3].astype(int)).argmax()),
            }
        return out

    def plot_speed_vs_dist(self, out_dir: str) -> None:
        for stype, rows in self._data.items():
            if not rows:
                continue
            name  = SIGN_TYPE_META.get(stype, {}).get("name", str(stype))
            color = SIGN_TYPE_META.get(stype, {}).get("color", "blue")
            arr   = np.array(rows)
            dists = arr[:, 0]
            e_spd = arr[:, 2]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(dists, e_spd, alpha=0.5, color=color, s=15, label="E[speed]")
            ax.scatter(dists, arr[:, 1], alpha=0.3, color="gray", s=10, label="ego_speed")
            # binned mean line
            if len(dists) > 5:
                bins   = np.linspace(dists.min(), dists.max(), min(15, len(dists) // 3 + 2))
                counts, edges = np.histogram(dists, bins=bins)
                sums,   _     = np.histogram(dists, bins=bins, weights=e_spd)
                valid = counts > 0
                cx    = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(cx[valid], (sums / counts)[valid], color=color,
                        linewidth=2, label="mean E[speed]")
            ax.set_xlabel("Distance to sign (m)")
            ax.set_ylabel("Speed (m/s)")
            ax.set_title(f"Model speed prediction vs. distance — {name}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = os.path.join(out_dir, f"speed_vs_dist_{name}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            print(f"  [plot] {path}")

    def plot_top_bin_hist(self, out_dir: str) -> None:
        for stype, rows in self._data.items():
            if not rows:
                continue
            name  = SIGN_TYPE_META.get(stype, {}).get("name", str(stype))
            color = SIGN_TYPE_META.get(stype, {}).get("color", "blue")
            arr   = np.array(rows)
            bins_idx = arr[:, 3].astype(int)
            counts   = np.bincount(bins_idx, minlength=len(SPEED_BINS))

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(range(len(SPEED_BINS)), counts, color=color, alpha=0.8)
            ax.set_xticks(range(len(SPEED_BINS)))
            ax.set_xticklabels([f"{b:.3f}" for b in SPEED_BINS], rotation=45, ha="right")
            ax.set_xlabel("Speed bin (m/s)")
            ax.set_ylabel("Frequency (steps)")
            ax.set_title(f"Top predicted speed bin — {name}")
            fig.tight_layout()
            path = os.path.join(out_dir, f"top_bin_hist_{name}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            print(f"  [plot] {path}")

    def plot_violation_rate(self, out_dir: str) -> None:
        """Grouped bar chart: violation rate and success rate per sign key."""
        vsumm = self.violation_summary()
        if not vsumm:
            return
        keys    = sorted(vsumm.keys())
        vrates  = [vsumm[k]["violation_rate"] for k in keys]
        srates  = [vsumm[k]["success_rate"]   for k in keys]
        n_eps   = [vsumm[k]["n_episodes"]      for k in keys]

        x     = np.arange(len(keys))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(9, len(keys) * 1.1), 4))
        v_bars = ax.bar(x - width / 2, vrates, width, color="tomato",    alpha=0.85, label="Violation rate")
        s_bars = ax.bar(x + width / 2, srates, width, color="steelblue", alpha=0.85, label="Success rate")
        for bar, n in zip(v_bars, n_eps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={n}",
                ha="center", va="bottom", fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Rate (fraction of episodes)")
        ax.set_title("Rule violation vs. success rate per sign type")
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = os.path.join(out_dir, "violation_rate.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print speed-classifier probs for all sign types "
            "(stop/speed_limit/min_speed/no_entry/no_stopping/detour/"
            "restricted_lane/only_auto) visible in x_objs."
        )
    )
    parser.add_argument("--checkpoint_file",   type=str, required=True)
    parser.add_argument("--num-episodes",      type=int,   default=13)
    parser.add_argument("--start-seed",        type=int,   default=1000)
    parser.add_argument("--max-steps",         type=int,   default=500)
    parser.add_argument("--traffic-density",   type=float, default=0.0)
    parser.add_argument(
        "--maps",       type=str, default=",".join(DEFAULT_MAPS),
        help="Comma-separated map codes to cycle through (default: X,T,S,O)",
    )
    parser.add_argument(
        "--sign-types", type=str, default=",".join(DEFAULT_SIGN_TYPES),
        help="Comma-separated sign keys to cycle through per episode",
    )
    parser.add_argument("--device",     type=str,   default=None)
    parser.add_argument("--min-dist",   type=float, default=float("inf"),
                        help="Only print when nearest sign is within this many metres")
    parser.add_argument("--output_dir", type=str,   default=None)
    parser.add_argument("--no-gifs",    action="store_true")
    args = parser.parse_args()

    maps       = [m.strip() for m in args.maps.split(",")       if m.strip()]
    sign_types = [s.strip() for s in args.sign_types.split(",") if s.strip()]
    device     = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir    = args.output_dir or str(
        PDD_BENCH_DIR / "outputs" / "eval_rule_sign_speed_probs"
    )
    os.makedirs(out_dir, exist_ok=True)
    if not args.no_gifs:
        os.makedirs(os.path.join(out_dir, "gifs"), exist_ok=True)

    print("=" * 72)
    print("PlanT rule-sign speed-prob diagnostic")
    print(f"  Checkpoint  : {args.checkpoint_file}")
    print(f"  Episodes    : {args.num_episodes}  start_seed={args.start_seed}")
    print(f"  Device      : {device}")
    print(f"  Maps        : {maps}")
    print(f"  Sign types  : {sign_types}")
    print(f"  Dist filter : {args.min_dist} m")
    print(f"  Output dir  : {out_dir}")
    print("=" * 72)

    net, config_all = load_plant_model(args.checkpoint_file, device=device)
    training_cfg    = getattr(config_all.model, "training", {}) or {}
    input_bev       = True
    input_ego_speed = bool(training_cfg.get("input_ego_speed", False))

    from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch  # type: ignore
    from carla_garage.plant2_control import plant2_predictions_to_action, get_target_speed_from_limit  # type: ignore

    # Build sign catalogue + _BEGIN_TO_END (deferred until paths are ready)
    sign_catalogue = _build_sign_catalogue()
    # _build_begin_to_end()

    random.seed(args.start_seed)
    stats = SignStats()
    total_sign_steps = defaultdict(int)

    for ep in range(args.num_episodes):
        ep_seed  = args.start_seed + ep
        map_code = maps[ep % len(maps)]
        sign_key = sign_types[ep % len(sign_types)]

        random.seed(ep_seed)
        np.random.seed(ep_seed)

        print(f"\n{'─'*72}")
        print(f"Episode {ep+1}/{args.num_episodes}  seed={ep_seed}  map={map_code}  sign={sign_key}")
        print(f"{'─'*72}")

        env = create_env(map_code, ep_seed, args.traffic_density, args.max_steps)
        try:
            obs, _info = env.reset()
            vehicle  = getattr(env, "vehicle", None) or getattr(env, "agent", None)
            engine   = env.engine
            sign_mgr = getattr(engine, "traffic_sign_manager", None)

            sign = spawn_sign_on_route(env, sign_key, sign_catalogue)
            if sign is None:
                print(f"  [WARN] Could not spawn sign — skipping episode")
                stats.finalize_episode(sign_key, had_violation=False)
                continue

            # Clear renderer for GIF
            if not args.no_gifs:
                renderer = getattr(env, "top_down_renderer", None)
                if renderer and hasattr(renderer, "_screen_frames"):
                    renderer._screen_frames.clear()

            ep_sign_steps: Dict[int, int] = defaultdict(int)
            ep_had_violation = False
            ep_viol_steps    = 0
            ep_unique_viols  = 0

            for step_i in range(args.max_steps):
                if vehicle is None:
                    break

                # ---- Build plant2 batch ----
                batch = metadrive_obs_to_plant2_batch(
                    engine, vehicle,
                    route_ego_20x2=None,
                    speed_limit_kmh=None,
                    max_objects=30,
                    max_distance=75.0,
                    range_factor_front=16.0,
                    input_bev=input_bev,
                    input_ego_speed=input_ego_speed,
                    bev_resolution=128,
                    bev_size_meters=64.0,
                    device=device,
                )

                # ---- Model inference ----
                with torch.no_grad():
                    _, _, pred_plan, _ = net(batch)
                pred_path, pred_wps, pred_speed = pred_plan

                # ---- Detect signs in x_objs ----
                signs_present = _signs_present(batch)
                ego_speed     = float(getattr(vehicle, "speed", 0.0))
                print(f"  signs_present: {signs_present}")
                for stype, dist in signs_present.items():
                    if dist > args.min_dist:
                        continue
                    ep_sign_steps[stype] += 1
                    total_sign_steps[stype] += 1

                    sname = SIGN_TYPE_META.get(stype, {}).get("name", str(stype))

                    if pred_speed is not None:
                        probs          = _speed_probs(pred_speed)
                        expected_speed = float((probs * SPEED_BINS).sum())
                        top_bin        = int(np.argmax(probs))
                        stats.record(stype, dist, ego_speed, probs)
                        print(
                            f"  step={step_i:4d} | {sname:<16}"
                            f" dist={dist:5.1f} m"
                            f" | ego={ego_speed:.2f} m/s"
                            f" | top_bin={top_bin}({SPEED_BINS[top_bin]:.3f})"
                            f" | E[spd]={expected_speed:.3f} m/s"
                        )
                        print(f"             probs: {_format_probs(probs)}")
                    else:
                        sname_f = f"{sname:<16}"
                        print(
                            f"  step={step_i:4d} | {sname_f}"
                            f" dist={dist:5.1f} m | pred_speed is None"
                        )

                # ---- Render for GIF ----
                if not args.no_gifs:
                    ego_pos_xy  = np.asarray(vehicle.position, dtype=np.float32)[:2]
                    ego_heading = float(vehicle.heading_theta)
                    pred_coords = pred_wps if pred_wps is not None else pred_path
                    render_kw: Dict[str, Any] = dict(
                        mode="top_down",
                        screen_record=True,
                        window=False,
                        screen_size=(640, 640),
                        overlay_ego_pos_world=ego_pos_xy,
                        overlay_ego_heading_rad=ego_heading,
                    )
                    if pred_coords is not None:
                        pred_ego = pred_coords.squeeze(0).detach().cpu().numpy().astype(np.float32)
                        c, s = np.cos(ego_heading), np.sin(ego_heading)
                        dx   = pred_ego[:, 0] * c - (-pred_ego[:, 1]) * s
                        dy   = pred_ego[:, 0] * s + (-pred_ego[:, 1]) * c
                        render_kw["overlay_pred_traj_world"] = (
                            np.stack([dx, dy], axis=-1) + ego_pos_xy
                        )
                    try:
                        env.render(**render_kw)
                    except Exception:
                        pass

                # ---- Step environment with PlanT action ----
                speed_limit_idx = int(batch["speed_limit"][0].item())
                target_speed    = get_target_speed_from_limit(speed_limit_idx)
                action_np, _    = plant2_predictions_to_action(
                    pred_plan,
                    current_speed=ego_speed,
                    target_speed_mps=target_speed,
                    speed_limit_idx=speed_limit_idx,
                    device=device,
                    return_waypoints=True,
                )
                action_np = np.clip(np.asarray(action_np, dtype=np.float32), -1.0, 1.0)
                _, _, terminated, truncated, _ = env.step(action_np)

                # ---- Violation checks (after env.step so vehicle has moved) ----
                if sign_mgr is not None and vehicle is not None:
                    # Per-step: _is_violating fires every step the breach is active
                    step_violating = any(
                        s._is_violating(vehicle) for s in sign_mgr.signs
                    )
                    if step_violating:
                        ep_viol_steps += 1
                        ep_had_violation = True
                        stats.record_violation_step(sign_key)

                    # Unique events: check_all_violations latches (fires once per event)
                    for s, violated in sign_mgr.check_all_violations(vehicle):
                        if violated:
                            ep_unique_viols += 1
                            ep_had_violation = True
                            stats.record_unique_violation(sign_key)
                            print(
                                f"  [VIOL] step={step_i:4d} {type(s).__name__}"
                                f" | ego={float(getattr(vehicle, 'speed', 0.0)):.2f} m/s"
                            )

                if terminated or truncated:
                    break

            # Episode summary
            for stype, cnt in ep_sign_steps.items():
                sname = SIGN_TYPE_META.get(stype, {}).get("name", str(stype))
                print(f"  -> {cnt} steps with {sname} visible")
            print(
                f"  -> violations: {ep_unique_viols} unique event(s),"
                f" {ep_viol_steps} violating step(s)"
                + (" [VIOLATED]" if ep_had_violation else " [COMPLIANT]")
            )
            stats.finalize_episode(sign_key, ep_had_violation)

            # Save GIF
            if not args.no_gifs:
                gif_path = os.path.join(
                    out_dir, "gifs",
                    f"ep{ep:03d}_seed{ep_seed}_{map_code}_{sign_key}.gif",
                )
                try:
                    renderer = getattr(env, "top_down_renderer", None)
                    if renderer is not None:
                        renderer.generate_gif(gif_path, duration=10)
                        print(f"  [GIF] {gif_path}")
                except Exception as exc:
                    print(f"  [WARN] GIF failed: {exc}")

        finally:
            env.close()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Steps with each sign type visible:")
    for stype in ALL_SIGN_TYPES:
        n    = total_sign_steps.get(stype, 0)
        name = SIGN_TYPE_META.get(stype, {}).get("name", str(stype))
        print(f"  {name:<20} {n:5d} steps")

    summary      = stats.summary()
    viol_summary = stats.violation_summary()

    print(f"\nPer-sign-type speed prediction summary:")
    hdr = f"  {'sign':<20} {'n':>6} {'dist_m':>7} {'ego_spd':>8} {'exp_spd':>8} {'top_bin':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, m in sorted(summary.items()):
        print(
            f"  {name:<20} {m['n_steps']:>6} {m['mean_dist_m']:>7.1f}"
            f" {m['mean_ego_speed_mps']:>8.3f} {m['mean_exp_speed_mps']:>8.3f}"
            f" {SPEED_BINS[m['top_bin_mode']]:>7.3f}"
        )

    print(f"\nRule violation summary (per sign key):")
    vhdr = (
        f"  {'sign_key':<20} {'eps':>5} {'viol_eps':>9} {'viol_rate':>10}"
        f" {'success_rate':>13} {'viol_steps':>11} {'uniq_events':>12} {'mean_vsteps':>12}"
    )
    print(vhdr)
    print("  " + "-" * (len(vhdr) - 2))
    for key, v in sorted(viol_summary.items()):
        print(
            f"  {key:<20} {v['n_episodes']:>5} {v['violation_episodes']:>9}"
            f" {v['violation_rate']:>10.3f} {v['success_rate']:>13.3f}"
            f" {v['total_viol_steps']:>11} {v['unique_viol_events']:>12}"
            f" {v['mean_viol_steps_per_ep']:>12.1f}"
        )

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump({"speed_prediction": summary, "violations": viol_summary}, f, indent=2)
    print(f"\nSummary JSON -> {json_path}")

    stats.plot_speed_vs_dist(out_dir)
    stats.plot_top_bin_hist(out_dir)
    stats.plot_violation_rate(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
