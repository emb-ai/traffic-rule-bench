"""Sign-compliance mixin: init, reset, and per-step dispatch."""
import logging

from traffic_bench.agents.compliance.crosswalk import CrosswalkCompliance
from traffic_bench.agents.compliance.detour import DetourCompliance
from traffic_bench.agents.compliance.dual_path import DualPathCompliance
from traffic_bench.agents.compliance.extra import ExtraCompliance
from traffic_bench.agents.compliance.junction import JunctionCompliance
from traffic_bench.agents.compliance.kinematics import Kinematics
from traffic_bench.agents.compliance.speed import SpeedCompliance
from traffic_bench.signs.blocked.no_traffic import NoTrafficSign
from traffic_bench.signs.detour.plate import DetourSign
from traffic_bench.signs.dual_path.direction import LaneAllowedDirectionSign
from traffic_bench.signs.dual_path.no_entry import NoEntrySign
from traffic_bench.signs.dual_path.no_turn import NoLeftTurnSign, NoRightTurnSign, NoUTurnSign
from traffic_bench.signs.dual_path.one_way import OneWayEntrySign
from traffic_bench.signs.dual_path.pg_direction import PGDirectionSign
from traffic_bench.signs.extra.bus_station import BusStationSign
from traffic_bench.signs.extra.direction_legacy import DirectionSign
from traffic_bench.signs.extra.lane_directions import LaneDirectionsSign
from traffic_bench.signs.extra.no_overtaking import NoOvertakingSign
from traffic_bench.signs.extra.no_stopping import NoStoppingAllowedSign
from traffic_bench.signs.extra.only_auto import OnlyAutoSign
from traffic_bench.signs.extra.restricted_lane import (
    EndOfRestrictedLaneSign,
    IntersectionRestrictedLaneSign,
    RestrictedLaneSign,
)
from traffic_bench.signs.extra.right_turn import RightTurnRule
from traffic_bench.signs.extra.traffic_light import TrafficLightSign
from traffic_bench.signs.junction import (
    EndMainRoadSign,
    MainRoadSign,
    RightHandYieldSign,
    RoundaboutSign,
    SecondaryRoadLeftSign,
    SecondaryRoadRightSign,
    SecondaryRoadSign,
    StopSign,
    YieldSign,
)
from traffic_bench.signs.speed.end_of_zone import BaseEndOfZoneSign
from traffic_bench.signs.speed.limit import SpeedLimitSign
from traffic_bench.signs.speed.min_speed import MinimumSpeedLimitSign
from traffic_bench.signs.speed.zone import ZoneSpeedLimitSign

logger = logging.getLogger(__name__)


class SignComplianceMixin(
    JunctionCompliance,
    DualPathCompliance,
    SpeedCompliance,
    DetourCompliance,
    CrosswalkCompliance,
    ExtraCompliance,
    Kinematics,
):
    """Traffic sign compliance logic shared between PPO and IDM experts."""

    STOP_WAIT_STEPS = 15
    NO_STOP_MIN_SPEED_KMH = 5.0
    BRAKE_ACTION = -1.0
    APPLY_UTURN_ZONE_ASSIST = False
    APPLY_LANE_DIRS_NAV_HOLD = True
    NO_ENTRY_STOP_MARGIN = 3.0
    PREEMPT_DETOUR_M = 10.0
    DETOUR_RETURN_CLEARANCE_M = 8.0
    DETOUR_QUEUE_LOOKAHEAD_M = 35.0
    PREEMPT_RESTRICTED_LANE_M = 50.0

    def _get_heading_pid(self):
        raise NotImplementedError

    def _get_lateral_pid(self):
        raise NotImplementedError

    def _init_sign_compliance(self):
        self._lc_target_lane = None
        self._lc_final_sumo_num = None
        self._stop_states = {}
        self._speed_cap = None
        self._speed_floor = None
        self._blocked_lanes = set()
        self._restricted_lanes = set()
        self._rerouted_edges = {}   # {(from, to): True/False} — True = rerouted OK
        self._has_priority = False  # True when on a main road (set by MainRoadSign)
        self._no_overtaking_active = False  # set per-step by NoOvertakingSign
        # Set per-step when 5.7.x / 4.1.x nav already avoids the forbidden hop.
        self._one_way_nav_clean = False
        # Mid-route U-turn assist (3.18.1/3.18.2 only) — steering only.
        self._no_turn_318_context = False
        self._uturn_via_lane = None
        self._uturn_source_lane = None
        self._uturn_hold_lane = None
        self._uturn_hold_steps = 0
        self._uturn_bias = 0.0
        self._uturn_phase = None  # "approach" | "center" | "spin"
        self._uturn_spinning = False
        self._uturn_spin_dir = None
        self._uturn_spin_dir_flipped = False
        self._uturn_saved_max_steering = None
        # 5.15.1: after the one post-LC compliant replan, never rewrite nav
        # again (NN policies often oscillate across peer lanes mid-merge).
        self._lane_dirs_nav_locked = False
        self._lane_dirs_hold_applied = False
        # Sticky per step while a LaneDirectionsSign is in the scene.
        # Blocks U-turn body-hold (`set_position`) — 5.15.1 is steering-only.
        self._lane_dirs_active = False

    def _reset_sign_compliance(self):
        """Call on episode reset to clear stale state."""
        self._stop_states.clear()
        self._rerouted_edges.clear()
        self._lc_target_lane = None
        self._lc_final_sumo_num = None
        self._has_priority = False
        self._no_overtaking_active = False
        self._lane_dirs_nav_locked = False
        self._lane_dirs_hold_applied = False
        self._lane_dirs_active = False
        self._one_way_nav_clean = False
        self._no_turn_318_context = False
        self._restore_uturn_steering_limit()
        self._uturn_via_lane = None
        self._uturn_source_lane = None
        self._uturn_hold_lane = None
        self._uturn_hold_steps = 0
        self._uturn_bias = 0.0
        self._uturn_phase = None
        self._uturn_spinning = False
        self._uturn_spin_dir = None
        self._uturn_spin_dir_flipped = False
        self._uturn_saved_max_steering = None

    def _get_signs(self):
        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "traffic_sign_manager"):
            return []
        return engine.traffic_sign_manager.signs

    def _process_signs(self):
        self._speed_cap = None
        self._speed_floor = None
        self._blocked_lanes.clear()
        self._restricted_lanes.clear()
        self._no_overtaking_active = False
        # Sticky within a step: any 3.18.1/3.18.2 sign in this scene.
        # Cleared each `_process_signs` call; re-set when those signs are seen.
        self._no_turn_318_context = False
        self._one_way_nav_clean = False
        self._lane_dirs_active = False

        for sign in self._get_signs():
            try:
                if isinstance(sign, StopSign):
                    self._handle_yield_sign(sign)
                    self._handle_stop_sign(sign)
                elif isinstance(sign, MinimumSpeedLimitSign):
                    self._handle_min_speed(sign)
                elif isinstance(sign, (SpeedLimitSign, ZoneSpeedLimitSign)):
                    self._handle_speed_limit(sign)
                elif isinstance(sign, (NoEntrySign, NoTrafficSign)):
                    self._handle_no_entry_or_no_traffic(sign)
                elif isinstance(sign, TrafficLightSign):
                    self._handle_traffic_light(sign)
                elif isinstance(sign, DetourSign):
                    self._handle_detour(sign)
                elif isinstance(sign, IntersectionRestrictedLaneSign):
                    self._handle_intersection_restricted_lane(sign)
                elif isinstance(sign, EndOfRestrictedLaneSign):
                    pass
                elif isinstance(sign, RestrictedLaneSign):
                    self._handle_restricted_lane(sign)
                elif isinstance(sign, NoStoppingAllowedSign):
                    self._handle_no_stopping(sign)
                elif isinstance(sign, BusStationSign):
                    self._handle_bus_station(sign)
                elif isinstance(sign, OnlyAutoSign):
                    self._handle_only_auto(sign)
                elif isinstance(sign, RightTurnRule):
                    self._handle_right_turn_rule(sign)
                elif isinstance(sign, RightHandYieldSign):
                    self._handle_yield_sign(sign)
                elif isinstance(sign, YieldSign):
                    self._handle_yield_sign(sign)
                elif isinstance(sign, EndMainRoadSign):
                    self._handle_end_main_road(sign)
                elif isinstance(sign, (NoRightTurnSign, NoLeftTurnSign)):
                    # 3.18.1 / 3.18.2 only — enables mid-route U-turn assist.
                    self._no_turn_318_context = True
                    self._handle_no_turn(sign)
                elif isinstance(sign, NoUTurnSign):
                    # 3.19: same turn blocking, but no mid-route U-turn assist.
                    self._handle_no_turn(sign)
                elif isinstance(sign, OneWayEntrySign):
                    self._handle_no_turn(sign)
                elif isinstance(sign, NoOvertakingSign):
                    self._handle_no_overtaking(sign)
                elif isinstance(sign, LaneDirectionsSign):
                    # Must run before DirectionSign: 5.15.1 subclasses it.
                    # Steering LC only — never the old 4.1 body snap onto a via.
                    self._lane_dirs_active = True
                    self._handle_direction_compliance(sign)
                elif isinstance(sign, (DirectionSign, PGDirectionSign, LaneAllowedDirectionSign)):
                    self._handle_direction_compliance(sign)
                elif isinstance(sign, MainRoadSign):
                    self._handle_main_road(sign)
                elif isinstance(sign, (SecondaryRoadSign,
                                       SecondaryRoadLeftSign,
                                       SecondaryRoadRightSign)):
                    # 2.3.x on the ego approach = has priority (same as 2.1).
                    # Ego on a secondary arm still yields via YieldSign (2.4).
                    self._handle_main_road(sign)
                elif isinstance(sign, RoundaboutSign):
                    pass  # informational 4.3 plate — yield via RoundaboutYieldSign
                elif isinstance(sign, BaseEndOfZoneSign):
                    pass  # informational — no action needed
            except Exception as exc:
                logger.debug("Error processing %s: %s", type(sign).__name__, exc)

        self._process_rules()
        # Pre-installed one-way compliant nav: skip hard priority yield so
        # continuous cross-traffic cannot freeze NN policies at the approach.
        if not getattr(self, "_one_way_nav_clean", False):
            self._handle_intersection_priority()       # PG (RoadPriorityAssigner)
            self._handle_intersection_priority_sumo()  # SUMO (lane.turns[i]["priority"])
