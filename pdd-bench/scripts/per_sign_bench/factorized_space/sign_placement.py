"""
Deterministic sign placement rules by sign_type category (§13 of the plan).

Returns a longitudinal offset (relative to the start of the spawn lane) and
lateral offset for a given (sign_type, lane). The runner may refuse to
materialize a scene if preconditions are violated (short lanes, missing
intersection ahead, etc.) rather than silently "adjusting" — this keeps the
benchmark dataset clean of placement hacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .space_definition import (
    UNIVERSAL_SIGNS,
    END_OF_ZONE_SIGNS,
    MULTI_LANE_SIGNS,
    RESTRICTED_LANE_SIGNS,
    END_RESTRICTED_LANE_SIGNS,
    EXIT_RESTRICTED_LANE_SIGNS,
    DETOUR_SIGNS,
    TURN_PROHIBITION_SIGNS,
    INTERSECTION_PRIORITY_SIGNS,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Distance from the sign to the start of the "active zone" (stop-line, obstacle,
# lane boundary). Defaults match the 10–15 m range of real GOST placement.
SIGN_SPAWN_DISTANCE_DEFAULT = 15.0      # m, approach distance from sign to decision point
LANE_START_OFFSET_DEFAULT = 10.0        # m, distance from lane beginning
LANE_END_BUFFER_DEFAULT = 5.0           # m, buffer from lane end
OBSTACLE_PROXIMITY_DEFAULT = 20.0       # m, distance before obstacle for detour signs
INTERSECTION_ENTRY_OFFSET = 10.0        # m from stop line back toward approach


@dataclass(frozen=True)
class PlacementRule:
    strategy: str                        # semantic tag for logging
    # Callable-like spec: these are evaluated by compute_placement()
    # against the selected lane. The strategy string drives which formula.
    min_lane_length: float = 25.0
    requires_intersection_ahead: bool = False
    requires_intersection_behind: bool = False  # for end_main_road
    lateral_side: str = "right"          # "right" | "left" | "center"


# Map sign_type → rule. Signs with no explicit entry fall through to a
# conservative default (straight_segment_middle).
SIGN_PLACEMENT_RULES: Dict[str, PlacementRule] = {}


def _register(sign_type: str, rule: PlacementRule) -> None:
    SIGN_PLACEMENT_RULES[sign_type] = rule


# Restricted-lane begin → placed at the beginning of the lane before an
# intersection so ego sees the sign as soon as it enters the road.
for s in RESTRICTED_LANE_SIGNS:
    _register(s, PlacementRule(
        strategy="lane_start_before_intersection",
        min_lane_length=30.0,
        requires_intersection_ahead=True,
        lateral_side="right",
    ))

# End of restricted lane → placed at the end of the lane, past the intersection.
for s in END_RESTRICTED_LANE_SIGNS:
    _register(s, PlacementRule(
        strategy="lane_end_after_intersection",
        min_lane_length=30.0,
        requires_intersection_behind=True,
        lateral_side="right",
    ))

# Exit to restricted lane → placed at the intersection approach (right lane).
for s in EXIT_RESTRICTED_LANE_SIGNS:
    _register(s, PlacementRule(
        strategy="intersection_entry_right_lane",
        min_lane_length=30.0,
        requires_intersection_ahead=True,
        lateral_side="right",
    ))

# Turn-prohibition → at intersection approach, centered on ego's lane.
for s in TURN_PROHIBITION_SIGNS:
    _register(s, PlacementRule(
        strategy="intersection_entry_center",
        min_lane_length=25.0,
        requires_intersection_ahead=True,
        lateral_side="center",
    ))

# Intersection priority signs (main_road, yield, secondary_road*, stop, pg_direction)
for s in INTERSECTION_PRIORITY_SIGNS:
    if s == "end_main_road":
        # End of main road → placed AFTER the intersection.
        _register(s, PlacementRule(
            strategy="just_after_intersection",
            min_lane_length=25.0,
            requires_intersection_behind=True,
            lateral_side="right",
        ))
    else:
        _register(s, PlacementRule(
            strategy="intersection_entry",
            min_lane_length=25.0,
            requires_intersection_ahead=True,
            lateral_side="right",
        ))

# stop, no_overtaking, bus_station — non-intersection, mid-segment
_register("stop", PlacementRule(
    strategy="intersection_entry",
    min_lane_length=25.0,
    requires_intersection_ahead=True,
    lateral_side="right",
))
_register("bus_station", PlacementRule(
    strategy="straight_segment_designated_stop",
    min_lane_length=40.0,
    lateral_side="right",
))

# Detour signs → before the obstacle that the runner will spawn.
for s in DETOUR_SIGNS:
    _register(s, PlacementRule(
        strategy="obstacle_proximity",
        min_lane_length=40.0,
        lateral_side="right",
    ))

# Speed/zone/minspeed and their END variants: straight segment, middle vs end.
_speed_start = [
    "speed20", "speed30", "speed40", "speed60",
    "zone_speed20", "zone_speed30", "zone_speed40", "zone_speed60",
    "minspeed30", "minspeed40", "minspeed50",
]
_speed_end = [
    "end_speed20", "end_speed30", "end_speed40", "end_speed60",
    "end_zone_speed20", "end_zone_speed30", "end_zone_speed40", "end_zone_speed60",
    "end_all",
]
for s in _speed_start:
    _register(s, PlacementRule(
        strategy="straight_segment_middle",
        min_lane_length=30.0,
        lateral_side="right",
    ))
for s in _speed_end:
    _register(s, PlacementRule(
        strategy="straight_segment_end",
        min_lane_length=30.0,
        lateral_side="right",
    ))

# no_stop → straight segment middle
_register("no_stop", PlacementRule(
    strategy="straight_segment_middle",
    min_lane_length=30.0,
    lateral_side="right",
))
_register("no_overtaking", PlacementRule(
    strategy="straight_segment_middle",
    min_lane_length=40.0,
    lateral_side="right",
))
_register("only_auto", PlacementRule(
    strategy="straight_segment_middle",
    min_lane_length=25.0,
    lateral_side="right",
))


# ---------------------------------------------------------------------------
# Placement resolver
# ---------------------------------------------------------------------------

# PlacementContext carries just enough about the scene geometry for the
# resolver to compute offsets. The runner fills these in.

@dataclass
class PlacementContext:
    lane_length: float                    # m
    lane_width: float                     # m
    has_intersection_ahead: bool = False  # True if the spawn lane ends at an intersection
    has_intersection_behind: bool = False
    sign_spawn_distance: float = SIGN_SPAWN_DISTANCE_DEFAULT
    obstacle_distance: Optional[float] = None  # for detour scenes, distance from lane start to obstacle


@dataclass
class Placement:
    longitudinal_offset: float   # longitudinal coordinate along lane (0 = lane start)
    lateral_offset: float        # positive = right side
    strategy: str
    failure_reason: Optional[str] = None  # set when preconditions not met


def compute_placement(sign_type: str, ctx: PlacementContext) -> Placement:
    """Compute (longitudinal_offset, lateral_offset) for a sign on a given lane.

    Returns Placement with failure_reason set when preconditions are not
    satisfied — callers should drop the scene rather than try to "fix" the
    geometry.
    """
    rule = SIGN_PLACEMENT_RULES.get(sign_type)
    if rule is None:
        # Conservative default: treat as speed-limit-like.
        rule = PlacementRule(
            strategy="straight_segment_middle",
            min_lane_length=25.0,
            lateral_side="right",
        )

    if ctx.lane_length < rule.min_lane_length:
        return Placement(
            longitudinal_offset=0.0, lateral_offset=0.0,
            strategy=rule.strategy,
            failure_reason=f"lane_too_short: {ctx.lane_length:.1f} < {rule.min_lane_length}",
        )
    if rule.requires_intersection_ahead and not ctx.has_intersection_ahead:
        return Placement(
            longitudinal_offset=0.0, lateral_offset=0.0,
            strategy=rule.strategy,
            failure_reason="no_intersection_ahead",
        )
    if rule.requires_intersection_behind and not ctx.has_intersection_behind:
        return Placement(
            longitudinal_offset=0.0, lateral_offset=0.0,
            strategy=rule.strategy,
            failure_reason="no_intersection_behind",
        )

    # Compute longitudinal offset by strategy.
    L = ctx.lane_length
    strategy = rule.strategy

    if strategy == "lane_start_before_intersection":
        longitudinal = LANE_START_OFFSET_DEFAULT
    elif strategy == "lane_end_after_intersection":
        longitudinal = max(L - LANE_END_BUFFER_DEFAULT, L * 0.85)
    elif strategy == "intersection_entry_right_lane":
        longitudinal = max(L - INTERSECTION_ENTRY_OFFSET - ctx.sign_spawn_distance, L * 0.6)
    elif strategy == "intersection_entry_center":
        longitudinal = max(L - INTERSECTION_ENTRY_OFFSET, L * 0.7)
    elif strategy == "intersection_entry":
        longitudinal = max(L - INTERSECTION_ENTRY_OFFSET, L * 0.7)
    elif strategy == "just_after_intersection":
        longitudinal = min(LANE_START_OFFSET_DEFAULT, L * 0.15)
    elif strategy == "obstacle_proximity":
        if ctx.obstacle_distance is None:
            # without a known obstacle, fall back to middle
            longitudinal = L * 0.5
        else:
            longitudinal = max(ctx.obstacle_distance - OBSTACLE_PROXIMITY_DEFAULT, LANE_START_OFFSET_DEFAULT)
    elif strategy == "straight_segment_middle":
        longitudinal = L * 0.5
    elif strategy == "straight_segment_end":
        longitudinal = max(L - LANE_END_BUFFER_DEFAULT - ctx.sign_spawn_distance, L * 0.8)
    elif strategy == "straight_segment_designated_stop":
        longitudinal = L * 0.5
    else:
        longitudinal = L * 0.5  # defensive default

    # Lateral offset: push sign just outside the right edge of the lane (or left/center).
    half_w = ctx.lane_width / 2 + 0.8
    if rule.lateral_side == "right":
        lateral = half_w
    elif rule.lateral_side == "left":
        lateral = -half_w
    else:  # center
        lateral = 0.0

    return Placement(
        longitudinal_offset=longitudinal,
        lateral_offset=lateral,
        strategy=strategy,
    )
