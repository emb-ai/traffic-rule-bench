"""
Factorized benchmark space definition.

Defines all axes, sign-topology compatibility filters, and exact counting.
Pure Python — no MetaDrive/env dependency required for counting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# Axis A: Topology templates with route intent
# ============================================================================

@dataclass(frozen=True)
class TopologyTemplate:
    block_id: str           # S, C, r, R, T, X, O
    route_intent: str       # through, exit_0, exit_1, exit_2, right, straight, left
    map_string: str         # block sequence string (e.g. "SS", "SX")
    socket_index: int       # which socket the route_intent maps to (0-based)
    block_name: str         # human-readable name

TOPOLOGY_TEMPLATES: List[TopologyTemplate] = [
    TopologyTemplate("S", "through",  "SS", 0, "Straight"),
    TopologyTemplate("C", "through",  "SC", 0, "Curve"),
    TopologyTemplate("r", "merge",    "Sr", 0, "InRamp merge"),    # spawn on ramp, merge into main
    TopologyTemplate("R", "exit",     "SR", 0, "OutRamp exit"),    # spawn on main, exit to ramp
    TopologyTemplate("T", "exit_0",   "ST", 0, "T-Intersection exit 0"),
    TopologyTemplate("T", "exit_1",   "ST", 1, "T-Intersection exit 1"),
    TopologyTemplate("X", "right",    "SX", 0, "X-Intersection right"),
    TopologyTemplate("X", "straight", "SX", 1, "X-Intersection straight"),
    TopologyTemplate("X", "left",     "SX", 2, "X-Intersection left"),
    TopologyTemplate("O", "exit_0",   "SO", 0, "Roundabout exit 0"),
    TopologyTemplate("O", "exit_1",   "SO", 1, "Roundabout exit 1"),
    TopologyTemplate("O", "exit_2",   "SO", 2, "Roundabout exit 2"),
]

TEMPLATE_ID_MAP = {i: t for i, t in enumerate(TOPOLOGY_TEMPLATES)}

# ============================================================================
# Axis B-F: Explicit grid axes
# ============================================================================

LANE_NUM_GRID = [2, 3, 4, 5]       # 4 values
LANE_WIDTH_GRID = [3.7]            # 1 value — fixed at MetaDrive default, does not affect agent behavior

# Semantic spawn lane positions, mapped to actual indices at decode time
SPAWN_LANE_SEMANTIC = ["left", "center", "right"]  # 3 values

def resolve_spawn_lane(semantic: str, lane_num: int) -> int:
    if semantic == "left":
        return 0
    elif semantic == "center":
        return lane_num // 2
    elif semantic == "right":
        return lane_num - 1
    raise ValueError(f"Unknown spawn lane: {semantic}")

def effective_spawn_lanes(lane_num: int) -> List[str]:
    """Return deduplicated semantic spawn lanes for given lane_num."""
    seen = {}
    result = []
    for s in SPAWN_LANE_SEMANTIC:
        idx = resolve_spawn_lane(s, lane_num)
        if idx not in seen:
            seen[idx] = s
            result.append(s)
    return result
# sampled from nuPlan initial_speed (see benchmark_runner.py). The constant is
# kept only as a fallback when nuPlan statistics are unavailable.
N_SPAWN_VELOCITY_SAMPLES = 5
SPAWN_VELOCITY_CLIP = (0.0, 22.0)  # nuPlan range

# accident_prob for detour scenes (3 values), non-detour gets [0.0] only
ACCIDENT_PROB_DETOUR = [0.0, 0.3, 0.5]
ACCIDENT_PROB_DEFAULT = [0.0]

# ============================================================================
# Axis G: Sign types with compatibility filters
# ============================================================================

# All PGMap-compatible sign types (44)
PGMAP_SIGN_KEYS = [
    # Universal (16): any topology, any lane_num ≥ 1, min_length 10m
    "stop", "speed20", "speed30", "speed40", "speed60",
    "zone_speed20", "zone_speed30", "zone_speed40", "zone_speed60",
    "minspeed30", "minspeed40", "minspeed50",
    "no_stop", "bus_station", "only_auto",
    "no_overtaking",     # dual PGMap/SUMO support
    # End-of-zone informational (9): cancel a speed/zone restriction
    "end_speed20", "end_speed30", "end_speed40", "end_speed60",
    "end_zone_speed20", "end_zone_speed30", "end_zone_speed40", "end_zone_speed60",
    "end_all",           # cancel all restrictions
    # Multi-lane (2): lane_num ≥ 2, min_length 15m
    "no_entry", "no_traffic",
    # Restricted-lane (4): lane_num ≥ 2, min_length 30m
    "bus_lane_road", "bike_lane_road", "bus_lane", "bike_lane",
    # End-of-restricted-lane standalone (4): informational
    "end_bus_lane_road", "end_bike_lane_road", "end_bus_lane", "end_bike_lane",
    # Exit-to-restricted-lane (4): 5.13.1/5.13.2/5.13.3/5.13.4 — placed on
    # intersection approach (right lane), indicate outgoing road has a
    # dedicated lane for buses/bikes. lane_num ≥ 2, T/X blocks only.
    "exit_to_bus_lane", "exit_to_bus_lane_left",
    "exit_to_bike_lane", "exit_to_bike_lane_left",
    # Detour (3): lane_num ≥ 3, min_length 30m
    "detour_right", "detour_left", "detour_either",
    # Turn-prohibition (3): T/X blocks only
    "no_right_turn", "no_left_turn", "no_uturn",
    # Intersection direction/priority (7): T/X blocks only
    "pg_direction",      # PGMap-native direction sign with turn layout
    "yield",             # yield to main road traffic at intersection
    "main_road",         # informational: you have priority
    "secondary_road",    # informational: intersection with secondary road
    "secondary_road_left",   # informational: secondary road on the left
    "secondary_road_right",  # informational: secondary road on the right
    "end_main_road",     # informational: priority road ends
]

# Sign categories for compatibility
UNIVERSAL_SIGNS = [
    "stop", "speed20", "speed30", "speed40", "speed60",
    "zone_speed20", "zone_speed30", "zone_speed40", "zone_speed60",
    "minspeed30", "minspeed40", "minspeed50",
    "no_stop", "bus_station", "only_auto",
    "no_overtaking",
]  # 16 signs

# Informational end-of-zone signs — never violate; placed standalone for vocabulary coverage.
END_OF_ZONE_SIGNS = [
    "end_speed20", "end_speed30", "end_speed40", "end_speed60",
    "end_zone_speed20", "end_zone_speed30", "end_zone_speed40", "end_zone_speed60",
    "end_all",
]  # 9 signs, any topology

MULTI_LANE_SIGNS = ["no_entry", "no_traffic"]  # 2 signs, need lane_num >= 2

RESTRICTED_LANE_SIGNS = [
    "bus_lane_road", "bike_lane_road", "bus_lane", "bike_lane",
]  # 4 signs, need lane_num >= 2

# Standalone end-of-restricted-lane (informational): same lane_num >= 2 requirement
END_RESTRICTED_LANE_SIGNS = [
    "end_bus_lane_road", "end_bike_lane_road", "end_bus_lane", "end_bike_lane",
]  # 4 signs

# Exit-to-restricted-lane: placed at the intersection approach to indicate the
# outgoing road has a dedicated lane. T/X blocks only, lane_num >= 2.
EXIT_RESTRICTED_LANE_SIGNS = [
    "exit_to_bus_lane", "exit_to_bus_lane_left",
    "exit_to_bike_lane", "exit_to_bike_lane_left",
]  # 4 signs

DETOUR_SIGNS = ["detour_right", "detour_left", "detour_either"]  # 3 signs, need lane_num >= 3

TURN_PROHIBITION_SIGNS = ["no_right_turn", "no_left_turn", "no_uturn"]  # 3 signs, T/X only

INTERSECTION_PRIORITY_SIGNS = [
    "pg_direction", "yield", "main_road",
    "secondary_road", "secondary_road_left", "secondary_road_right",
    "end_main_road",
]  # 7 signs, T/X only

INTERSECTION_BLOCK_IDS = {"T", "X"}

# All currently active PG block types (mirrors TOPOLOGY_TEMPLATES coverage)
ALL_BLOCK_IDS = {"S", "C", "r", "R", "T", "X", "O", "y", "Y"}

# Per-sign block-type whitelists. If a sign appears in this dict, it is ONLY
# allowed on the listed block types (overrides the broad category filter).
# Signs not listed here use the default category-based compatibility.
SIGN_BLOCK_OVERRIDES = {
    "stop":          {"X", "T", "O"},                         # intersections + roundabout only
    "bus_station":   {"S", "C", "y", "Y"},                    # straight/curve/bottleneck only
    "no_overtaking": {"S", "C", "y", "Y", "r", "R"},          # non-intersection road only
}

# Signs that need a real physical detour (alternative route avoiding the sign edge)
# to be a meaningful test. These come from the CityMap pipeline in the unified
# benchmark — PGMap's linear topology cannot provide the alternative.
PROHIBITION_NEEDING_DETOUR = {
    "no_entry", "no_traffic",
    "no_right_turn", "no_left_turn", "no_uturn",
}

# Signs excluded from PGMap (documented)
EXCLUDED_FROM_PGMAP = [
    "direction",        # SUMO EdgeRoadNetwork only (use pg_direction for PGMap)
    "traffic_light",    # needs SUMO signal phases
    "oneway_r",         # SUMO EdgeRoadNetwork only
    "oneway_l",         # SUMO EdgeRoadNetwork only
    "oneway_s",         # SUMO EdgeRoadNetwork only
    "lane_allowed_dir", # SUMO EdgeRoadNetwork only (use pg_direction for PGMap)
]


def compatible_signs(block_id: str, lane_num: int,
                     exclude_prohibition_detour: bool = True) -> List[str]:
    """Return list of compatible sign keys for a given block type and lane_num.

    Args:
        exclude_prohibition_detour: when True (default), excludes the 5 signs
            in PROHIBITION_NEEDING_DETOUR which are routed through the CityMap
            pipeline instead. Set False to get the legacy full set.
    """
    signs = list(UNIVERSAL_SIGNS)         # always included
    signs.extend(END_OF_ZONE_SIGNS)        # informational, any topology

    if lane_num >= 2:
        signs.extend(MULTI_LANE_SIGNS)
        signs.extend(RESTRICTED_LANE_SIGNS)
        signs.extend(END_RESTRICTED_LANE_SIGNS)  # informational, paired with begin

    if lane_num >= 3:
        signs.extend(DETOUR_SIGNS)

    if block_id in INTERSECTION_BLOCK_IDS:
        signs.extend(TURN_PROHIBITION_SIGNS)
        signs.extend(INTERSECTION_PRIORITY_SIGNS)
        if lane_num >= 2:
            signs.extend(EXIT_RESTRICTED_LANE_SIGNS)

    if exclude_prohibition_detour:
        signs = [s for s in signs if s not in PROHIBITION_NEEDING_DETOUR]

    # Per-sign block-type whitelist (overrides broad category filter)
    signs = [
        s for s in signs
        if s not in SIGN_BLOCK_OVERRIDES or block_id in SIGN_BLOCK_OVERRIDES[s]
    ]
    return signs


def sign_accident_combos(block_id: str, lane_num: int) -> List[Tuple[str, float]]:
    """Return list of (sign_key, accident_prob) valid combos for a cell."""
    signs = compatible_signs(block_id, lane_num)
    combos = []
    for sk in signs:
        if sk in DETOUR_SIGNS and lane_num >= 3:
            for ap in ACCIDENT_PROB_DETOUR:
                combos.append((sk, ap))
        else:
            combos.append((sk, 0.0))
    return combos


# ============================================================================
# Counting
# ============================================================================

@dataclass
class CellInfo:
    """One cell in the factorized space = (template_id, lane_num)."""
    template_id: int
    lane_num: int
    sign_accident_combos: List[Tuple[str, float]]
    n_spawn_lanes: int  # after deduplication
    # Constant across all cells:
    # n_lane_widths = 5, n_spawn_vels = 7, n_profiles = from arg

    def cell_size(self) -> int:
        # spawn_velocity is no longer an axis; it's sampled N_SPAWN_VELOCITY_SAMPLES
        # times per base-scene at materialization.
        return (
            len(self.sign_accident_combos)
            * len(LANE_WIDTH_GRID)
            * self.n_spawn_lanes
        )


def build_cells() -> List[CellInfo]:
    """Build all valid (template_id, lane_num) cells with their sign combos."""
    cells = []
    for tid, tmpl in enumerate(TOPOLOGY_TEMPLATES):
        for ln in LANE_NUM_GRID:
            combos = sign_accident_combos(tmpl.block_id, ln)
            n_sl = len(effective_spawn_lanes(ln))
            cells.append(CellInfo(
                template_id=tid,
                lane_num=ln,
                sign_accident_combos=combos,
                n_spawn_lanes=n_sl,
            ))
    return cells


def compute_total_size() -> Tuple[int, List[CellInfo]]:
    """Compute total benchmark space size (profile is sampled per-scene, not an axis)."""
    cells = build_cells()
    total = sum(c.cell_size() for c in cells)
    return total, cells


def build_block_catalog() -> dict:
    """Build block catalog for serialization."""
    catalog = {
        "topology_templates": [],
        "axes": {
            "LANE_NUM": LANE_NUM_GRID,
            "LANE_WIDTH": LANE_WIDTH_GRID,
            "SPAWN_LANE_SEMANTIC": SPAWN_LANE_SEMANTIC,
            "N_SPAWN_VELOCITY_SAMPLES": N_SPAWN_VELOCITY_SAMPLES,
            "SPAWN_VELOCITY_CLIP_MS": list(SPAWN_VELOCITY_CLIP),
            "ACCIDENT_PROB_DETOUR": ACCIDENT_PROB_DETOUR,
            "ACCIDENT_PROB_DEFAULT": ACCIDENT_PROB_DEFAULT,
        },
        "sign_categories": {
            "universal": UNIVERSAL_SIGNS,
            "multi_lane": MULTI_LANE_SIGNS,
            "restricted_lane": RESTRICTED_LANE_SIGNS,
            "end_restricted_lane": END_RESTRICTED_LANE_SIGNS,
            "exit_restricted_lane": EXIT_RESTRICTED_LANE_SIGNS,
            "detour": DETOUR_SIGNS,
            "turn_prohibition": TURN_PROHIBITION_SIGNS,
            "intersection_priority": INTERSECTION_PRIORITY_SIGNS,
        },
        "excluded_from_pgmap": EXCLUDED_FROM_PGMAP,
        "intersection_block_ids": sorted(INTERSECTION_BLOCK_IDS),
    }
    for tid, tmpl in enumerate(TOPOLOGY_TEMPLATES):
        catalog["topology_templates"].append({
            "template_id": tid,
            "block_id": tmpl.block_id,
            "route_intent": tmpl.route_intent,
            "map_string": tmpl.map_string,
            "socket_index": tmpl.socket_index,
            "block_name": tmpl.block_name,
        })
    return catalog


def build_compatibility_matrix() -> dict:
    """Build compatibility matrix: (block_id, lane_num) -> sign combos."""
    matrix = {}
    for tmpl in TOPOLOGY_TEMPLATES:
        for ln in LANE_NUM_GRID:
            key = f"{tmpl.block_id}_L{ln}"
            if key not in matrix:
                combos = sign_accident_combos(tmpl.block_id, ln)
                matrix[key] = {
                    "block_id": tmpl.block_id,
                    "lane_num": ln,
                    "n_sign_combos": len(combos),
                    "sign_combos": [{"sign": s, "accident_prob": a} for s, a in combos],
                    "compatible_signs": compatible_signs(tmpl.block_id, ln),
                }
    return matrix


def build_space_manifest() -> dict:
    """Build full space manifest with exact counts.

    Agent profile is not an axis — it is sampled per-scene from nuPlan using
    the scene's deterministic seed, so every scene has a unique realistic profile.
    """
    total, cells = compute_total_size()
    cell_breakdown = []
    for c in cells:
        tmpl = TOPOLOGY_TEMPLATES[c.template_id]
        cell_breakdown.append({
            "template_id": c.template_id,
            "block_id": tmpl.block_id,
            "route_intent": tmpl.route_intent,
            "lane_num": c.lane_num,
            "n_sign_accident_combos": len(c.sign_accident_combos),
            "n_spawn_lanes": c.n_spawn_lanes,
            "cell_size": c.cell_size(),
        })

    return {
        "total_combinations": total,
        "n_topology_templates": len(TOPOLOGY_TEMPLATES),
        "n_lane_nums": len(LANE_NUM_GRID),
        "n_lane_widths": len(LANE_WIDTH_GRID),
        "n_spawn_velocity_samples": N_SPAWN_VELOCITY_SAMPLES,
        "profile_sampling": "per-scene from nuPlan (seed = 100000 + flat_index), applied to NPC only",
        "spawn_velocity_sampling": (
            f"per-scene: {N_SPAWN_VELOCITY_SAMPLES} draws from nuPlan initial_speed, "
            f"clipped to {SPAWN_VELOCITY_CLIP} m/s"
        ),
        "n_cells": len(cells),
        "cell_breakdown": cell_breakdown,
        "counting_formula": (
            f"sum over {len(cells)} cells of "
            f"(n_sign_accident * {len(LANE_WIDTH_GRID)} * n_spawn_lanes)"
        ),
    }


# ============================================================================
# CLI: print space info
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print factorized space info")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    total, cells = compute_total_size()
    print(f"Total benchmark space (base scenes): {total:,} combinations")
    print(f"  Topology templates: {len(TOPOLOGY_TEMPLATES)}")
    print(f"  Lane nums: {LANE_NUM_GRID}")
    print(f"  Lane widths: {LANE_WIDTH_GRID}")
    print(f"  spawn_velocity: {N_SPAWN_VELOCITY_SAMPLES} nuPlan samples per base-scene "
          f"(clip {SPAWN_VELOCITY_CLIP} m/s)")
    print(f"  Agent profiles: sampled per-scene from nuPlan, applied to NPC only")
    print(f"  Cells (template × lane_num): {len(cells)}")

    # Per-block-type breakdown
    from collections import defaultdict
    by_block = defaultdict(int)
    for c in cells:
        tmpl = TOPOLOGY_TEMPLATES[c.template_id]
        by_block[tmpl.block_id] += c.cell_size()
    print("\n  Per block type:")
    for bid, count in sorted(by_block.items(), key=lambda x: -x[1]):
        print(f"    {bid}: {count:,}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "benchmark_space.json", "w") as f:
            json.dump(build_space_manifest(), f, indent=2)
        with open(save_dir / "block_catalog.json", "w") as f:
            json.dump(build_block_catalog(), f, indent=2)
        with open(save_dir / "compatibility_matrix.json", "w") as f:
            json.dump(build_compatibility_matrix(), f, indent=2)
        print(f"\nSaved to {save_dir}/")


# Pipeline constants
PIPELINE_PGMAP = "pgmap"
PIPELINE_CITYMAP = "citymap"
PIPELINE_NONE = "none"  # no synthetic (P-only)

PDD_TO_SYNTHETIC: Dict[str, Tuple[str, List[str]]] = {
    # Priority (2.x) — intersection topologies, PGMap
    "2.1":    (PIPELINE_PGMAP, ["main_road"]),
    "2.2":    (PIPELINE_PGMAP, ["end_main_road"]),
    "2.3.1":  (PIPELINE_PGMAP, ["secondary_road"]),
    "2.3.2":  (PIPELINE_PGMAP, ["secondary_road_right"]),
    "2.3.3":  (PIPELINE_PGMAP, ["secondary_road_left"]),
    "2.4":    (PIPELINE_PGMAP, ["yield"]),
    "2.5":    (PIPELINE_NONE,  []),

    # Prohibitions with physical detour -- CityMap
    "3.1":    (PIPELINE_CITYMAP, ["no_entry"]),
    "3.2":    (PIPELINE_CITYMAP, ["no_traffic"]),
    "3.18.1": (PIPELINE_CITYMAP, ["no_right_turn"]),
    "3.18.2": (PIPELINE_CITYMAP, ["no_left_turn"]),
    "3.19":   (PIPELINE_CITYMAP, ["no_uturn"]),

    "3.20":   (PIPELINE_PGMAP, ["no_overtaking"]),
    "3.24":   (PIPELINE_NONE,  []),                   
    "3.25":   (PIPELINE_NONE,  []),                   
    "3.27":   (PIPELINE_NONE,  []),                
    "3.31":   (PIPELINE_PGMAP, ["end_all"]),

    # Mandatory (4.x)
    "4.1.1":  (PIPELINE_NONE, []),
    "4.1.2":  (PIPELINE_NONE, []),
    "4.1.3":  (PIPELINE_NONE, []),
    "4.1.4":  (PIPELINE_NONE, []),
    "4.1.5":  (PIPELINE_NONE, []),
    "4.1.6":  (PIPELINE_NONE, []),
    "4.2.1":  (PIPELINE_PGMAP, ["detour_right"]),
    "4.2.2":  (PIPELINE_PGMAP, ["detour_left"]),
    "4.2.3":  (PIPELINE_PGMAP, ["detour_either"]),
    "4.6":    (PIPELINE_PGMAP, ["minspeed30", "minspeed40", "minspeed50"]),

    # Informational (5.x)
    "5.3":    (PIPELINE_NONE,  []),
    "5.5":    (PIPELINE_NONE,  []),
    "5.7.1":  (PIPELINE_NONE,  []),
    "5.7.2":  (PIPELINE_NONE,  []),
    "5.11.1": (PIPELINE_PGMAP, ["bus_lane_road"]),
    "5.11.2": (PIPELINE_PGMAP, ["bike_lane_road"]),
    "5.12.1": (PIPELINE_PGMAP, ["end_bus_lane_road"]),
    "5.12.2": (PIPELINE_PGMAP, ["end_bike_lane_road"]),
    "5.13.1": (PIPELINE_PGMAP, ["exit_to_bus_lane"]),
    "5.13.2": (PIPELINE_PGMAP, ["exit_to_bus_lane_left"]),
    "5.13.3": (PIPELINE_PGMAP, ["exit_to_bike_lane"]),
    "5.13.4": (PIPELINE_PGMAP, ["exit_to_bike_lane_left"]),
    "5.14.1": (PIPELINE_PGMAP, ["bus_lane"]),
    "5.14.2": (PIPELINE_PGMAP, ["bike_lane"]),
    "5.14.3": (PIPELINE_PGMAP, ["end_bus_lane"]),
    "5.15.2": (PIPELINE_NONE,  []),
    "5.16":   (PIPELINE_NONE,  []),
    "5.31":   (PIPELINE_PGMAP, ["zone_speed20", "zone_speed30", "zone_speed40", "zone_speed60"]),
    "5.32":   (PIPELINE_PGMAP, ["end_zone_speed20", "end_zone_speed30", "end_zone_speed40", "end_zone_speed60"]),
}


ZONE_LENGTH_BINS: List[float] = [20.0, 40.0, 60.0, 80.0]

@dataclass(frozen=True)
class PairedSignBundle:
    pdd_a: str             # PDD code of start-of-zone sign
    pdd_b: str             # PDD code of end-of-zone sign
    sign_key_a: str        
    sign_key_b: str        
    min_lane_num: int = 1  

PAIRED_SIGN_BUNDLES: List[PairedSignBundle] = [
    # Main road zone
    PairedSignBundle("2.1", "2.2", "main_road", "end_main_road"),
    # Speed limit zones (4 speed variants each)
    PairedSignBundle("3.24", "3.25", "speed20", "end_speed20"),
    PairedSignBundle("3.24", "3.25", "speed30", "end_speed30"),
    PairedSignBundle("3.24", "3.25", "speed40", "end_speed40"),
    PairedSignBundle("3.24", "3.25", "speed60", "end_speed60"),
    # "End of all restrictions" — universal closer
    PairedSignBundle("3.24", "3.31", "speed20", "end_all"),
    PairedSignBundle("3.24", "3.31", "speed30", "end_all"),
    PairedSignBundle("3.24", "3.31", "speed40", "end_all"),
    PairedSignBundle("3.24", "3.31", "speed60", "end_all"),
    PairedSignBundle("3.27", "3.31", "no_stop", "end_all"),
    PairedSignBundle("3.20", "3.31", "no_overtaking", "end_all"),
    # Zone speed (5.31 → 5.32), 4 variants
    PairedSignBundle("5.31", "5.32", "zone_speed20", "end_zone_speed20"),
    PairedSignBundle("5.31", "5.32", "zone_speed30", "end_zone_speed30"),
    PairedSignBundle("5.31", "5.32", "zone_speed40", "end_zone_speed40"),
    PairedSignBundle("5.31", "5.32", "zone_speed60", "end_zone_speed60"),
    # Restricted-lane pairs (require lane_num >= 2)
    PairedSignBundle("5.11.1", "5.12.1", "bus_lane_road", "end_bus_lane_road", min_lane_num=2),
    PairedSignBundle("5.11.2", "5.12.2", "bike_lane_road", "end_bike_lane_road", min_lane_num=2),
    PairedSignBundle("5.14.1", "5.14.3", "bus_lane", "end_bus_lane", min_lane_num=2),
    PairedSignBundle("5.14.2", "5.14.4", "bike_lane", "end_bike_lane", min_lane_num=2),
]


# ============================================================================
# Axis J: Bike-related signs. Drives spawn of bicycle NPCs (§14).
# ============================================================================

BIKE_RELATED_SIGNS = {
    "bike_lane_road", "bike_lane",
    "end_bike_lane_road", "end_bike_lane",
    "exit_to_bike_lane", "exit_to_bike_lane_left",
}


# ============================================================================
# PDD → human-readable Russian name (for manifests / reports)
# ============================================================================

PDD_HUMAN_NAME = {
    "2.1": "Главная дорога",
    "2.2": "Конец главной дороги",
    "2.3.1": "Пересечение с второстепенной дорогой",
    "2.3.2": "Примыкание второстепенной дороги (справа)",
    "2.3.3": "Примыкание второстепенной дороги (слева)",
    "2.4": "Уступите дорогу",
    "2.5": "Движение без остановки запрещено",
    "3.1": "Въезд запрещён",
    "3.2": "Движение запрещено",
    "3.18.1": "Поворот направо запрещён",
    "3.18.2": "Поворот налево запрещён",
    "3.19": "Разворот запрещён",
    "3.20": "Обгон запрещён",
    "3.24": "Ограничение максимальной скорости",
    "3.25": "Конец ограничения максимальной скорости",
    "3.27": "Остановка запрещена",
    "3.31": "Конец всех ограничений",
    "4.1.1": "Движение прямо",
    "4.1.2": "Движение направо",
    "4.1.3": "Движение налево",
    "4.1.4": "Движение прямо или направо",
    "4.1.5": "Движение прямо или налево",
    "4.1.6": "Движение направо или налево",
    "4.2.1": "Объезд препятствия справа",
    "4.2.2": "Объезд препятствия слева",
    "4.2.3": "Объезд препятствия справа или слева",
    "4.6": "Ограничение минимальной скорости",
    "5.3": "Дорога для автомобилей",
    "5.5": "Дорога с односторонним движением",
    "5.7.1": "Выезд на дорогу с односторонним движением (направо)",
    "5.7.2": "Выезд на дорогу с односторонним движением (налево)",
    "5.11.1": "Дорога с полосой для маршрутных ТС",
    "5.11.2": "Дорога с полосой для велосипедов",
    "5.12.1": "Конец дороги с полосой для маршрутных ТС",
    "5.12.2": "Конец дороги с полосой для велосипедов",
    "5.13.1": "Выезд на дорогу с полосой для маршрутных ТС",
    "5.13.2": "Выезд на дорогу с полосой для маршрутных ТС (левый запрещён)",
    "5.13.3": "Выезд на дорогу с полосой для велосипедов",
    "5.13.4": "Выезд на дорогу с полосой для велосипедов (левый запрещён)",
    "5.14.1": "Полоса для маршрутных ТС",
    "5.14.2": "Полоса для велосипедистов",
    "5.14.3": "Конец полосы для маршрутных ТС",
    "5.15.2": "Направления движения по полосе",
    "5.16": "Место остановки автобуса/троллейбуса",
    "5.31": "Зона с ограничением максимальной скорости",
    "5.32": "Конец зоны с ограничением максимальной скорости",
}
