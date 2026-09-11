"""Unit tests for 4.1.x outgoing helpers (no MetaDrive import)."""

from __future__ import annotations

import math
import unittest

from traffic_bench.signs.outgoing import (
    allow_forbid_outgoing,
    filter_same_edge_lane_ids,
    heading_delta_to_dir,
    is_internal_lane_id,
    normalize_turn_direction,
)


def _edge_id(lane_index):
    """Mirror BaseTrafficSign._sumo_edge_id_from_lane_index for tests."""
    if lane_index is None:
        return None
    lane_str = str(lane_index)
    if lane_str.startswith("lane_"):
        lane_str = lane_str[len("lane_") :]
    parts = lane_str.split("_")
    for _ in range(2):
        if len(parts) > 1 and parts[-1].isdigit():
            parts = parts[:-1]
        else:
            break
    return "_".join(parts) if parts else None


class HeadingDeltaTests(unittest.TestCase):
    def test_straight_left_right_uturn(self):
        approach = 0.0
        self.assertEqual(heading_delta_to_dir(approach, 0.0), "s")
        self.assertEqual(heading_delta_to_dir(approach, math.radians(10)), "s")
        self.assertEqual(heading_delta_to_dir(approach, math.pi / 2), "l")
        self.assertEqual(heading_delta_to_dir(approach, -math.pi / 2), "r")
        self.assertEqual(heading_delta_to_dir(approach, math.pi), "t")


class ApproachArmFilterTests(unittest.TestCase):
    def test_keeps_same_edge_peers_drops_other_arms(self):
        approach = "-10857972"
        sign = f"lane_{approach}_0"
        candidates = [
            f"lane_{approach}_0",
            f"lane_{approach}_1",
            "lane_-51250520#3_0",  # other arm / outgoing
            "lane_:96487624_8_0",  # internal
            "junction_96487624",
        ]
        kept = filter_same_edge_lane_ids(sign, candidates, _edge_id)
        self.assertEqual(
            set(kept),
            {f"lane_{approach}_0", f"lane_{approach}_1"},
        )

    def test_internal_helper(self):
        self.assertTrue(is_internal_lane_id("lane_:96487624_8_0"))
        self.assertFalse(is_internal_lane_id("lane_-10857972_1"))


class AllowForbidTests(unittest.TestCase):
    def test_411_left_exit_forbidden(self):
        # Failure-case geometry: straight + left outgoings, plate allows straight.
        left_exit = "1058357057#0"
        straight_exit = "-51250520#3"
        all_outgoing = {left_exit, straight_exit}
        by_dir = {
            "l": {left_exit},
            "r": set(),
            "s": {straight_exit},
            "t": set(),
        }
        allowed, forbidden = allow_forbid_outgoing(all_outgoing, by_dir, {"s"})
        self.assertEqual(allowed, {straight_exit})
        self.assertEqual(forbidden, {left_exit})

    def test_416_straight_forbidden(self):
        left_exit = "L"
        right_exit = "R"
        straight_exit = "S"
        all_outgoing = {left_exit, right_exit, straight_exit}
        by_dir = {
            "l": {left_exit},
            "r": {right_exit},
            "s": {straight_exit},
            "t": set(),
        }
        allowed, forbidden = allow_forbid_outgoing(
            all_outgoing, by_dir, {"l", "r", "t"}
        )
        self.assertEqual(allowed, {left_exit, right_exit})
        self.assertEqual(forbidden, {straight_exit})


class NormalizeTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_turn_direction("left"), "l")
        self.assertEqual(normalize_turn_direction("RIGHT"), "r")
        self.assertEqual(normalize_turn_direction("straight"), "s")
        self.assertEqual(normalize_turn_direction("u-turn"), "t")


if __name__ == "__main__":
    unittest.main()
