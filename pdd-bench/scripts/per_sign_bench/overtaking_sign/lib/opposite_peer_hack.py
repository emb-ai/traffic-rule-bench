"""Pretend the opposite directed edge is a same-way left/right peer.

On real SUMO 1+1 maps the oncoming edge is not a lateral neighbor, so IDM/PlanT2
never see it in ``current_ref_lanes``. This hack rewires MetaDrive's
``EdgeRoadNetwork`` graph so the opposite lane appears in ``left_lanes`` /
``right_lanes`` (default: left — overtaking side for right-hand traffic).
"""

from __future__ import annotations

from typing import Literal, Optional

from lib.lane_keys import make_lane_key

Side = Literal["left", "right"]


def resolve_opposite_lane_key(row: dict) -> Optional[str]:
    opp = row.get("opposite_edge_id") or row.get("opposite_edge")
    if not opp:
        return None
    return make_lane_key(str(opp), 0)


def wire_opposite_as_peer(
    env,
    ego_lane_key: str,
    opposite_lane_key: str,
    *,
    side: Side = "left",
) -> bool:
    """Attach ``opposite_lane_key`` as a lateral peer of ``ego_lane_key``.

    Also wires the reverse (ego as peer of opposite) so peer queries are symmetric.
    Returns True if the graph was patched.
    """
    try:
        rn = env.engine.current_map.road_network
        graph = rn.graph
    except Exception as exc:
        print(f"[OppositePeer] no road network: {exc}")
        return False

    if ego_lane_key not in graph or opposite_lane_key not in graph:
        print(
            f"[OppositePeer] missing lanes ego={ego_lane_key} "
            f"opp={opposite_lane_key}"
        )
        return False

    field = "left_lanes" if side == "left" else "right_lanes"
    rev_field = "right_lanes" if side == "left" else "left_lanes"

    def _add_peer(host_key: str, peer_key: str, peer_field: str) -> None:
        info = graph[host_key]
        peers = list(getattr(info, peer_field) or [])
        if peer_key not in peers:
            peers.append(peer_key)
        graph[host_key] = info._replace(**{peer_field: peers})

    _add_peer(ego_lane_key, opposite_lane_key, field)
    _add_peer(opposite_lane_key, ego_lane_key, rev_field)

    # Refresh nav ref lanes so the current episode sees the new peer immediately.
    try:
        vehicle = env.agent
        nav = getattr(vehicle, "navigation", None)
        if nav is not None:
            start = getattr(vehicle.lane, "index", None) or ego_lane_key
            nav.current_ref_lanes = rn.get_peer_lanes_from_index(start)
            nxt = None
            ckpts = getattr(nav, "checkpoints", None) or []
            if len(ckpts) >= 2:
                nxt = ckpts[min(1, len(ckpts) - 1)]
            if nxt:
                nav.next_ref_lanes = rn.get_peer_lanes_from_index(nxt)
            n_peers = len(nav.current_ref_lanes or [])
            print(
                f"[OppositePeer] wired {opposite_lane_key} as {side} of "
                f"{ego_lane_key}  (current_ref_lanes={n_peers})"
            )
        else:
            print(
                f"[OppositePeer] wired {opposite_lane_key} as {side} of "
                f"{ego_lane_key}"
            )
    except Exception as exc:
        print(f"[OppositePeer] wired graph but nav refresh failed: {exc}")

    return True


def enable_idm_overtake_on_policy(policy_obj) -> None:
    """Turn on MetaDrive IDM overtake / lane-change flags when present."""
    if policy_obj is None:
        return
    for name, val in (
        ("enable_idm_overtake", True),
        ("enable_lane_change", True),
    ):
        if hasattr(policy_obj, name):
            try:
                setattr(policy_obj, name, val)
            except Exception:
                pass
