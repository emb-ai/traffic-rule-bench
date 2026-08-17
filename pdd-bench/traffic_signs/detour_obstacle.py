"""
Helper to spawn obstacles on a lane for detour signs (4.2.x).

Called from the environment reset to place traffic cones or barriers
at the sign position, giving the agent something to detour around.
"""

import numpy as np
from metadrive.component.static_object.traffic_object import TrafficCone


DETOUR_SIGN_TYPES = {"4.2.1", "4.2.2", "4.2.3"}

# Number of cones forming the obstacle cluster
CONE_COUNT = 4
CONE_SPACING = 1.5  # longitudinal spacing (meters)


def spawn_detour_obstacle(engine, lane, sign):
    """
    Spawn a small cluster of traffic cones on *lane* centred at the sign's
    obstacle position so the vehicle has a visible reason to detour.

    The cones ARE the reason: a 4.2.x frame without them shows an expert
    swerving for nothing, which imitation cannot learn and which no reader of
    the dump would notice. So a failure to spawn is reported, never hidden —
    the same lesson the auxiliary convoys taught at junctions.

    Parameters
    ----------
    engine : MetaDriveEngine
        The simulation engine (used for ``spawn_object``).
    lane : AbstractLane
        The lane where cones should appear.
    sign : DetourSign
        The detour sign instance (provides ``obstacle_long``).

    Returns
    -------
    int
        How many cones were actually spawned (0 means the scene has no obstacle).
    """
    obstacle_long = getattr(sign, "obstacle_long", getattr(sign, "placement_long", lane.length / 2))

    half = (CONE_COUNT - 1) * CONE_SPACING / 2
    spawned = 0
    first_error = None
    for i in range(CONE_COUNT):
        long = obstacle_long - half + i * CONE_SPACING
        long = np.clip(long, 0.5, lane.length - 0.5)
        pos = lane.position(long, 0)  # centre of lane
        heading = lane.heading_theta_at(long)
        try:
            engine.spawn_object(
                TrafficCone,
                lane=lane,
                position=pos,
                heading_theta=heading,
                static=True,
                force_spawn=True,
            )
            spawned += 1
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc

    if spawned < CONE_COUNT:
        print(f"[detour] spawned {spawned}/{CONE_COUNT} cones on {getattr(lane, 'index', lane)}"
              f" at long={obstacle_long:.1f}"
              + (f"; first error: {first_error!r}" if first_error is not None else ""),
              flush=True)
    return spawned
