import random
import numpy as np

from traffic_signs.stop_sign import StopSign


def _same_road_direction(idx_a, idx_b):
    if idx_a is None or idx_b is None:
        return False
    try:
        if len(idx_a) < 2 or len(idx_b) < 2:
            return False
    except Exception:
        return False
    return idx_a[0] == idx_b[0] and idx_a[1] == idx_b[1]


def _agent_direction_indices(base_env):
    indices = []
    try:
        agent = getattr(base_env, "vehicle", None) or getattr(base_env, "agent", None)
        if agent is None:
            return indices
        lane = getattr(agent, "lane", None)
        lane_idx = getattr(lane, "index", None) if lane is not None else None
        if lane_idx is not None:
            indices.append(lane_idx)
        nav = getattr(agent, "navigation", None)
        if nav is not None and hasattr(nav, "current_ref_lanes"):
            for ref_lane in nav.current_ref_lanes or []:
                ref_idx = getattr(ref_lane, "index", None)
                if ref_idx is not None:
                    indices.append(ref_idx)
    except Exception:
        return indices
    return indices


def add_stop_signs_to_map(
    env,
    stop_sign_probability=0.3,
    max_signs=10,
    min_distance=25.0,
    min_lane_length=15.0,
    sign_class=StopSign,
    longitudinal_offset=-3.0,
    shoulder_width_min=0.8,
):
    """
    Add stop signs to the current map with basic filtering.

    Args:
        env: MetaDrive env (wrapped ok)
        stop_sign_probability: probability to place a sign on a lane
        max_signs: maximum number of signs
        min_distance: minimum distance between signs
        min_lane_length: minimum lane length to consider
        sign_class: traffic sign class to spawn
        longitudinal_offset: offset relative to lane end
        shoulder_width_min: minimum shoulder width for lateral placement
    """
    try:
        base_env = env.unwrapped
        while hasattr(base_env, "env"):
            base_env = base_env.env

        if not hasattr(base_env, "engine") or not hasattr(
            base_env.engine, "traffic_sign_manager"
        ):
            return 0

        sign_mgr = base_env.engine.traffic_sign_manager
        if sign_mgr is None:
            return 0

        if sign_mgr.signs:
            sign_ids = [sign.id for sign in sign_mgr.signs if hasattr(sign, "id")]
            if sign_ids:
                base_env.engine.clear_objects(sign_ids)
                if hasattr(sign_mgr, "spawned_objects") and sign_mgr.spawned_objects is not None:
                    for sign_id in sign_ids:
                        sign_mgr.spawned_objects.pop(sign_id, None)
            sign_mgr.signs.clear()

        if not hasattr(sign_mgr, "spawned_objects") or sign_mgr.spawned_objects is None:
            sign_mgr.spawned_objects = {}

        if not hasattr(base_env, "current_map"):
            return 0

        road_network = base_env.current_map.road_network
        available_lanes = []

        for start_road in road_network.graph.values():
            for end_road in start_road.values():
                for lane in end_road:
                    if lane.length >= min_lane_length:
                        available_lanes.append(lane)

        if not available_lanes:
            return 0

        signs_added = 0
        added_positions = []

        random.shuffle(available_lanes)

        for lane in available_lanes:
            if signs_added >= max_signs:
                break

            if random.random() < stop_sign_probability:
                try:
                    target_long = lane.length + longitudinal_offset
                    long_pos = np.clip(target_long, 0.1, lane.length - 0.1)

                    lane_width = lane.width_at(long_pos)
                    shoulder_width = max(lane_width, shoulder_width_min)
                    lateral_offset = lane_width / 2 + shoulder_width

                    sign_position = lane.position(long_pos, lateral_offset)

                    too_close = False
                    for existing_pos in added_positions:
                        distance = np.sqrt(
                            (sign_position[0] - existing_pos[0]) ** 2
                            + (sign_position[1] - existing_pos[1]) ** 2
                        )
                        if distance < min_distance:
                            too_close = True
                            break

                    if too_close:
                        continue

                    sign_mgr.add_sign(
                        sign_class,
                        lane=lane,
                        # longitudinal_offset=longitudinal_offset,
                        lateral_offset=lateral_offset,
                        use_random_lane=False,
                    )

                    added_positions.append(sign_position)
                    signs_added += 1
                except Exception:
                    continue

        return signs_added
    except Exception as exc:
        print(f"Error adding stop signs: {exc}")
        return 0


def add_signs_by_count(env, sign_counts_dict, min_distance=15.0):
    """
    Add traffic signs to the map by  count for each type.
    
    """
    try:
        base_env = env.unwrapped
        while hasattr(base_env, "env"):
            base_env = base_env.env

        if not hasattr(base_env, "engine") or not hasattr(base_env.engine, "traffic_sign_manager"):
            return {}

        sign_mgr = base_env.engine.traffic_sign_manager
        if sign_mgr is None:
            return {}

        # clear existing signs
        if sign_mgr.signs:
            sign_ids = [sign.id for sign in sign_mgr.signs if hasattr(sign, "id")]
            if sign_ids:
                base_env.engine.clear_objects(sign_ids)
                if hasattr(sign_mgr, "spawned_objects") and sign_mgr.spawned_objects is not None:
                    for sign_id in sign_ids:
                        sign_mgr.spawned_objects.pop(sign_id, None)
            sign_mgr.signs.clear()

        if not hasattr(base_env, "current_map"):
            return {}

        # get all available lanes
        road_network = base_env.current_map.road_network
        available_lanes = []
        for start_road in road_network.graph.values():
            for end_road in start_road.values():
                for lane in end_road:
                    if lane.length >= 15.0:
                        available_lanes.append(lane)

        # Prefer lanes matching current agent driving direction
        dir_indices = _agent_direction_indices(base_env)
        if dir_indices:
            filtered = [
                lane for lane in available_lanes
                if any(_same_road_direction(getattr(lane, "index", None), d_idx) for d_idx in dir_indices)
            ]
            if filtered:
                available_lanes = filtered

        if not available_lanes:
            return {}

        random.shuffle(available_lanes)
        added_positions = []
        result_counts = {sign_class.__name__: 0 for sign_class in sign_counts_dict.keys()}
        
        all_signs_to_add = []
        for sign_class, target_count in sign_counts_dict.items():
            all_signs_to_add.extend([sign_class] * target_count)
        random.shuffle(all_signs_to_add)  
        
        lane_idx = 0
        sign_idx = 0
        attempts = 0
        max_attempts = max(len(all_signs_to_add) * len(available_lanes) * 4, 100)

        while sign_idx < len(all_signs_to_add) and attempts < max_attempts:
            attempts += 1
            sign_class = all_signs_to_add[sign_idx]
            lane = available_lanes[lane_idx % len(available_lanes)]
            lane_idx += 1
            
            try:
                sign = sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)
                sign_position = sign.position

                # Check distance to other signs
                too_close = False
                for existing_pos in added_positions:
                    distance = np.sqrt(
                        (sign_position[0] - existing_pos[0]) ** 2
                        + (sign_position[1] - existing_pos[1]) ** 2
                    )
                    if distance < min_distance:
                        too_close = True
                        break

                if too_close:
                    try:
                        sign_id = getattr(sign, "id", None)
                        if sign_id is not None:
                            try:
                                base_env.engine.clear_objects([sign_id])
                            except Exception:
                                pass
                        if hasattr(sign, 'destroy'):
                            sign.destroy()
                        if sign in sign_mgr.signs:
                            sign_mgr.signs.remove(sign)
                        if sign_id is not None and hasattr(sign_mgr, "spawned_objects"):
                            sign_mgr.spawned_objects.pop(sign_id, None)
                    except:
                        pass
                    continue

                added_positions.append(sign_position)
                result_counts[sign_class.__name__] += 1
                sign_idx += 1  
            except Exception as e:
                if sign_idx < 10: 
                    print(f"  Warning: Failed to add {sign_class.__name__}: {e}")
                continue

        return result_counts
    except Exception as exc:
        print(f"Error adding signs: {exc}")
        return {}


__all__ = ["add_stop_signs_to_map", "add_signs_by_count"]
