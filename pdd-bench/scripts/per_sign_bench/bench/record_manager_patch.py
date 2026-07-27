"""RecordManager tolerance patch for post-reset sign / aux spawns.

Between env.reset() finishing (RecordManager.after_reset clears reset_frame
and current_frames to None) and the first env.step() (before_step creates a
new current_frames), RecordManager has no active frame — but TrafficSignManager
and yield aux agents spawn objects in that window, which would crash on
`current_frame.spawn_info` / policy_info assertions.

Sign objects are static — we don't need them in the recording (they're in the
sidecar). Aux vehicles have physics bodies and are captured from the first
step via collect_objects_states.

Shared by expert_replay.py and yield_sign/run_benchmark.py.
"""
from __future__ import annotations

_RM_PATCHED = False


def patch_record_manager_once() -> None:
    """Idempotent monkey-patch of MetaDrive RecordManager (safe if MD missing)."""
    global _RM_PATCHED
    if _RM_PATCHED:
        return
    try:
        from metadrive.manager.record_manager import RecordManager
        from metadrive.utils.utils import is_map_related_class
        from metadrive.constants import ObjectState
    except ImportError as exc:
        print(f"[warn] RecordManager patch deferred: {exc}")
        return

    def _tolerant_add_spawn_info(self, obj, object_class, kwargs):
        if is_map_related_class(object_class) or not self.engine.record_episode:
            return
        if self.reset_frame is None and self.current_frames is None:
            return
        try:
            frame = self.current_frame
        except (TypeError, AttributeError):
            return
        name = obj.name
        if name in frame.spawn_info:
            return
        self._episode_obj_names.add(name)
        frame.spawn_info[name] = {
            ObjectState.CLASS: object_class,
            ObjectState.INIT_KWARGS: kwargs,
            ObjectState.NAME: name,
        }

    RecordManager.add_spawn_info = _tolerant_add_spawn_info

    import copy as _copy

    def _tolerant_collect_objects_states(self):
        from metadrive.utils.utils import is_map_related_instance
        policy_mapping = self.engine.get_policies()
        frame = self.current_frame
        for name, obj in self.engine.get_objects().items():
            if is_map_related_instance(obj):
                continue
            if getattr(obj, "_body", None) is None:
                continue  # static / bodyless (e.g. traffic signs)
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
        frame._agent_to_object = _copy.deepcopy(
            self.engine.agent_manager._agent_to_object
        )
        frame._object_to_agent = _copy.deepcopy(
            self.engine.agent_manager._object_to_agent
        )

    RecordManager.collect_objects_states = _tolerant_collect_objects_states

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
            return
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


# Back-compat alias used by expert_replay.py
_patch_record_manager_once = patch_record_manager_once
