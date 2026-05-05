"""Bare driving policies with NO sign-compliance post-processing.

Used as a comparison baseline against the sign-compliant variants.

    idm        — pure MetaDrive IDMPolicy
    ppo        — pure MetaDrive ExpertPolicy (PPO)
    carl_base  — CaRL PPO network, no sign mixin
    plant2_base — PlanT2 transformer, no sign mixin
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from metadrive.policy.base_policy import BasePolicy


class CarlBasePolicy(BasePolicy):
    """CaRL PPO — no sign compliance. Pure NN driving."""

    _carl_adapter = None
    _carl_checkpoint: Optional[str] = None
    _carl_device: str = "cpu"

    @classmethod
    def set_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> None:
        cls._carl_checkpoint = checkpoint_path
        cls._carl_device = device
        cls._carl_adapter = None  # force reload on next act()

    @classmethod
    def _get_adapter(cls):
        if cls._carl_adapter is None:
            if cls._carl_checkpoint is None:
                raise RuntimeError(
                    "CarlBasePolicy: call set_checkpoint(<path>) before env build."
                )
            from agents.carl_in_metadrive.carl_adapter import CaRLMetaDriveAdapter
            cls._carl_adapter = CaRLMetaDriveAdapter(
                cls._carl_checkpoint, device=cls._carl_device
            )
        return cls._carl_adapter

    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        try:
            self._get_adapter().reset()
        except Exception:
            pass

    def act(self, agent_id=None):
        try:
            action = self._get_adapter().get_action(self.control_object, self.engine)
            return [float(action[0]), float(action[1])]
        except Exception as exc:
            print(f"[CarlBasePolicy] get_action failed: {exc}")
            return [0.0, 0.0]


class PlanT2BasePolicy(BasePolicy):
    """PlanT2 transformer — no sign compliance. Pure NN driving."""

    _plant2_adapter = None
    _plant2_checkpoint: Optional[str] = None
    _plant2_repo_dir: Optional[Path] = None
    _plant2_device: str = "cpu"

    @classmethod
    def set_checkpoint(
        cls, checkpoint_path: str, plant_repo_dir, device: str = "cpu"
    ) -> None:
        cls._plant2_checkpoint = str(checkpoint_path)
        cls._plant2_repo_dir = Path(plant_repo_dir)
        cls._plant2_device = device
        cls._plant2_adapter = None

    @classmethod
    def _get_adapter(cls):
        if cls._plant2_adapter is None:
            if cls._plant2_checkpoint is None or cls._plant2_repo_dir is None:
                raise RuntimeError(
                    "PlanT2BasePolicy: call set_checkpoint(<ckpt>, <repo>) before env build."
                )
            from agents.plant2_in_metadrive.plant2_adapter import PlanT2MetaDriveAdapter
            cls._plant2_adapter = PlanT2MetaDriveAdapter(
                cls._plant2_checkpoint,
                cls._plant2_repo_dir,
                device=cls._plant2_device,
            )
        return cls._plant2_adapter

    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        try:
            self._get_adapter().reset()
        except Exception:
            pass

    def act(self, agent_id=None):
        try:
            action = self._get_adapter().get_action(self.control_object, self.engine)
            return [float(action[0]), float(action[1])]
        except Exception as exc:
            print(f"[PlanT2BasePolicy] get_action failed: {exc}")
            return [0.0, 0.0]
