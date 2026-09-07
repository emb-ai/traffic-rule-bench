"""Load IDM / PPO / CaRL / PlanT2 checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch

from traffic_bench.eval.engine.sim.checkpoints import (
    PLAIN_PLANT2_POLICIES,
    PLANT2_POLICIES,
    resolve_nn_checkpoint,
)

EVAL_DIR = Path(__file__).resolve().parent.parent
SDC_ROOT = EVAL_DIR.parent.parent

def resolve_model_path(policy: str, model_path: str | None) -> str | None:
    """Use ``--model-path`` when set; else fall back to repo defaults."""
    return resolve_nn_checkpoint(policy, model_path)


def _nn_device() -> str:
    import os

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_vis = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"[run] NN device={device} "
        f"(cuda_available={torch.cuda.is_available()}, n_visible={n_vis}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})",
        flush=True,
    )
    return device


def _load_policy_models(policy: str, model_path: str | None, plant2_action_mode: str = "pid"):
    policy_cls = None

    if policy == "carl":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl")
        from traffic_bench.agents.carl import PlainCarlPolicy
        PlainCarlPolicy.set_checkpoint(model_path, device=_nn_device())
        policy_cls = PlainCarlPolicy
    elif policy in PLAIN_PLANT2_POLICIES:
        if not model_path:
            raise ValueError(f"--model-path is required for --policy {policy}")
        PLANT2_PATH = SDC_ROOT / "third_party" / "plant2"
        from traffic_bench.agents.plant2 import PlainPlanT2Policy
        PlainPlanT2Policy.set_checkpoint(
            model_path, PLANT2_PATH, device=_nn_device(), action_mode=plant2_action_mode,
            # No governor. The adapter default (50 km/h) clipped desired_speed for
            # every PlanT2 policy, which makes a 4.6 plate above 50 km/h impossible to
            # satisfy by construction and pins the pretrained model at a flat 50 under
            # every plate. The longitudinal target stays bounded by the model's own
            # speed_limit token (80 km/h), the value training saw.
            max_speed_kmh=None,
        )
        policy_cls = PlainPlanT2Policy
    elif policy == "carl_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl_rule")
        from traffic_bench.agents.carl_rule import CarlSignCompliantPolicy
        CarlSignCompliantPolicy.set_checkpoint(model_path, device=_nn_device())
        policy_cls = CarlSignCompliantPolicy
    elif policy == "plant2_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2_rule")
        PLANT2_PATH = SDC_ROOT / "third_party" / "plant2"
        from traffic_bench.agents.plant2_rule import PlanT2SignCompliantPolicy
        PlanT2SignCompliantPolicy.set_checkpoint(
            model_path, PLANT2_PATH, device=_nn_device(), action_mode=plant2_action_mode,
            # No governor. The adapter default (50 km/h) clipped desired_speed for
            # every PlanT2 policy, which makes a 4.6 plate above 50 km/h impossible to
            # satisfy by construction and pins the pretrained model at a flat 50 under
            # every plate. The longitudinal target stays bounded by the model's own
            # speed_limit token (80 km/h), the value training saw.
            max_speed_kmh=None,
        )
        policy_cls = PlanT2SignCompliantPolicy

    return {
        "policy_cls": policy_cls,
    }


