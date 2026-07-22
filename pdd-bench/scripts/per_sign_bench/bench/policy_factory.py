"""Load NN-policy checkpoints and resolve the BasePolicy class."""
from __future__ import annotations

import sys

from bench._paths import SDC_ROOT

try:
    import torch
except Exception:
    torch = None


def _load_policy_models(policy: str, model_path: str | None, plant2_action_mode: str = "pid"):
    """Load model checkpoints and resolve the BasePolicy class to instantiate.

    Returns dict with:
      "policy_cls":   BasePolicy subclass; set as agent_policy in env config
                      OR instantiated manually for IDM-family.
    """
    policy_cls = None

    if policy == "carl":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl")
        CARL_ADAPTER_PATH = SDC_ROOT / "carl_in_metadrive"
        aps = str(CARL_ADAPTER_PATH)
        if aps not in sys.path:
            sys.path.insert(0, aps)
        from agents.policies.plain_carl_policy import PlainCarlPolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PlainCarlPolicy.set_checkpoint(model_path, device=device)
        policy_cls = PlainCarlPolicy
    elif policy == "plant2":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2")
        PLANT2_PATH = SDC_ROOT / "plant2"
        if str(PLANT2_PATH) not in sys.path:
            sys.path.insert(0, str(PLANT2_PATH))
        from agents.policies.plain_plant2_policy import PlainPlanT2Policy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PlainPlanT2Policy.set_checkpoint(
            model_path, PLANT2_PATH, device=device, action_mode=plant2_action_mode,
        )
        policy_cls = PlainPlanT2Policy
    elif policy == "carl_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl_rule")
        CARL_ADAPTER_PATH = SDC_ROOT / "carl_in_metadrive"
        aps = str(CARL_ADAPTER_PATH)
        if aps not in sys.path:
            sys.path.insert(0, aps)
        from agents.policies.carl_sign_compliant import CarlSignCompliantPolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        CarlSignCompliantPolicy.set_checkpoint(model_path, device=device)
        policy_cls = CarlSignCompliantPolicy
    elif policy == "plant2_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2_rule")
        PLANT2_PATH = SDC_ROOT / "plant2"
        if str(PLANT2_PATH) not in sys.path:
            sys.path.insert(0, str(PLANT2_PATH))
        from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PlanT2SignCompliantPolicy.set_checkpoint(
            model_path, PLANT2_PATH, device=device, action_mode=plant2_action_mode,
        )
        policy_cls = PlanT2SignCompliantPolicy

    return {
        "policy_cls": policy_cls,
    }


def make_ego_policy(policy_type, models, base_env, seed,
                    ego_variant="default", ego_sample_seed_base=42,
                    ego_hold_speed_ms=None):
    """Resolve + instantiate the ego BasePolicy and apply its IDM ego variant.

    All policies implement the uniform BasePolicy.act(name) interface — built-in
    IDM-family/expert classes here, plus the NN classes resolved by
    _load_policy_models (PlainCarl/PlainPlanT2 raw, or the *SignCompliant overlays).

    For IDM-family egos the variant selects fixed defaults (`default`) or sampled
    params (`s1..s4`, seed = ego_sample_seed_base + seed + k*1000003).

    Returns (policy_obj, sampled_ego_params); sampled_ego_params is None unless an
    `s*` variant was applied.
    """
    import os

    from metadrive.policy.idm_policy import IDMPolicy
    from metadrive.policy.expert_policy import ExpertPolicy
    from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
    from agents.policies.curve_aware_idm import CurveAwareIDMPolicy
    from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy
    from factorized_space.ego_defaults import (apply_ego_defaults, apply_ego_sampled,
                                               sample_ego_params)

    # Base idm gets the same defensive layer as the rule expert/NPC
    # (curvature cap, steering lookahead, braking for crossing traffic) —
    # the idm/idm_rule pair then differs only in sign knowledge.
    # EGO_CURVE_AWARE=0 restores the raw MetaDrive IDMPolicy.
    ego_idm_cls = (IDMPolicy if os.environ.get("EGO_CURVE_AWARE", "1") == "0"
                   else CurveAwareIDMPolicy)

    builtin = {
        "idm": ego_idm_cls,
        "comprehensive_rule_expert": ComprehensiveRuleExpertPolicy,
        "rule_compliant": RuleCompliantExpertPolicy,
        "ppo_lidar": ExpertPolicy,
    }
    policy_cls = builtin.get(policy_type) or models.get("policy_cls")
    if policy_cls is None:
        raise RuntimeError(f"policy_cls for --policy {policy_type} not loaded; "
                           "check _load_policy_models")

    policy_obj = policy_cls(base_env.vehicle, seed)

    # NPC profile is applied via IDMPolicy class attrs globally. Keep ego
    # IDM-family policies deterministic by overriding defaults on the instance.
    sampled_ego_params = None
    if policy_type in ("idm", "comprehensive_rule_expert"):
        if ego_variant.startswith("s") and ego_variant[1:].isdigit():
            k = int(ego_variant[1:])
            sample_seed = int(ego_sample_seed_base) + int(seed) + k * 1000003
            # variant_k selects the style quantile band in EGO_SAMPLER=styles
            # (s1 slow … s4 fast); ignored in legacy mode.
            sampled_ego_params = sample_ego_params(sample_seed, variant_k=k)
            apply_ego_sampled(policy_obj, sampled_ego_params)
        else:
            apply_ego_defaults(policy_obj)
            # "Ego holds v0" (braking spawn, default variant only): the desired
            # speed is raised to the spawn speed so a sign-unaware IDM enters
            # the zone above the limit (otherwise it decays to 36 km/h right at
            # the sign — vacuous on v40). The rule expert gets the same — the
            # sign caps it (mixin), so the pair gap = pure sign knowledge.
            # s1–s4 are left alone (styles). The curvature cap
            # (CurveAwareIDMPolicy/expert) still applies on top.
            # Rollback: EGO_HOLD_V0=0.
            if ego_hold_speed_ms and os.environ.get("EGO_HOLD_V0", "1") != "0":
                v_kmh = float(ego_hold_speed_ms) * 3.6
                policy_obj.NORMAL_SPEED = max(float(policy_obj.NORMAL_SPEED), v_kmh)
                policy_obj.MAX_SPEED = max(float(policy_obj.MAX_SPEED), 1.15 * v_kmh)

    return policy_obj, sampled_ego_params
