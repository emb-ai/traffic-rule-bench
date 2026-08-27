"""Canonical policy ids and their legacy spellings.

The rule-augmented IDM and PPO experts were renamed to match their modules
(``agents/idm_rule.py``, ``agents/ppo_rule.py``). Old spellings still appear
in recorded trajectories, eval outputs and scripts, so every entry point maps
them to the canonical id with :func:`canonical_policy_name`.
"""
from __future__ import annotations

LEGACY_POLICY_NAMES: dict[str, str] = {
    "comprehensive_rule_expert": "idm_rule",
    "rule_compliant": "ppo_rule",
}


def canonical_policy_name(name: str) -> str:
    """``comprehensive_rule_expert[_s1]`` → ``idm_rule[_s1]``; others unchanged."""
    text = str(name or "").strip()
    for old, new in LEGACY_POLICY_NAMES.items():
        if text == old:
            return new
        if text.startswith(old + "_"):
            return new + text[len(old):]
    return text
