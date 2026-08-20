"""Plain CaRL policy — raw CaRL PPO behaviour without sign-compliance overlay.

Used by `--policy carl` in run_benchmark.py to evaluate the bare CaRL
network. For the sign-compliant variant (used by `--policy carl_rule`), see
`carl_sign_compliant.CarlSignCompliantPolicy` (this is its parent).
"""

from pdd_bench.agents.policies.carl_sign_compliant import CarlSignCompliantPolicy


class PlainCarlPolicy(CarlSignCompliantPolicy):
    APPLY_RULE_OVERLAY = False
