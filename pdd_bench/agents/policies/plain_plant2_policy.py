"""Plain PlanT2 policy — raw PlanT2 transformer without sign-compliance overlay.

Used by `--policy plant2` in run_benchmark.py to evaluate the bare
PlanT2 network. For the sign-compliant variant (`--policy plant2_rule`),
see `plant2_sign_compliant.PlanT2SignCompliantPolicy` (this is its parent).
"""

from pdd_bench.agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy


class PlainPlanT2Policy(PlanT2SignCompliantPolicy):
    APPLY_RULE_OVERLAY = False
