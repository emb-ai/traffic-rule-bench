# agents/

Thin policy files + sign-compliance handlers. Hydra ids stay
(`policy=comprehensive_rule_expert`, `carl_rule`, …).

| File | Class | CLI id |
|---|---|---|
| `idm_rule.py` | `ComprehensiveRuleExpertPolicy` | `comprehensive_rule_expert` |
| `ppo_rule.py` | `RuleCompliantExpertPolicy` | `rule_compliant` |
| `carl.py` / `carl_rule.py` | plain / overlay CaRL | `carl` / `carl_rule` |
| `plant2.py` / `plant2_rule.py` | plain / overlay PlanT2 | `plant2` / `plant2_rule` |
| `curve_aware_idm.py` | `CurveAwareIDMPolicy` | unwired |

`compliance/mixin.py` is the only import policies need. Handlers live next to
the sign family (`junction.py`, `dual_path.py`, …).
