# agents/

Agent policies and sign-compliance overlays used by the benchmark.


| File                           | Class                           | CLI id                      |
| ------------------------------ | ------------------------------- | --------------------------- |
| `idm_rule.py`                  | `ComprehensiveRuleExpertPolicy` | `comprehensive_rule_expert` |
| `ppo_rule.py`                  | `RuleCompliantExpertPolicy`     | `rule_compliant`            |
| `carl.py` / `carl_rule.py`     | plain / overlay CaRL            | `carl` / `carl_rule`        |
| `plant2.py` / `plant2_rule.py` | plain / overlay PlanT2          | `plant2` / `plant2_rule`    |


Handlers live next to the sign family (`junction.py`, `dual_path.py`, …).