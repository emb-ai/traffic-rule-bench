"""Print actual IDM/model parameters used by each agent in all_runs.jsonl.

Shows for each (policy, variant):
  - One example of ego_idm_params (the dict applied to ego policy at runtime)
  - For comprehensive (IDM-based): NORMAL_SPEED, ACC, DEACC, DISTANCE/TIME_WANTED, ...
  - For carl/plant2/ppo: model checkpoint info (from row metadata)

Usage:
    python3 show_agent_params.py [path/to/all_runs.jsonl]
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = Path("/Users/victoria_s/sdc_new_signs/all_runs.jsonl")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

# ---------------------------------------------------------------------------
# Load and group
# ---------------------------------------------------------------------------
samples: dict = defaultdict(list)
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    pol = r.get("policy", "?")
    var = r.get("variant", "?")
    key = (pol, var)
    if len(samples[key]) < 1:
        samples[key].append(r)

print(f"Loaded {sum(len(v) for v in samples.values())} sample rows from {path.name}")
print(f"Unique (policy, variant) combinations: {len(samples)}\n")


# ---------------------------------------------------------------------------
# Resolve DEFAULT_EGO_PARAMS from ego_defaults.py (for "default" variant)
# ---------------------------------------------------------------------------
def load_default_ego_params() -> dict | None:
    ed = Path(__file__).resolve().parent / \
        "pdd-bench/scripts/per_sign_bench/factorized_space/ego_defaults.py"
    if not ed.exists():
        return None
    text = ed.read_text()
    import re
    m = re.search(r"DEFAULT_EGO_PARAMS\s*=\s*\{([^}]+)\}", text)
    if not m:
        return None
    d = {}
    for line in m.group(1).splitlines():
        kv = re.match(r'\s*"(\w+)"\s*:\s*([\d.]+)', line)
        if kv:
            d[kv.group(1)] = float(kv.group(2))
    return d


DEFAULT_EGO = load_default_ego_params() or {}


# ---------------------------------------------------------------------------
# Print policies summary
# ---------------------------------------------------------------------------
PARAM_KEYS = ["NORMAL_SPEED", "MAX_SPEED", "CREEP_SPEED",
              "ACC_FACTOR", "DEACC_FACTOR",
              "DISTANCE_WANTED", "TIME_WANTED", "LANE_CHANGE_FREQ"]

print("=" * 105)
print("IDM ego parameters (applied to ego policy via apply_ego_sampled / apply_ego_defaults)")
print("=" * 105)
print(f"{'policy':<16}{'variant':<10}{'NORMAL':>9}{'MAX':>9}{'CREEP':>9}{'ACC':>8}"
      f"{'DEACC':>8}{'DIST_W':>10}{'TIME_W':>9}{'LC_FREQ':>10}")
print("-" * 105)

# Sort: comprehensive variants first (default, s1..s4), then others alphabetically
def sort_key(k):
    pol, var = k
    pol_order = 0 if pol == "comprehensive" else 1
    var_order = 0 if var == "default" else (
        int(var[1:]) if var.startswith("s") and var[1:].isdigit() else 99)
    return (pol_order, pol, var_order)

for key in sorted(samples.keys(), key=sort_key):
    pol, var = key
    r = samples[key][0]
    p = r.get("ego_idm_params")

    if isinstance(p, dict):
        line = f"{pol:<16}{var:<10}"
        for k in PARAM_KEYS:
            v = p.get(k)
            line += f"{v:>9.2f}" if isinstance(v, (int, float)) else f"{'-':>9}"
        print(line)
    elif p == "DEFAULT_EGO_PARAMS" and DEFAULT_EGO:
        line = f"{pol:<16}{var:<10}"
        for k in PARAM_KEYS:
            v = DEFAULT_EGO.get(k)
            line += f"{v:>9.2f}" if isinstance(v, (int, float)) else f"{'-':>9}"
        print(line + "   [DEFAULT_EGO_PARAMS]")
    else:
        # carl / plant2 / rule_compliant — they don't use IDM params on ego
        line = f"{pol:<16}{var:<10}"
        line += "   [non-IDM ego policy → ego_idm_params not applicable]"
        print(line)

print()


# ---------------------------------------------------------------------------
# Per-policy meta info — model checkpoints, env details
# ---------------------------------------------------------------------------
print("=" * 105)
print("Per-policy notes")
print("=" * 105)

policy_notes = {
    "comprehensive":   "ComprehensiveRuleExpertPolicy = IDMPolicy + sign-compliance mixin (rules from traffic_signs/*)",
    "rule_compliant":  "RuleCompliantExpertPolicy = ExpertPolicy (small RL policy from MetaDrive) + sign-compliance mixin",
    "carl":            "CarlSignCompliantPolicy   = CaRL PPO net (~700MB ckpt nuPlan_51479_1B/model_best.pth) + sign mixin",
    "plant2":          "PlanT2SignCompliantPolicy = PlanT2 transformer (epoch%3D029_final_3.ckpt) + sign mixin",
}
for pol, note in policy_notes.items():
    n_eps = sum(len(samples[(p, v)]) for (p, v) in samples if p == pol)
    n_vars = len({v for (p, v) in samples if p == pol})
    print(f"  {pol:<16} (variants={n_vars})  {note}")
print()


# ---------------------------------------------------------------------------
# Counts per (policy, variant)
# ---------------------------------------------------------------------------
print("=" * 105)
print("Episode counts per (policy, variant) — across all signs and scenes")
print("=" * 105)
cnt = defaultdict(int)
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    cnt[(r.get("policy"), r.get("variant"))] += 1

for key in sorted(cnt.keys(), key=sort_key):
    pol, var = key
    label = f"{pol}/{var}" if pol == "comprehensive" else pol
    print(f"  {label:<25} {cnt[key]:>5}")
print(f"  {'TOTAL':<25} {sum(cnt.values()):>5}")
