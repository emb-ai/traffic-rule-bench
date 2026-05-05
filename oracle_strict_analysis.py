"""Strict-quality oracle analysis (compliance-first ranking).

Run as a script:
    python3 oracle_strict_analysis.py

Or open in VSCode/Jupyter — cells are marked with `# %%`.

Strategy:
  1. Hard filter (universal): crashed=False AND out_of_road=False
  2. Sign-specific filter:
       - sign 3.1 (no entry): dest is OPTIONAL — stopping before the sign with
         viol=0 IS the correct behavior, not a failure.
       - other signs: dest=True is required.
  3. Rank candidates by:
       primary:   compliance  (viol == 0  → 1, else 0)
       secondary: Comfort × time_efficiency (harmonic mean, higher is better)
  4. Compare with previous oracle.
"""

# %% ──────────────────────────────────────────────────────────────────────────
# 1. Setup — paths and constants
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

# EDIT these paths if needed:
RUN_DIR = Path("/Users/victoria_s/sdc_new_signs/sign_experts_20260428_174805")
ALL_RUNS = RUN_DIR / "all_runs.jsonl"
OUT_PATH = RUN_DIR / "oracle_strict_summary.json"

# EDIT these to tune the oracle:
HORIZON = 600              # for time_efficiency = 1 - final_step / HORIZON
COMFORT_FIELD = "smoothness_ratio"   # or "frame_smooth_ratio" for softer comfort

# Signs where dest is OPTIONAL (stopping is the correct behavior).
# For all other signs, dest=True is required.
SIGNS_WITHOUT_DEST_REQUIREMENT = {"3.1"}   # no-entry: stopping is correct

print(f"Run dir:       {RUN_DIR}")
print(f"all_runs.jsonl: exists={ALL_RUNS.exists()}  ({ALL_RUNS.stat().st_size if ALL_RUNS.exists() else 0} bytes)")
print(f"Output:        {OUT_PATH}")


# %% ──────────────────────────────────────────────────────────────────────────
# 2. Load data
# ────────────────────────────────────────────────────────────────────────────
def load_runs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def normalize_sign(s):
    return s.replace("_", ".") if isinstance(s, str) else s


rows = load_runs(ALL_RUNS)
valid_rows = [r for r in rows if r.get("valid")]
print(f"Loaded {len(rows)} rows  (valid: {len(valid_rows)})")
print(f"Policies: {sorted({r.get('policy') for r in valid_rows})}")
print(f"Variants: {sorted({r.get('variant') for r in valid_rows})}")
print(f"Signs: {sorted({normalize_sign(r.get('sign_code') or r.get('sign_slug')) for r in valid_rows})}")


# %% ──────────────────────────────────────────────────────────────────────────
# 3. Per-episode metrics: time_efficiency, comfort, hmean
# ────────────────────────────────────────────────────────────────────────────
def time_efficiency(r: dict) -> float:
    """1 - final_step / HORIZON. Higher = finished faster."""
    if HORIZON <= 0:
        return 0.0
    return max(0.0, 1.0 - int(r.get("final_step", 0) or 0) / HORIZON)


def comfort(r: dict) -> float:
    return float(r.get(COMFORT_FIELD, 0.0) or 0.0)


def harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean of two non-negative numbers. Returns 0 if either is 0."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def ct_hmean(r: dict) -> float:
    """Comfort × time_efficiency (harmonic mean). The oracle's ranking score."""
    return harmonic_mean(comfort(r), time_efficiency(r))


# Sanity check on a few rows
for r in valid_rows[:3]:
    print(f"  {r.get('policy')}/{r.get('variant')} {r.get('scene_id')}: "
          f"steps={r.get('final_step')}  smooth={comfort(r):.2f}  "
          f"time_eff={time_efficiency(r):.2f}  CT_hmean={ct_hmean(r):.3f}  "
          f"dest={r.get('arrived_dest')}  crash={r.get('crashed')}  "
          f"oor={r.get('out_of_road')}")


# %% ──────────────────────────────────────────────────────────────────────────
# 4. Filter: hard filter (no crash, no oor) + per-sign dest requirement
# ────────────────────────────────────────────────────────────────────────────
def sign_of(r: dict) -> str:
    return normalize_sign(r.get("sign_code") or r.get("sign_slug") or "?")


def is_compliant(r: dict) -> bool:
    """0 violations recorded."""
    return int(r.get("total_violations", 0) or 0) == 0


def passes_filter(r: dict) -> bool:
    """Hard universal filter + per-sign dest requirement.

    For sign 3.1 (no-entry): dest is OPTIONAL because stopping in front of the
    sign is the correct behavior. We only require: not crashed, not oor.
    For other signs: dest=True is required.
    """
    if r.get("crashed") or r.get("out_of_road"):
        return False
    if sign_of(r) in SIGNS_WITHOUT_DEST_REQUIREMENT:
        return True   # dest not required for these signs
    return bool(r.get("arrived_dest"))


success_rows = [r for r in valid_rows if passes_filter(r)]

print(f"Total valid rows:        {len(valid_rows)}")
print(f"Pass filter:             {len(success_rows)}  "
      f"({100*len(success_rows)/len(valid_rows):.1f}%)")
print(f"Filtered out:            {len(valid_rows) - len(success_rows)}")
print()
print(f"Signs without dest requirement: {sorted(SIGNS_WITHOUT_DEST_REQUIREMENT)}")
print()
print("Per-policy filter counts:")
by_pol = Counter(r.get("policy") for r in success_rows)
for pol, n in by_pol.most_common():
    total_pol = sum(1 for r in valid_rows if r.get("policy") == pol)
    print(f"  {pol:<16} {n:>4}/{total_pol:>4}  ({100*n/max(total_pol,1):.0f}%)")
print()
print("Per-sign filter counts (showing how many candidates per sign):")
by_sign_filt = Counter(sign_of(r) for r in success_rows)
for sg, n in sorted(by_sign_filt.items()):
    total = sum(1 for r in valid_rows if sign_of(r) == sg)
    print(f"  {sg:<10} {n:>4}/{total:>4}  ({100*n/max(total,1):.0f}%)")


# %% ──────────────────────────────────────────────────────────────────────────
# 5. Oracle selection: pick best CT_hmean per (sign, scene_id)
# ────────────────────────────────────────────────────────────────────────────
def scene_key(r: dict):
    return (
        normalize_sign(r.get("sign_code") or r.get("sign_slug") or "?"),
        r.get("scene_id") or r.get("scene_uid"),
    )


# Group success rows by scene
by_scene = defaultdict(list)
for r in success_rows:
    by_scene[scene_key(r)].append(r)

# All scenes (also ones with NO success → skipped)
all_scenes_set = {scene_key(r) for r in valid_rows}

oracle_picks = []
skipped = []
for key in sorted(all_scenes_set):
    candidates = by_scene.get(key, [])
    if not candidates:
        skipped.append(key)
        continue
    # Rank: PRIMARY = compliance (viol==0), SECONDARY = CT_hmean.
    # Tiebreakers: comfort, then time_efficiency.
    # max() with tuple → larger tuple wins.
    best = max(candidates, key=lambda r: (
        1 if is_compliant(r) else 0,    # primary: compliance
        ct_hmean(r),                     # secondary: comfort × time
        comfort(r),                      # tiebreaker
        time_efficiency(r),              # tiebreaker
    ))
    oracle_picks.append(best)

print(f"Total scenes:            {len(all_scenes_set)}")
print(f"Oracle picks (selected): {len(oracle_picks)}")
print(f"Skipped scenes (no success across all policies/variants): {len(skipped)}")
if skipped:
    print(f"\nFirst 10 skipped scenes:")
    for sg, sid in skipped[:10]:
        print(f"  {sg} / {sid}")


# %% ──────────────────────────────────────────────────────────────────────────
# 6. Oracle aggregate metrics
# ────────────────────────────────────────────────────────────────────────────
def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "dest_rate": sum(1 for r in rows if r.get("arrived_dest")) / n,
        "crash_rate": sum(1 for r in rows if r.get("crashed")) / n,
        "oor_rate": sum(1 for r in rows if r.get("out_of_road")) / n,
        "ego_fault_rate": sum(1 for r in rows if r.get("crashed_ego_fault")) / n,
        "npc_fault_rate": sum(1 for r in rows if r.get("crashed_npc_fault")) / n,
        "avg_viol": mean(int(r.get("total_violations", 0) or 0) for r in rows),
        "viol0_rate": sum(1 for r in rows
                          if int(r.get("total_violations", 0) or 0) == 0) / n,
        "avg_comfort": mean(comfort(r) for r in rows),
        "avg_time_eff": mean(time_efficiency(r) for r in rows),
        "avg_ct_hmean": mean(ct_hmean(r) for r in rows),
        "avg_steps": mean(int(r.get("final_step", 0) or 0) for r in rows),
        "avg_reward": mean(float(r.get("total_reward", 0.0) or 0.0) for r in rows),
    }


m = aggregate(oracle_picks)
print(f"=== Oracle (strict) metrics ===")
print(f"  n_picks:              {m['n']}")
print(f"  dest_rate:            {m['dest_rate']:.2f}  (always 1.00 by filter)")
print(f"  crash_rate:           {m['crash_rate']:.2f}  (always 0.00 by filter)")
print(f"  oor_rate:             {m['oor_rate']:.2f}  (always 0.00 by filter)")
print(f"  avg_violations:       {m['avg_viol']:.2f}")
print(f"  viol0_rate:           {m['viol0_rate']:.2f}")
print(f"  avg_comfort ({COMFORT_FIELD}): {m['avg_comfort']:.3f}")
print(f"  avg_time_eff:         {m['avg_time_eff']:.3f}")
print(f"  ★ avg_CT_hmean:       {m['avg_ct_hmean']:.3f}  ← oracle ranking metric")
print(f"  avg_steps:            {m['avg_steps']:.0f}")


# %% ──────────────────────────────────────────────────────────────────────────
# 7. Pick distribution: which policy/variant wins most scenes
# ────────────────────────────────────────────────────────────────────────────
def policy_label(r: dict) -> str:
    p, v = r.get("policy"), r.get("variant")
    return f"{p}/{v}" if p == "comprehensive" else p


pick_dist = Counter(policy_label(r) for r in oracle_picks)
total = sum(pick_dist.values())

print("=== Oracle pick distribution ===")
for pol, n in pick_dist.most_common():
    print(f"  {pol:<25} {n:>4}  ({100*n/total:>5.1f}%)")

# Aggregate by base policy
print("\nGrouped by base policy:")
base = Counter()
for k, n in pick_dist.items():
    base[k.split("/", 1)[0]] += n
for pol, n in base.most_common():
    print(f"  {pol:<16} {n:>4}  ({100*n/total:>5.1f}%)")


# %% ──────────────────────────────────────────────────────────────────────────
# 8. Per-sign × policy/variant breakdown
# ────────────────────────────────────────────────────────────────────────────
by_sign_pick = defaultdict(Counter)
for r in oracle_picks:
    sign = normalize_sign(r.get("sign_code") or r.get("sign_slug"))
    by_sign_pick[sign][policy_label(r)] += 1

signs = sorted(by_sign_pick)
all_keys = sorted({k for s in signs for k in by_sign_pick[s].keys()})

print("=== Per-sign × policy/variant pick counts ===")
hdr = f"  {'policy/variant':<25}"
for s in signs:
    hdr += f"{s:>10}"
hdr += f"{'TOTAL':>8}"
print(hdr)
print("  " + "-" * (25 + 10 * len(signs) + 8))
for k in all_keys:
    line = f"  {k:<25}"
    total_k = 0
    for s in signs:
        n = by_sign_pick[s].get(k, 0)
        total_k += n
        line += f"{n:>10}"
    line += f"{total_k:>8}"
    print(line)


# %% ──────────────────────────────────────────────────────────────────────────
# 9. Per-sign metrics for the oracle picks
# ────────────────────────────────────────────────────────────────────────────
print("=== Per-sign oracle metrics ===")
print(f"  {'sign':<10} {'n':>5} {'viol':>7} {'comfort':>9} {'time_eff':>10} {'CT_hmean':>10} {'steps':>7}")
print("  " + "-" * 70)
for sign in signs:
    eps = [r for r in oracle_picks
           if normalize_sign(r.get("sign_code") or r.get("sign_slug")) == sign]
    mm = aggregate(eps)
    print(f"  {sign:<10} {mm['n']:>5} {mm['avg_viol']:>7.2f} "
          f"{mm['avg_comfort']:>9.3f} {mm['avg_time_eff']:>10.3f} "
          f"{mm['avg_ct_hmean']:>10.3f} {mm['avg_steps']:>7.0f}")


# %% ──────────────────────────────────────────────────────────────────────────
# 10. Compare with OLD oracle (compliance > dest, hard filter no-crash/no-oor)
# ────────────────────────────────────────────────────────────────────────────
MIN_ENGAGED_STEPS = 200  # for meaningful_compliance

def is_disqualified(r):
    return bool(r.get("crashed")) or bool(r.get("out_of_road"))

def meaningful_compliant(r):
    if is_disqualified(r):
        return False
    if int(r.get("total_violations", 0) or 0) > 0:
        return False
    return r.get("arrived_dest") or int(r.get("final_step", 0) or 0) >= MIN_ENGAGED_STEPS

def old_rank(r):
    return (
        1 if meaningful_compliant(r) else 0,
        1 if r.get("arrived_dest") else 0,
        -int(r.get("total_violations", 0) or 0),
        comfort(r),
        int(r.get("final_step", 0) or 0),
    )


# Old oracle: filter is_disqualified, then rank by old logic
by_scene_old = defaultdict(list)
for r in valid_rows:
    by_scene_old[scene_key(r)].append(r)

old_picks = []
for key, eps in sorted(by_scene_old.items()):
    valid_eps = [r for r in eps if not is_disqualified(r)]
    if not valid_eps:
        continue
    old_picks.append(max(valid_eps, key=old_rank))

m_old = aggregate(old_picks)

print("=== Comparison: NEW (CT_hmean) vs OLD (compliance>dest) oracle ===")
print(f"  {'metric':<25} {'OLD':>10} {'NEW':>10}")
print("  " + "-" * 50)
for label, key in [
    ("n_picks", "n"),
    ("dest_rate", "dest_rate"),
    ("crash_rate", "crash_rate"),
    ("oor_rate", "oor_rate"),
    ("avg_violations", "avg_viol"),
    ("avg_comfort", "avg_comfort"),
    ("avg_time_eff", "avg_time_eff"),
    ("★ avg_CT_hmean", "avg_ct_hmean"),
    ("avg_steps", "avg_steps"),
]:
    fmt = ".3f" if key in ("avg_comfort", "avg_time_eff", "avg_ct_hmean") else \
          ".0f" if key in ("n", "avg_steps") else ".2f"
    print(f"  {label:<25} {format(m_old[key], fmt):>10} {format(m[key], fmt):>10}")

# How often picks differ
old_keys = {(scene_key(r), policy_label(r), r.get("seed")) for r in old_picks}
new_keys = {(scene_key(r), policy_label(r), r.get("seed")) for r in oracle_picks}
overlap = len(old_keys & new_keys)
print(f"\n  Picks identical to OLD: {overlap}/{len(oracle_picks)}  "
      f"({100*overlap/max(len(oracle_picks),1):.0f}%)")


# %% ──────────────────────────────────────────────────────────────────────────
# 11. Save oracle to JSON (optional)
# ────────────────────────────────────────────────────────────────────────────
def serialize_pick(r):
    return {
        "scene_id": r.get("scene_id"),
        "scene_uid": r.get("scene_uid"),
        "sign_code": normalize_sign(r.get("sign_code") or r.get("sign_slug")),
        "policy": r.get("policy"),
        "variant": r.get("variant"),
        "seed": r.get("seed"),
        "pkl_path": r.get("pkl_path"),
        "sidecar_path": r.get("sidecar_path"),
        "final_step": r.get("final_step"),
        "smoothness_ratio": r.get("smoothness_ratio"),
        "frame_smooth_ratio": r.get("frame_smooth_ratio"),
        "comfort": comfort(r),
        "time_efficiency": time_efficiency(r),
        "ct_hmean": ct_hmean(r),
        "total_violations": r.get("total_violations"),
    }


summary = {
    "strategy": "filter (dest=True, crash=False, oor=False) → rank by CT_hmean",
    "horizon": HORIZON,
    "comfort_field": COMFORT_FIELD,
    "n_total_scenes": len(all_scenes_set),
    "n_picks": len(oracle_picks),
    "n_skipped_scenes": len(skipped),
    "skipped_scenes": [{"sign": sg, "scene": sid} for sg, sid in skipped],
    "oracle_metrics": m,
    "pick_distribution": dict(pick_dist),
    "scenes": [serialize_pick(r) for r in oracle_picks],
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
print(f"Saved: {OUT_PATH}")
