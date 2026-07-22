#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGN_CLASS_MAP = {
    # Source: sdc/pdd-bench/envs/sumo_env.py:SIGN_TYPE_TO_CLASS
    "2.1":    "MainRoadSign",
    "2.2":    "EndMainRoadSmartSign",
    "2.3.1":  "SecondaryRoadSign",
    "2.3.2":  "SecondaryRoadRightSign",
    "2.3.3":  "SecondaryRoadLeftSign",
    "2.4":    "YieldSign",
    "2.5":    "StopSign",
    "3.1":    "NoEntrySign",
    "3.2":    "NoTrafficSign",
    "3.18.1": "NoRightTurnSign",
    "3.18.2": "NoLeftTurnSign",
    "3.19":   "NoUTurnSign",
    "3.20":   "NoOvertakingSign",
    "3.21":   "EndOfNoOvertakingSign",
    "3.24":   "SpeedLimitSign",
    "3.25":   "EndOfSpeedLimitSign",
    "3.27":   "NoStoppingAllowedSign",
    "3.31":   "EndOfAllRestrictionsSign",
    "4.1.1":  "LaneAllowedDirectionSign4_1_1",
    "4.1.2":  "LaneAllowedDirectionSign4_1_2",
    "4.1.3":  "LaneAllowedDirectionSign4_1_3",
    "4.1.4":  "LaneAllowedDirectionSign4_1_4",
    "4.1.5":  "LaneAllowedDirectionSign4_1_5",
    "4.1.6":  "LaneAllowedDirectionSign4_1_6",
    "4.2.1":  "DetourRightSign",
    "4.2.2":  "DetourLeftSign",
    "4.2.3":  "DetourEitherSign",
    "4.6":    "MinimumSpeedLimitSign",
    "5.3":    "OnlyAutoSign",
    "5.4":    "EndOfOnlyAutoSign",
    "5.5":    "OneWayEntrySignS",
    "5.7.1":  "OneWayEntrySignR",
    "5.7.2":  "OneWayEntrySignL",
    "5.11.1": "BusLaneRoadSign",
    "5.11.2": "BikeLaneRoadSign",
    "5.12.1": "EndBusLaneRoadSign",
    "5.12.2": "EndBikeLaneRoadSign",
    "5.13.1": "ExitToBusLaneSign",
    "5.13.2": "ExitToBusLaneSignLeft",
    "5.13.3": "ExitToBikeLaneSign",
    "5.13.4": "ExitToBikeLaneSignLeft",
    "5.14.1": "BusLaneSign",
    "5.14.2": "BikeLaneSign",
    "5.14.3": "EndBusLaneSign",
    "5.14.4": "EndBikeLaneSign",
    "5.15.2": "DirectionSign",
    "5.16":   "BusStationSign",
    "5.21":   "ResidentialZoneSign",
    "5.22":   "EndOfResidentialZoneSign",
    "5.31":   "ZoneSpeedLimitSign",
    "5.32":   "EndOfZoneSpeedLimitSign",
}

NO_ENTRY_SIGNS = {"3.1", "3.2", "3.18.1", "3.18.2", "3.19"}
HORIZON_DEFAULT = 600
BETA_DEFAULT = 0.25
NON_IDM_POLICIES = {"rule_compliant", "carl", "plant2"}
MIN_FINAL_STEP = 30
MIN_ROUTE_COMPLETION = 0.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_sign(s):
    if not isinstance(s, str):
        return s
    return s.replace("_", ".") if ("_" in s and "." not in s) else s


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            print(f"[warn] bad json in {p}", file=sys.stderr)
    return rows


def load_runs(run_dirs, jsonl_files):
    rows = []
    for rd in run_dirs:
        f = Path(rd) / "all_runs.jsonl"
        loaded = load_jsonl(f)
        print(f"  {f}: {len(loaded)} rows")
        rows.extend(loaded)
    for jf in jsonl_files:
        loaded = load_jsonl(jf)
        print(f"  {jf}: {len(loaded)} rows")
        rows.extend(loaded)
    return rows


def is_compliant(r, target_class):
    vbc = r.get("violations_by_class") or {}
    return int(vbc.get(target_class, 0) or 0) == 0


def recompute_dest(r, target_sign, target_class, horizon=HORIZON_DEFAULT):
    arr_prev = bool(r.get("arrived_dest"))
    if target_sign in NO_ENTRY_SIGNS and is_compliant(r, target_class):
        return True
    return arr_prev


def passes_filter(r, target_sign, target_class, horizon=HORIZON_DEFAULT,
                  min_final_step=MIN_FINAL_STEP,
                  min_route_completion=MIN_ROUTE_COMPLETION):
    if not r.get("valid"):
        return False
    if r.get("crashed") or r.get("out_of_road"):
        return False
    if not is_compliant(r, target_class):
        return False
    # Anti-bug for min_step_episodes; use for expert_selection not metric compute
    # (metric compute passes min_final_step=0 to disable)
    if min_final_step > 0:
        fs = int(r.get("final_step") or 0)
        if fs < min_final_step:
            return False
    # route_completion soft-success (off by default, min_route_completion=0.0)
    if min_route_completion > 0.0:
        if float(r.get("route_completion") or 0.0) >= min_route_completion:
            return True
    return recompute_dest(r, target_sign, target_class, horizon)


def time_eff(r, scene_min_step):
    """Step 3: relative time_eff = scene_min_step / final_step ∈ (0, 1].
    """
    fs = max(1, int(r.get("final_step") or 1))
    return scene_min_step / fs


def comfort(r):
    """Step 4."""
    return float(r.get("frame_smooth_ratio") or 0.0)


def pick_best_idm(idm_eps, scene_min_step, beta=BETA_DEFAULT, strategy="f1"):
    """Pre-select the best comprehensive IDM variant.

    strategy:
      'f1'    (default) — by F1(time_eff, comfort); ties prefer 'default', then faster.
      'speed' (legacy)  — by min(final_step).
    """
    if not idm_eps:
        return None, None

    if strategy == "speed":
        default = next((r for r in idm_eps if r.get("variant") == "default"), None)
        samples = [r for r in idm_eps if r.get("variant") != "default"]
        if default is None:
            if not samples:
                return None, None
            winner = min(samples, key=lambda r: int(r.get("final_step") or 10**9))
            return winner, winner.get("variant")
        if not samples:
            return default, "default"
        best_sample = min(samples, key=lambda r: int(r.get("final_step") or 10**9))
        if int(best_sample.get("final_step") or 10**9) < int(default.get("final_step") or 10**9):
            return best_sample, best_sample.get("variant")
        return default, "default"

    # F1-strategy (default)
    scored = []
    for r in idm_eps:
        t = time_eff(r, scene_min_step)
        c = comfort(r)
        scored.append((f1_score(t, c, beta), r))
    scored.sort(key=lambda x: (
        -x[0],
        0 if x[1].get("variant") == "default" else 1,
        int(x[1].get("final_step") or 10**9),
    ))
    winner = scored[0][1]
    return winner, winner.get("variant")


def f1_score(t, c, beta=BETA_DEFAULT):
    if t <= 0.0 or c <= 0.0:
        return 0.0
    b2 = beta * beta
    return (b2 + 1.0) * t * c / (b2 * t + c)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_expert_per_scene(rows, signs, beta=BETA_DEFAULT,
                              horizon=HORIZON_DEFAULT,
                              max_per_sign=None,
                              min_route_completion=MIN_ROUTE_COMPLETION,
                              min_final_step=MIN_FINAL_STEP,
                              idm_pick_strategy="f1",
                              top_n=1):
    """Returns (picks, scene_groups, filter_records).
      picks          — list[dict] (top_n per scene, ranked 1..top_n)
      scene_groups   — dict[(sign,scene)] -> all valid rows
      filter_records — dict[(sign,scene)] -> list[(row, passes_bool)]
    """
    sign_set = set(normalize_sign(s) for s in signs)
    scene_groups = defaultdict(list)
    filter_records = defaultdict(list)

    for r in rows:
        if not r.get("valid"):
            continue
        sign = normalize_sign(r.get("sign_code") or r.get("sign_slug"))
        if sign not in sign_set:
            continue
        scene_key = r.get("scene_uid") or r.get("scene_id")
        if not scene_key:
            continue
        target_class = SIGN_CLASS_MAP.get(sign)
        if target_class is None:
            continue
        scene_groups[(sign, scene_key)].append(r)
        ok = passes_filter(r, sign, target_class, horizon,
                           min_final_step, min_route_completion)
        filter_records[(sign, scene_key)].append((r, ok))

    picks = []
    for (sign, scene_key), eps in scene_groups.items():
        target_class = SIGN_CLASS_MAP[sign]
        passing = [r for r in eps
                   if passes_filter(r, sign, target_class, horizon,
                                    min_final_step, min_route_completion)]
        if not passing:
            continue

        idm_eps = [r for r in passing if r.get("policy") == "comprehensive"]
        best_idm, best_idm_variant = pick_best_idm(
            idm_eps,
            max(1, min(int(r.get("final_step") or 10**9) for r in passing)),
            beta=beta, strategy=idm_pick_strategy)

        candidates = []
        if best_idm is not None:
            candidates.append(best_idm)
        for r in passing:
            if r.get("policy") in NON_IDM_POLICIES:
                candidates.append(r)

        if not candidates:
            continue
        scene_min_step = max(1, min(int(r.get("final_step") or 10**9)
                                     for r in candidates))

        scored = []
        for r in candidates:
            t = time_eff(r, scene_min_step)
            c = comfort(r)
            scored.append((f1_score(t, c, beta), r, t, c))
        scored.sort(key=lambda x: -x[0])

        n_emit = min(top_n, len(scored))
        for rank in range(1, n_emit + 1):
            win_score, winner, win_t, win_c = scored[rank - 1]
            picks.append({
                "sign": sign,
                "scene_id": winner.get("scene_id"),
                "scene_uid": winner.get("scene_uid") or scene_key,
                "scene_key": scene_key,
                "rank": rank,
                "n_emitted_per_scene": n_emit,
                "winner_policy": winner.get("policy"),
                "winner_variant": winner.get("variant"),
                "best_idm_variant": best_idm_variant,
                "final_step": int(winner.get("final_step") or 0),
                "route_completion": float(winner.get("route_completion") or 0.0),
                "time_eff": round(win_t, 6),
                "comfort": round(win_c, 6),
                "f1_score": round(win_score, 6),
                "beta": beta,
                "horizon": horizon,
                "scene_min_step": scene_min_step,
                "passing_candidates_n": len(candidates),
                "all_passing_n": len(passing),
                "all_attempts_n": len(eps),
                "pkl_path": winner.get("pkl_path"),
                "gif_path": winner.get("gif_path"),
                "sidecar_path": winner.get("sidecar_path"),
                "initial_speed_mps": float(winner.get("initial_speed_mps") or 0.0),
                "all_candidates": [
                    {
                        "policy": r.get("policy"),
                        "variant": r.get("variant"),
                        "final_step": int(r.get("final_step") or 0),
                        "time_eff": round(t, 4),
                        "comfort": round(c, 4),
                        "f1": round(s, 4),
                    }
                    for s, r, t, c in scored
                ],
            })

    if max_per_sign is not None:
        by_sign = defaultdict(lambda: defaultdict(list))
        for p in picks:
            by_sign[p["sign"]][p["scene_key"]].append(p)
        limited = []
        for sign, scenes in by_sign.items():
            scene_keys_sorted = sorted(
                scenes.keys(),
                key=lambda k: (
                    -max(p["f1_score"] for p in scenes[k] if p["rank"] == 1),
                    k,
                ),
            )
            for sk in scene_keys_sorted[:max_per_sign]:
                limited.extend(sorted(scenes[sk], key=lambda p: p["rank"]))
        picks = limited

    return picks, scene_groups, filter_records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(picks, scene_groups, filter_records, signs, beta, horizon):
    sign_set = set(normalize_sign(s) for s in signs)

    # 1) Coverage per sign
    sign_total = defaultdict(set)
    sign_passing = defaultdict(set)
    sign_oracle = defaultdict(set)
    for (sign, scene), eps in scene_groups.items():
        sign_total[sign].add(scene)
    for (sign, scene), recs in filter_records.items():
        if any(ok for _, ok in recs):
            sign_passing[sign].add(scene)
    for p in picks:
        sign_oracle[p["sign"]].add(p.get("scene_key") or p.get("scene_uid")
                                     or p.get("scene_id"))

    print("\n" + "=" * 70)
    print("Coverage per sign")
    print("=" * 70)
    print(f"{'sign':<10} {'scenes':>7} {'passing':>8} {'picked':>7} {'cov%':>7}")
    print("-" * 50)
    for sign in sorted(sign_set):
        st = len(sign_total[sign])
        sp = len(sign_passing[sign])
        so = len(sign_oracle[sign])
        cov = so / st if st else 0
        print(f"{sign:<10} {st:>7} {sp:>8} {so:>7} {cov:>6.0%}")

    # 2) Filter rates per (policy, variant)
    pol_stats = defaultdict(lambda: {"total": 0, "pass": 0,
                                     "compl": 0, "arr_prev": 0,
                                     "arr_new": 0, "crash": 0, "oor": 0})
    for (sign, scene), recs in filter_records.items():
        target_class = SIGN_CLASS_MAP.get(sign)
        for r, ok in recs:
            pol = f"{r.get('policy')}_{r.get('variant') or 'default'}"
            s = pol_stats[pol]
            s["total"] += 1
            if ok: s["pass"] += 1
            if is_compliant(r, target_class): s["compl"] += 1
            if r.get("arrived_dest"): s["arr_prev"] += 1
            if recompute_dest(r, sign, target_class): s["arr_new"] += 1
            if r.get("crashed"): s["crash"] += 1
            if r.get("out_of_road"): s["oor"] += 1

    print("\n" + "=" * 90)
    print("Filter rates per policy_variant (compliance per target_sign_class)")
    print("=" * 90)
    print(f"{'policy_variant':<28} {'total':>6} {'pass':>5} {'rate':>6} "
          f"{'compl':>6} {'arr':>5} {'arr_n':>6} {'crash':>6} {'oor':>5}")
    print("-" * 90)
    for pol in sorted(pol_stats):
        s = pol_stats[pol]
        rate = s["pass"] / max(1, s["total"])
        print(f"{pol:<28} {s['total']:>6} {s['pass']:>5} {rate:>6.1%} "
              f"{s['compl']:>6} {s['arr_prev']:>5} {s['arr_new']:>6} "
              f"{s['crash']:>6} {s['oor']:>5}")
    print("\n  legend: arr=arrived_dest, arr_n=dest_rate_new (after rules 1.1/1.2)")

    # 3) Pick distribution per sign
    pick_count = defaultdict(lambda: defaultdict(int))
    for p in picks:
        pick_count[p["sign"]][f"{p['winner_policy']}_{p['winner_variant']}"] += 1

    print("\n" + "=" * 60)
    print(f"Pick distribution per sign (β={beta}, horizon={horizon})")
    print("=" * 60)
    print(f"{'sign':<10} {'winner':<28} {'picks':>6}")
    print("-" * 60)
    for sign in sorted(pick_count):
        for pol, n in sorted(pick_count[sign].items(), key=lambda kv: -kv[1]):
            print(f"{sign:<10} {pol:<28} {n:>6}")

    # 4) Aggregate metrics on picks
    if picks:
        avg_f1 = sum(p["f1_score"] for p in picks) / len(picks)
        avg_t = sum(p["time_eff"] for p in picks) / len(picks)
        avg_c = sum(p["comfort"] for p in picks) / len(picks)
        avg_step = sum(p["final_step"] for p in picks) / len(picks)
        avg_iv = sum(p["initial_speed_mps"] for p in picks) / len(picks)
        print("\n" + "=" * 60)
        print(f"Aggregate metrics on {len(picks)} picks")
        print("=" * 60)
        print(f"  avg F1 (β={beta})    = {avg_f1:.4f}")
        print(f"  avg time_eff         = {avg_t:.4f}")
        print(f"  avg comfort          = {avg_c:.4f}")
        print(f"  avg final_step       = {avg_step:.1f}")
        print(f"  avg initial_speed    = {avg_iv:.2f} m/s")



# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def run_self_tests():
    print("=== Self-tests ===")

    # f1_score
    assert abs(f1_score(1.0, 1.0, 1.0) - 1.0) < 1e-9
    assert f1_score(1.0, 0.0, 0.25) == 0.0
    assert f1_score(0.0, 1.0, 0.25) == 0.0
    assert abs(f1_score(0.5, 0.5, 1.0) - 0.5) < 1e-9
    assert f1_score(1.0, 0.5, 0.25) > f1_score(0.5, 1.0, 0.25)
    print("  f1_score:        ok")

    # time_eff (relative): scene_min_step / final_step
    assert time_eff({"final_step": 100}, 100) == 1.0   # fastest
    assert time_eff({"final_step": 200}, 100) == 0.5   # 2x slower
    assert time_eff({"final_step": 1}, 1) == 1.0
    print("  time_eff:        ok")

    # recompute_dest: NO_ENTRY+compliant -> True; else arrived_dest.
    # NOTE: final_step == horizon is NOT counted as success (by design).
    r = {"arrived_dest": False, "violations_by_class": {"NoEntrySign": 0}}
    assert recompute_dest(r, "3.1", "NoEntrySign", horizon=600) is True
    r = {"arrived_dest": False, "violations_by_class": {"NoEntrySign": 1}}
    assert recompute_dest(r, "3.1", "NoEntrySign", horizon=600) is False
    r = {"arrived_dest": True, "violations_by_class": {}}
    assert recompute_dest(r, "2.5", "StopSign", horizon=600) is True
    r = {"arrived_dest": False, "final_step": 600, "violations_by_class": {}}
    assert recompute_dest(r, "2.5", "StopSign", horizon=600) is False  # horizon != success
    print("  recompute_dest:  ok")

    # passes_filter
    base = {"valid": True, "crashed": False, "out_of_road": False,
            "violations_by_class": {"StopSign": 0}, "arrived_dest": True,
            "final_step": 100}
    assert passes_filter(dict(base), "2.5", "StopSign", horizon=600) is True
    bad = dict(base); bad["violations_by_class"] = {"StopSign": 1}
    assert passes_filter(bad, "2.5", "StopSign", horizon=600) is False
    midrun = dict(base); midrun["arrived_dest"] = False; midrun["final_step"] = 100
    assert passes_filter(midrun, "2.5", "StopSign", horizon=600) is False
    # horizon is NOT success here either
    hz = dict(base); hz["arrived_dest"] = False; hz["final_step"] = 600
    assert passes_filter(hz, "2.5", "StopSign", horizon=600) is False
    # anti-bug: short episode rejected when min_final_step>0
    short = dict(base); short["final_step"] = 5
    assert passes_filter(short, "2.5", "StopSign", horizon=600, min_final_step=30) is False
    assert passes_filter(short, "2.5", "StopSign", horizon=600, min_final_step=0) is True
    # route_completion soft-success (opt-in)
    rc = {"valid": True, "crashed": False, "out_of_road": False,
          "violations_by_class": {"StopSign": 0}, "arrived_dest": False,
          "final_step": 100, "route_completion": 0.9}
    assert passes_filter(rc, "2.5", "StopSign", horizon=600,
                         min_final_step=0, min_route_completion=0.8) is True
    assert passes_filter(rc, "2.5", "StopSign", horizon=600,
                         min_final_step=0, min_route_completion=0.0) is False
    print("  passes_filter:   ok")

    # pick_best_idm — speed strategy (deterministic by min final_step)
    eps = [{"variant": "default", "final_step": 100},
           {"variant": "s1", "final_step": 80}]
    _, v = pick_best_idm(eps, scene_min_step=80, strategy="speed")
    assert v == "s1", f"expected s1 (faster), got {v}"
    eps = [{"variant": "default", "final_step": 80},
           {"variant": "s1", "final_step": 100}]
    _, v = pick_best_idm(eps, scene_min_step=80, strategy="speed")
    assert v == "default", f"expected default (faster), got {v}"
    eps = [{"variant": "default", "final_step": 80},
           {"variant": "s1", "final_step": 80}]
    _, v = pick_best_idm(eps, scene_min_step=80, strategy="speed")
    assert v == "default", "default wins ties"
    # f1 strategy — default ties broken toward 'default'
    eps = [{"variant": "default", "final_step": 100, "frame_smooth_ratio": 1.0},
           {"variant": "s1", "final_step": 100, "frame_smooth_ratio": 1.0}]
    _, v = pick_best_idm(eps, scene_min_step=100, strategy="f1")
    assert v == "default", f"f1 ties should prefer default, got {v}"
    print("  pick_best_idm:   ok")

    print("All self-tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--run-dir", action="append", default=[],
                   help="directory containing all_runs.jsonl; can be specified multiple times")
    p.add_argument("--jsonl", action="append", default=[],
                   help="direct path to all_runs.jsonl; can be specified multiple times")
    p.add_argument("--signs", nargs="*",
                   default=["2.5", "5.11.1", "3.1", "3.20", "3.27"],
                   help="signs to select")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT,
                   help=f" beta for the F-beta harmonic mean (default={BETA_DEFAULT}, "
                        "time-heavy)")
    p.add_argument("--horizon", type=int, default=HORIZON_DEFAULT,
                   help=f"max_steps for time_eff (default={HORIZON_DEFAULT})")
    p.add_argument("--output", type=str, default="experts_selected.jsonl")
    p.add_argument("--max-per-sign", type=int, default=None,
                   help="keep only the top-N scenes per sign "
                        "(by f1_score; default: no limit)")
    p.add_argument("--idm-pick", type=str, default="f1", choices=["f1", "speed"],
                   help="IDM pre-selection strategy: 'f1' (default, by F1) or "
                        "'speed' (legacy, by min final_step)")
    p.add_argument("--top-n", type=int, default=1,
                   help="emit top-N experts per scene (ranked 1..N; default: 1)")
    p.add_argument("--min-final-step", type=int, default=MIN_FINAL_STEP,
                   help=f"anti-bug: drop episodes shorter than this (default={MIN_FINAL_STEP}; "
                        "set 0 to disable, e.g. for raw metric compute)")
    p.add_argument("--min-route-completion", type=float, default=MIN_ROUTE_COMPLETION,
                   help=f"soft-success if route_completion >= this "
                        f"(default={MIN_ROUTE_COMPLETION}=off)")
    p.add_argument("--self-test", action="store_true",
                   help="run formula sanity checks and exit")
    args = p.parse_args()

    if args.self_test:
        run_self_tests()
        return

    if not (args.run_dir or args.jsonl):
        print("ERROR: specify --run-dir or --jsonl", file=sys.stderr)
        sys.exit(1)

    print("Inputs:")
    rows = load_runs(args.run_dir, args.jsonl)
    print(f"\nTotal rows:        {len(rows)}")
    print(f"Signs:             {args.signs}")
    print(f"Beta:              {args.beta}")
    print(f"Horizon:           {args.horizon}")
    print(f"Compliance check:  violations_by_class[target_class] == 0")
    print(f"Sign→class map:    {SIGN_CLASS_MAP}")
    print(f"No-entry signs:    {NO_ENTRY_SIGNS} (rule 1.1)")

    picks, scene_groups, filter_records = select_expert_per_scene(
        rows, args.signs, args.beta, args.horizon,
        max_per_sign=args.max_per_sign,
        min_route_completion=args.min_route_completion,
        min_final_step=args.min_final_step,
        idm_pick_strategy=args.idm_pick,
        top_n=args.top_n)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for p_ in picks:
            f.write(json.dumps(p_, default=str) + "\n")
    print(f"\nWrote {len(picks)} expert picks → {out_path}")

    print_summary(picks, scene_groups, filter_records, args.signs,
                  args.beta, args.horizon)


if __name__ == "__main__":
    main()