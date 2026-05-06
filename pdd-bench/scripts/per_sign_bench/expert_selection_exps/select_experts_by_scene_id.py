#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from select_experts import (    # noqa: E402
    SIGN_CLASS_MAP, NO_ENTRY_SIGNS, HORIZON_DEFAULT, BETA_DEFAULT,
    NON_IDM_POLICIES, MIN_ROUTE_COMPLETION, MIN_FINAL_STEP,
    normalize_sign, load_runs,
    passes_filter, time_eff, comfort, f1_score, pick_best_idm,
    print_summary, run_self_tests,
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def select_expert_per_scene_id(rows, signs, beta=BETA_DEFAULT,
                                  horizon=HORIZON_DEFAULT,
                                  max_per_sign=None,
                                  min_route_completion=MIN_ROUTE_COMPLETION,
                                  min_final_step=MIN_FINAL_STEP,
                                  idm_pick_strategy="f1",
                                  time_eff_formula="min_over_final",
                                  top_n=1,
                                  per_agent_best=True):
    """group by (sign, scene_id)
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
        scene_id = r.get("scene_id")
        if not scene_id:
            continue
        target_class = SIGN_CLASS_MAP.get(sign)
        if target_class is None:
            continue
        scene_groups[(sign, scene_id)].append(r)
        ok = passes_filter(r, sign, target_class, horizon,
                            min_route_completion, min_final_step)
        filter_records[(sign, scene_id)].append((r, ok))

    picks = []
    for (sign, scene_id), eps in scene_groups.items():
        target_class = SIGN_CLASS_MAP[sign]
        passing = [r for r in eps if passes_filter(r, sign, target_class, horizon,
                                                     min_route_completion,
                                                     min_final_step)]
        if not passing:
            continue

        final_steps = [int(r.get("final_step") or 10**9) for r in passing]
        scene_min_step = max(1, min(final_steps))
        scene_max_step = max(1, max(final_steps))
        if time_eff_formula == "absolute":
            scene_max_step = horizon

        idm_eps = [r for r in passing if r.get("policy") == "comprehensive"]

        if per_agent_best:
            idm_candidates_for_pick = []
            if idm_eps:
                by_variant = defaultdict(list)
                for r in idm_eps:
                    by_variant[r.get("variant")].append(r)
                for var_eps in by_variant.values():
                    scored_var = []
                    for r in var_eps:
                        t = time_eff(r, scene_min_step, scene_max_step, time_eff_formula)
                        c = comfort(r)
                        scored_var.append((f1_score(t, c, beta), r))
                    scored_var.sort(key=lambda x: -x[0])
                    idm_candidates_for_pick.append(scored_var[0][1])
            best_idm, best_idm_variant = pick_best_idm(
                idm_candidates_for_pick, scene_min_step, scene_max_step, beta,
                strategy=idm_pick_strategy,
                time_eff_formula=time_eff_formula)

            # NON-COMP per-policy best
            candidates = []
            if best_idm is not None:
                candidates.append(best_idm)
            for pol in NON_IDM_POLICIES:
                pol_eps = [r for r in passing if r.get("policy") == pol]
                if not pol_eps:
                    continue
                scored_pol = []
                for r in pol_eps:
                    t = time_eff(r, scene_min_step, scene_max_step, time_eff_formula)
                    c = comfort(r)
                    scored_pol.append((f1_score(t, c, beta), r))
                scored_pol.sort(key=lambda x: -x[0])
                candidates.append(scored_pol[0][1])
        else:

            best_idm, best_idm_variant = pick_best_idm(
                idm_eps, scene_min_step, scene_max_step, beta,
                strategy=idm_pick_strategy,
                time_eff_formula=time_eff_formula)

            candidates = []
            if best_idm is not None:
                candidates.append(best_idm)
            for r in passing:
                if r.get("policy") in NON_IDM_POLICIES:
                    candidates.append(r)

        if not candidates:
            continue

        scored = []
        for r in candidates:
            t = time_eff(r, scene_min_step, scene_max_step, time_eff_formula)
            c = comfort(r)
            scored.append((f1_score(t, c, beta), r, t, c))
        scored.sort(key=lambda x: -x[0])

        # top-N
        n_emit = min(top_n, len(scored))
        for rank in range(1, n_emit + 1):
            s_score, s_winner, s_t, s_c = scored[rank - 1]
            picks.append({
                "sign": sign,
                "scene_id": scene_id,
                "scene_uid": s_winner.get("scene_uid"),   
                "rank": rank,
                "n_emitted_per_scene_id": n_emit,
                "winner_policy": s_winner.get("policy"),
                "winner_variant": s_winner.get("variant"),
                "best_idm_variant": best_idm_variant,
                "final_step": int(s_winner.get("final_step") or 0),
                "route_completion": float(s_winner.get("route_completion") or 0.0),
                "time_eff": round(s_t, 6),
                "comfort": round(s_c, 6),
                "f1_score": round(s_score, 6),
                "beta": beta,
                "horizon": horizon,
                "scene_min_step": scene_min_step,
                "scene_max_step": scene_max_step,
                "passing_candidates_n": len(candidates),
                "all_passing_n": len(passing),
                "all_attempts_n": len(eps),
                "scene_uids_n": len({r.get("scene_uid") for r in passing
                                       if r.get("scene_uid")}),
                "pkl_path": s_winner.get("pkl_path"),
                "gif_path": s_winner.get("gif_path"),
                "sidecar_path": s_winner.get("sidecar_path"),
                "initial_speed_mps": float(s_winner.get("initial_speed_mps") or 0.0),
                "all_candidates": [
                    {
                        "policy": r.get("policy"),
                        "variant": r.get("variant"),
                        "scene_uid": r.get("scene_uid"),
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
            by_sign[p["sign"]][p["scene_id"]].append(p)
        limited = []
        for sign, scene_ids in by_sign.items():
            scene_ids_sorted = sorted(
                scene_ids.keys(),
                key=lambda k: (
                    -max(p["f1_score"] for p in scene_ids[k] if p["rank"] == 1),
                    k,
                ),
            )
            for sid in scene_ids_sorted[:max_per_sign]:
                limited.extend(sorted(scene_ids[sid], key=lambda p: p["rank"]))
        picks = limited

    return picks, scene_groups, filter_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", action="append", default=[],
                   help="directory containing all_runs.jsonl")
    p.add_argument("--jsonl", action="append", default=[],
                   help="direct path to a .jsonl file with trajectories")
    p.add_argument("--signs", nargs="*",
                   help="list of target signs, e.g. 2.5 5.11.1 3.1")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT,
                   help=f"F-beta parameter for the F1 score (default={BETA_DEFAULT}; "
                        f">1 — comfort-heavy, <1 — time-heavy)")
    p.add_argument("--horizon", type=int, default=HORIZON_DEFAULT,
                   help=f"max_steps for time_eff (default={HORIZON_DEFAULT})")
    p.add_argument("--output", type=str, default="experts_by_scene_id.jsonl")
    p.add_argument("--max-per-sign", type=int, default=None,
                   help="keep only the top-N scene_ids per sign (default: no limit)")
    p.add_argument("--top-n", type=int, default=1, choices=[1, 2, 3],
                   help="number of top trajectories to emit per scene_id "
                        "(default 1=winner only)")

    p.add_argument("--min-route-completion", type=float, default=MIN_ROUTE_COMPLETION,
                   help=f"minimum route_completion (default={MIN_ROUTE_COMPLETION}; "
                        f"0 = LEGACY)")
    p.add_argument("--min-final-step", type=int, default=MIN_FINAL_STEP,
                   help=f"minimum final_step (default={MIN_FINAL_STEP}; 0 = LEGACY)")
    p.add_argument("--idm-pick", choices=["f1", "speed"], default="f1",
                   help="pick_best_idm strategy: 'f1' (NEW) or 'speed' (LEGACY)")
    p.add_argument("--time-eff-formula",
                   choices=["min_over_final", "inv_final_max", "linear", "absolute"],
                   default="min_over_final",
                   help="time_eff formula; see select_experts.time_eff")
    p.add_argument("--per-agent-best", dest="per_agent_best",
                   action="store_true", default=True,
                   help="default: two-stage selection: for each comp variant, "
                        "first select the best-by-F1 trajectory across seeds, "
                        "then apply pick_best_idm")
    p.add_argument("--no-per-agent-best", dest="per_agent_best",
                   action="store_false",
                   help="disable two-stage selection: pick_best_idm operates directly "
                        "on variant×seed candidates")
    p.add_argument("--legacy", action="store_true",
                   help="shortcut for LEGACY mode")
    p.add_argument("--self-test", action="store_true",
                   help="run sanity checks and exit")
    args = p.parse_args()

    if args.self_test:
        run_self_tests()
        return

    if args.legacy:
        args.min_route_completion = 0.0
        args.min_final_step = 0
        args.idm_pick = "speed"

    if not (args.run_dir or args.jsonl):
        print("ERROR: specify --run-dir or --jsonl", file=sys.stderr)
        sys.exit(1)

    print("Inputs:")
    rows = load_runs(args.run_dir, args.jsonl)
    print(f"\nTotal rows:           {len(rows)}")
    print(f"Signs:                {args.signs}")
    print(f"Aggregation:          by unique scene_id, not scene_uid")
    print(f"Beta:                 {args.beta}")
    print(f"Horizon:              {args.horizon}")
    print(f"top_n:                {args.top_n}")
    print(f"min_route_completion: {args.min_route_completion}  "
          f"{'(LEGACY/disabled)' if args.min_route_completion == 0 else '(NEW)'}")
    print(f"min_final_step:       {args.min_final_step}  "
          f"{'(LEGACY/disabled)' if args.min_final_step == 0 else '(NEW)'}")
    print(f"idm_pick_strategy:    {args.idm_pick}  "
          f"{'(NEW)' if args.idm_pick == 'f1' else '(LEGACY)'}")
    print(f"time_eff_formula:     {args.time_eff_formula}")
    print(f"per_agent_best:       {args.per_agent_best}  "
          f"({'two-stage selection for comp' if args.per_agent_best else 'single-step selection (LEGACY-like)'})")
    print(f"No-entry signs:       {NO_ENTRY_SIGNS} (rule 1.1)")

    picks, scene_groups, filter_records = select_expert_per_scene_id(
        rows, args.signs, args.beta, args.horizon,
        max_per_sign=args.max_per_sign,
        min_route_completion=args.min_route_completion,
        min_final_step=args.min_final_step,
        idm_pick_strategy=args.idm_pick,
        time_eff_formula=args.time_eff_formula,
        top_n=args.top_n,
        per_agent_best=args.per_agent_best)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for p_ in picks:
            f.write(json.dumps(p_, default=str) + "\n")
    print(f"\nWrote {len(picks)} expert picks → {out_path}")
    print(f"  Unique scene_ids with >=1 winner: "
          f"{len({(p['sign'], p['scene_id']) for p in picks})}")

    print(f"\n=== Per-sign summary ===")
    by_sign = defaultdict(list)
    for pi in picks:
        by_sign[pi["sign"]].append(pi)
    print(f"  {'sign':<10} {'scene_ids':>10} {'picks':>6} {'avg_f1':>8} "
          f"{'top winner policy distribution':<40}")
    for sign in sorted(by_sign):
        items = by_sign[sign]
        unique_sids = len({p["scene_id"] for p in items})
        avg_f1 = sum(p["f1_score"] for p in items) / max(1, len(items))
        rank1 = [p for p in items if p["rank"] == 1]
        from collections import Counter
        pol_dist = Counter(p["winner_policy"] for p in rank1)
        pol_str = ", ".join(f"{p}={n}" for p, n in pol_dist.most_common())
        print(f"  {sign:<10} {unique_sids:>10} {len(items):>6} {avg_f1:>7.3f}  {pol_str}")


if __name__ == "__main__":
    main()