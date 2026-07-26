#!/usr/bin/env python3
"""Oracle expert selection for no_entry (3.1 + 3.2) trajectories.

Thin adaptation of per_sign_bench/select_experts_coverage.py:
  * loads all_runs from no_entry collect_trajectories layout
    (<root>/*/3_1/all_runs.jsonl, <root>/*/3_2/all_runs.jsonl,
     and <root>/_merged/all_runs.jsonl)
  * catalog may be the one written by expert_replay_no_entry.py, or built
    on the fly from --manifest
  * default signs are 3.1 3.2, default horizon 1500

Usage:
  python select_experts_coverage.py \\
      --root output/trajectories_<ts> \\
      --catalog output/trajectories_<ts>/catalog.jsonl \\
      --signs 3.1 3.2 --horizon 1500 \\
      --out-dir output/trajectories_<ts>/experts

  # Smoke (no separate catalog file — build from manifest):
  python select_experts_coverage.py \\
      --root output/trajectories_smoke \\
      --manifest ../benchmark_output/combined/catalog_train80.jsonl \\
      --min-join-rate 0.0 \\
      --out-dir output/trajectories_smoke/experts
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIGN_DIR = SCRIPT_DIR.parent
PER_SIGN_DIR = SIGN_DIR.parent
for p in (str(PER_SIGN_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from select_experts import (  # noqa: E402
    SIGN_CLASS_MAP,
    MIN_FINAL_STEP,
    BETA_DEFAULT,
    normalize_sign,
    select_expert_per_scene,
)


def _row_uid(r: dict) -> str:
    seed = int(r.get("seed") or r.get("deterministic_seed") or 0)
    sid = r.get("scene_id") or f"scene_{seed}"
    return (f"{sid}_lane{int(r.get('spawn_lane_num', 0) or 0)}"
            f"_seed{seed}_v{int(r.get('var_idx', 0) or 0)}")


def load_rows(roots: list[str]) -> tuple[list[dict], dict]:
    files: list[Path] = []
    for root in roots:
        root_p = Path(root)
        # no_entry layout (3.1 + 3.2 under each policy)
        files += sorted(root_p.glob("*/3_1/all_runs.jsonl"))
        files += sorted(root_p.glob("*/3_2/all_runs.jsonl"))
        files += sorted(root_p.glob("*/*/all_runs.jsonl"))
        merged = root_p / "_merged" / "all_runs.jsonl"
        if merged.is_file():
            files.append(merged)
        # also allow a single all_runs.jsonl at root
        direct = root_p / "all_runs.jsonl"
        if direct.is_file():
            files.append(direct)
    # dedup paths
    files = sorted({f.resolve() for f in files})
    if not files:
        sys.exit(f"ERROR: no all_runs.jsonl found under {roots}")

    last: dict[tuple, dict] = {}
    n_rows = n_dup = n_bad = 0
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            n_rows += 1
            # Ensure scene_uid
            if not r.get("scene_uid"):
                r["scene_uid"] = _row_uid(r)
            key = (r.get("policy"), r.get("variant"),
                   normalize_sign(r.get("sign_code") or r.get("sign_slug")),
                   r.get("scene_uid"))
            if key in last:
                n_dup += 1
            last[key] = r
    stats = {
        "files": len(files),
        "rows": n_rows,
        "dups_removed": n_dup,
        "dups_cross_node": 0,
        "bad_json": n_bad,
        "rows_final": len(last),
    }
    return list(last.values()), stats


def catalog_from_manifest(path: str, sign_set: set) -> tuple[dict, dict, dict]:
    uid2map: dict[str, str] = {}
    uids_by_sign = collections.defaultdict(set)
    maps_by_sign = collections.defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("valid") is False:
                continue
            sign = normalize_sign(r.get("sign_code") or r.get("pdd_code") or "3.1")
            if sign not in sign_set:
                continue
            uid = r.get("scene_uid") or _row_uid(r)
            net = r.get("net_path") or ""
            uid2map[uid] = net
            uids_by_sign[sign].add(uid)
            if net:
                maps_by_sign[sign].add(net)
    return uid2map, uids_by_sign, maps_by_sign


def load_catalog(path: str, sign_set: set) -> tuple[dict, dict, dict]:
    uid2map: dict[str, str] = {}
    uids_by_sign = collections.defaultdict(set)
    maps_by_sign = collections.defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("valid") is False:
                continue
            sign = normalize_sign(r.get("sign_code") or "3.1")
            if sign not in sign_set:
                continue
            uid = r.get("scene_uid") or _row_uid(r)
            net = r.get("net_path") or ""
            uid2map[uid] = net
            uids_by_sign[sign].add(uid)
            if net:
                maps_by_sign[sign].add(net)
    return uid2map, uids_by_sign, maps_by_sign


def _ref_passes(r: dict, tclass: str, min_final_step: int) -> bool:
    from select_experts import passes_filter
    return passes_filter(
        r, normalize_sign(r.get("sign_code") or "3.1"), tclass,
        horizon=10**9, min_final_step=min_final_step,
    )


def reference_winner(group, tclass, beta, min_final_step):
    """Independent local argmax over F1 (verification)."""
    from select_experts import (
        f1_score, time_eff, comfort, passes_filter, IDM_FAMILY_POLICIES,
        NON_IDM_POLICIES, pick_best_idm,
    )
    sign = normalize_sign(group[0].get("sign_code") or "3.1")
    passing = [r for r in group
               if passes_filter(r, sign, tclass, 10**9, min_final_step)]
    if not passing:
        return None
    idm_eps = [r for r in passing if r.get("policy") in IDM_FAMILY_POLICIES]
    scene_min = max(1, min(int(r.get("final_step") or 10**9) for r in passing))
    best_idm, _ = pick_best_idm(idm_eps, scene_min, beta=beta, strategy="f1")
    cands = []
    if best_idm is not None:
        cands.append(best_idm)
    for r in passing:
        if r.get("policy") in NON_IDM_POLICIES:
            cands.append(r)
    if not cands:
        return None
    scene_min = max(1, min(int(r.get("final_step") or 10**9) for r in cands))
    scored = []
    for r in cands:
        t = time_eff(r, scene_min)
        c = comfort(r)
        scored.append((f1_score(t, c, beta), r))
    scored.sort(key=lambda x: -x[0])
    top_f1 = scored[0][0]
    top = [r for s, r in scored if abs(s - top_f1) < 1e-9]
    return top_f1, top


def main() -> None:
    ap = argparse.ArgumentParser(
        description="No-entry (3.1+3.2) oracle selection (top-1 / top-2 / map)"
    )
    ap.add_argument("--root", action="append", default=[],
                    help="collection OUT_BASE (repeatable)")
    ap.add_argument("--catalog", default=None,
                    help="catalog.jsonl from expert_replay_no_entry")
    ap.add_argument("--manifest", default=None,
                    help="If no --catalog, build catalog from this manifest")
    ap.add_argument("--signs", nargs="*", default=["3.1", "3.2"])
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--min-final-step", type=int, default=MIN_FINAL_STEP)
    ap.add_argument("--min-join-rate", type=float, default=0.90,
                    help="Min catalog join rate (set 0 for smoke)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check-files", action="store_true")
    ap.add_argument("--path-map", default=None,
                    help="PREFIX_OLD=PREFIX_NEW for pkl/sidecar paths")
    args = ap.parse_args()

    if not args.root:
        sys.exit("ERROR: provide --root")
    if not args.catalog and not args.manifest:
        # try default catalog under first root
        cand = Path(args.root[0]) / "catalog.jsonl"
        if cand.is_file():
            args.catalog = str(cand)
        else:
            sys.exit("ERROR: provide --catalog or --manifest")

    sign_set = {normalize_sign(s) for s in args.signs}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, lstats = load_rows(args.root)
    print(f"all_runs files: {lstats['files']}, rows: {lstats['rows']}, "
          f"dups removed: {lstats['dups_removed']}, "
          f"bad json: {lstats['bad_json']}, final: {lstats['rows_final']}")

    if args.catalog:
        uid2map, uids_by_sign, maps_by_sign = load_catalog(args.catalog, sign_set)
        print(f"catalog: {args.catalog}")
    else:
        uid2map, uids_by_sign, maps_by_sign = catalog_from_manifest(
            args.manifest, sign_set
        )
        print(f"catalog-from-manifest: {args.manifest}")

    run_uids = {r.get("scene_uid") for r in rows
                if normalize_sign(r.get("sign_code")
                                  or r.get("sign_slug")) in sign_set}
    run_uids.discard(None)
    hit = sum(1 for u in run_uids if u in uid2map)
    rate = hit / max(1, len(run_uids))
    print(f"catalog join: {hit}/{len(run_uids)} uids ({rate:.1%})")
    if rate < args.min_join_rate:
        sys.exit(
            f"ERROR: join rate {rate:.1%} < --min-join-rate {args.min_join_rate}. "
            "Pass the catalog written during collection, or --min-join-rate 0."
        )

    picks, scene_groups, _ = select_expert_per_scene(
        rows, sorted(sign_set), beta=args.beta, horizon=args.horizon,
        min_final_step=args.min_final_step, top_n=2)
    for p in picks:
        p["net_path"] = uid2map.get(p["scene_key"])

    rank1 = [p for p in picks if p["rank"] == 1]
    by_map = collections.defaultdict(list)
    n_unmapped = 0
    for p in rank1:
        if p["net_path"] is None:
            n_unmapped += 1
            continue
        by_map[(p["sign"], p["net_path"])].append(p)
    map_picks = [max(ps, key=lambda p: (p["f1_score"], p["scene_key"]))
                 for ps in by_map.values()]
    if n_unmapped:
        print(f"[warn] rank-1 picks without a catalog map: {n_unmapped}")

    print("\n=== Verification ===")
    n_win_mismatch = n_invariant = 0
    for p in rank1:
        sign = p["sign"]
        tclass = SIGN_CLASS_MAP[sign]
        group = scene_groups[(sign, p["scene_key"])]
        ref = reference_winner(group, tclass, args.beta, args.min_final_step)
        if ref is None:
            n_win_mismatch += 1
            print(f"  [MISMATCH] {sign} {p['scene_key']}: no reference candidates")
            continue
        ref_f1, ref_top = ref
        ok_f1 = abs(ref_f1 - p["f1_score"]) < 1e-6
        ok_who = any(r.get("policy") == p["winner_policy"]
                     and r.get("variant") == p["winner_variant"]
                     for r in ref_top)
        if not (ok_f1 and ok_who):
            n_win_mismatch += 1
            print(f"  [MISMATCH] {sign} {p['scene_key']}: "
                  f"pick={p['winner_policy']}_{p['winner_variant']} "
                  f"f1={p['f1_score']:.6f} vs ref f1={ref_f1:.6f}")
        win_rows = [r for r in group
                    if r.get("policy") == p["winner_policy"]
                    and r.get("variant") == p["winner_variant"]]
        if not win_rows or not _ref_passes(win_rows[-1], tclass,
                                           args.min_final_step):
            n_invariant += 1
            print(f"  [INVARIANT] {sign} {p['scene_key']}")
    print(f"winner recomputation: {len(rank1) - n_win_mismatch}/{len(rank1)} "
          f"matched; invariant failures: {n_invariant}")

    n_missing_files = 0
    if args.check_files:
        old = new = None
        if args.path_map and "=" in args.path_map:
            old, new = args.path_map.split("=", 1)
        for p in rank1:
            for fkey in ("pkl_path", "sidecar_path", "gif_path"):
                fp = p.get(fkey)
                if not fp:
                    continue
                if old:
                    fp = fp.replace(old, new)
                if not Path(fp).exists():
                    n_missing_files += 1
                    print(f"  [NO FILE] {fkey}: {fp}")
        print(f"sidecar/gif files missing: {n_missing_files}")

    print(f"\n{'sign':<7}{'cat uids':>9}{'covered':>9}{'%':>6}"
          f"{'cat maps':>9}{'covered':>9}{'%':>6}")
    uncovered_uids, uncovered_maps = {}, {}
    for sign in sorted(sign_set):
        cat_u = uids_by_sign.get(sign, set())
        cat_m = maps_by_sign.get(sign, set())
        got_u = {p["scene_key"] for p in rank1 if p["sign"] == sign}
        got_m = {p["net_path"] for p in map_picks if p["sign"] == sign}
        uncovered_uids[sign] = sorted(cat_u - got_u)
        uncovered_maps[sign] = sorted(cat_m - got_m)
        print(f"{sign:<7}{len(cat_u):>9}{len(cat_u & got_u):>9}"
              f"{len(cat_u & got_u) / max(1, len(cat_u)):>6.0%}"
              f"{len(cat_m):>9}{len(cat_m & got_m):>9}"
              f"{len(cat_m & got_m) / max(1, len(cat_m)):>6.0%}")

    def metrics_table(label, sel):
        print(f"\n=== Selection metrics: {label} ({len(sel)} trajectories) ===")
        dist = collections.Counter(
            (p["sign"], f"{p['winner_policy']}_{p['winner_variant']}")
            for p in sel)
        print("winners:")
        for (sign, pol), n in sorted(dist.items()):
            print(f"  {sign:<7}{pol:<38}{n:>6}")
        if sel:
            af1 = sum(p["f1_score"] for p in sel) / len(sel)
            at = sum(p["time_eff"] for p in sel) / len(sel)
            ac = sum(p["comfort"] for p in sel) / len(sel)
            print(f"  avg F1={af1:.4f}  time_eff={at:.3f}  comfort={ac:.3f}")

    metrics_table("top-1 (best per scene_uid)", rank1)
    metrics_table("top-2 (up to 2 experts per scene_uid)", picks)
    metrics_table("map (best per map)", map_picks)

    def dump(name, sel):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for x in sel:
                f.write(json.dumps(x, default=str) + "\n")
        print(f"  {p}  ({len(sel)})")

    print("\noutput files:")
    dump("experts_scene_uid_top1.jsonl", rank1)
    dump("experts_scene_uid_top2.jsonl", picks)
    dump("experts_map.jsonl", map_picks)
    dump("all_runs_dedup.jsonl", rows)
    json.dump({
        "load": lstats, "join_rate": rate,
        "n_top1": len(rank1), "n_top2": len(picks), "n_map": len(map_picks),
        "winner_mismatches": n_win_mismatch,
        "invariant_failures": n_invariant,
        "missing_files": n_missing_files,
        "uncovered_uids": uncovered_uids,
        "uncovered_maps": uncovered_maps,
    }, open(out_dir / "coverage_report.json", "w", encoding="utf-8"),
       indent=1, ensure_ascii=False)
    print(f"  {out_dir / 'coverage_report.json'}")

    if n_win_mismatch or n_invariant:
        sys.exit("VERIFICATION FAILED: see MISMATCH/INVARIANT above")
    print("\nVerification passed.")


if __name__ == "__main__":
    main()
