#!/usr/bin/env python3
"""Expert selection over collected trajectory trees (full scene_uid coverage)
with built-in verification.

Strategy  top-1 : best expert per unique (sign, scene_uid).
Strategy  top-2 : top-2 experts per scene_uid (rank 1 and rank 2, when the
                  scene has >= 2 filter-passing candidates).
Strategy  map   : top-1 reduced to the best pick per (sign, map);
                  map = net_path from the catalog (the fv-split map identity).

For every strategy the script prints metrics of the selected set: uid
coverage, avg F1 / time_eff / comfort / final_step / v0, and the winner
distribution per policy.

Built-in verification:
  1. dedup of repeated all_runs rows (resume after deleting broken pkls
     re-records the episode) — the LAST row wins; dup count is reported;
  2. join of run scene_uids against the catalog using the exact
     run_benchmark uid formula (>= 90% hit rate required, else exit);
  3. independent recomputation of every scene winner (local argmax over a
     local F1 implementation) compared against the pick;
  4. per-pick invariants: valid, event-compliant on the target class,
     no crash/OOR, arrived_dest, final_step >= min;
  5. coverage: how many catalog uids/maps received an expert (uncovered
     lists are saved to out-dir);
  6. --check-files: pkl_path/sidecar_path existence (with --path-map
     PREFIX_OLD=PREFIX_NEW for a foreign mount prefix).

Usage (on the server, from per_sign_bench):
  python3 select_experts_coverage.py \\
      --root $SM/traj_fv_train80_nodeA_20260719 \\
      --root $SM/traj_fv_train80_nodeB_20260719 \\
      --catalog $SM/run_v61_a6/catalog_fv_train80.jsonl \\
      --signs 3.24 4.6 5.21 5.31 --horizon 1500 \\
      --out-dir $SM/experts_fv_train80
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from select_experts import (  # noqa: E402
    SIGN_CLASS_MAP, IDM_FAMILY_POLICIES, NON_IDM_POLICIES,
    MIN_FINAL_STEP, BETA_DEFAULT,
    normalize_sign, select_expert_per_scene,
)


def _row_uid(r: dict) -> str:
    """Catalog scene_uid — the exact run_benchmark/expert_replay formula."""
    seed = int(r.get("seed") or r.get("deterministic_seed") or 0)
    sid = r.get("scene_id") or f"scene_{seed}"
    return (f"{sid}_lane{int(r.get('spawn_lane_num', 0) or 0)}"
            f"_seed{seed}_v{int(r.get('var_idx', 0) or 0)}")


# ---------------------------------------------------------------------------
# Loading + dedup
# ---------------------------------------------------------------------------
def load_rows(roots: list[str]) -> tuple[list[dict], dict]:
    files = []
    for root in roots:
        files += sorted(Path(root).glob("*/*/all_runs.jsonl"))
    if not files:
        sys.exit(f"ERROR: no */*/all_runs.jsonl found under {roots}")

    last: dict[tuple, dict] = {}
    src_root: dict[tuple, str] = {}
    n_rows = n_dup = n_dup_cross = n_bad = 0
    for f in files:
        root = str(f.parents[2])
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
            key = (r.get("policy"), r.get("variant"),
                   normalize_sign(r.get("sign_code") or r.get("sign_slug")),
                   r.get("scene_uid"))
            if key in last:
                n_dup += 1
                if src_root.get(key) != root:
                    n_dup_cross += 1
            last[key] = r          # the last row wins (resume re-records)
            src_root[key] = root
    stats = {"files": len(files), "rows": n_rows, "dups_removed": n_dup,
             "dups_cross_node": n_dup_cross, "bad_json": n_bad,
             "rows_final": len(last)}
    return list(last.values()), stats


def load_catalog(path: str, sign_set: set) -> tuple[dict, dict, dict]:
    """-> uid2map, uids_by_sign, maps_by_sign (only rows of requested signs;
    catalog valid:false rows are skipped — they never entered the collection
    manifests)."""
    uid2map: dict[str, str] = {}
    uids_by_sign = collections.defaultdict(set)
    maps_by_sign = collections.defaultdict(set)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("valid") is False:
            continue
        sign = normalize_sign(r.get("sign_code"))
        if sign not in sign_set:
            continue
        uid = _row_uid(r)
        uid2map[uid] = r.get("net_path")
        uids_by_sign[sign].add(uid)
        maps_by_sign[sign].add(r.get("net_path"))
    return uid2map, dict(uids_by_sign), dict(maps_by_sign)


# ---------------------------------------------------------------------------
# Independent winner recomputation (verification step 3)
# ---------------------------------------------------------------------------
def _ref_f1(t: float, c: float, beta: float) -> float:
    if t <= 0.0 or c <= 0.0:
        return 0.0
    b2 = beta * beta
    return (b2 + 1.0) * t * c / (b2 * t + c)


def _ref_passes(r: dict, target_class: str, min_fs: int,
                min_in_zone: int = 0) -> bool:
    if not r.get("valid") or r.get("crashed") or r.get("out_of_road"):
        return False
    vbc = r.get("violations_by_class_event")
    if vbc is None:
        vbc = r.get("violations_by_class") or {}
    if int(vbc.get(target_class, 0) or 0) != 0:
        return False
    if int(r.get("final_step") or 0) < min_fs:
        return False
    if min_in_zone > 0:
        inz = r.get("in_zone_total_steps")
        if inz is not None and int(inz or 0) < min_in_zone:
            return False
    return bool(r.get("arrived_dest"))


def reference_winner(group: list[dict], target_class: str,
                     beta: float, min_fs: int, min_in_zone: int = 0):
    """Own argmax: (f1, rows tied at best f1) of the best candidate, or None."""
    passing = [r for r in group
               if _ref_passes(r, target_class, min_fs, min_in_zone)]
    if not passing:
        return None
    idm = [r for r in passing if r.get("policy") in IDM_FAMILY_POLICIES]
    best_idm = None
    if idm:
        pool_min = max(1, min(int(r.get("final_step") or 10**9)
                              for r in passing))
        best_idm = max(
            idm,
            key=lambda r: (
                _ref_f1(pool_min / max(1, int(r.get("final_step") or 1)),
                        float(r.get("frame_smooth_ratio") or 0.0), beta),
                1 if r.get("variant") == "default" else 0,
                -int(r.get("final_step") or 10**9),
            ))
    cands = ([best_idm] if best_idm is not None else []) + \
            [r for r in passing if r.get("policy") in NON_IDM_POLICIES]
    if not cands:
        return None
    smin = max(1, min(int(r.get("final_step") or 10**9) for r in cands))
    scored = [(_ref_f1(smin / max(1, int(r.get("final_step") or 1)),
                       float(r.get("frame_smooth_ratio") or 0.0), beta), r)
              for r in cands]
    best_f1 = max(s for s, _ in scored)
    top = [r for s, r in scored if abs(s - best_f1) < 1e-12]
    return best_f1, top


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", required=True,
                    help="collection OUT_BASE (repeatable: nodeA, nodeB)")
    ap.add_argument("--catalog", required=True,
                    help="catalog_fv_train80.jsonl — the map (net_path) source")
    ap.add_argument("--signs", nargs="+", default=["3.24", "4.6", "5.21", "5.31"])
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--min-final-step", type=int, default=MIN_FINAL_STEP)
    ap.add_argument("--min-in-zone-steps", type=int, default=0,
                    help="drop candidates with in_zone_total_steps below this "
                         "(anti-vacuous compliance; default 0 = off)")
    ap.add_argument("--geometry-audit", default=None,
                    help="audit CSV from check_spawn_sign_geometry --all-runs; "
                         "candidates whose audit verdict fails the geometry "
                         "rule are dropped BEFORE selection, so each scene "
                         "keeps its best geometrically-correct trajectory")
    ap.add_argument("--geometry-max-dist", type=float, default=8.0,
                    help="max closest-approach to the sign point, m (with "
                         "--geometry-audit; in-zone > 0 is always required)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check-files", action="store_true",
                    help="check pkl/sidecar existence")
    ap.add_argument("--path-map", default=None,
                    help="OLD=NEW path prefix replacement for --check-files")
    args = ap.parse_args()

    sign_set = {normalize_sign(s) for s in args.signs}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. run rows (deduped) ---
    rows, lstats = load_rows(args.root)
    print(f"all_runs files: {lstats['files']}, rows: {lstats['rows']}, "
          f"dups removed: {lstats['dups_removed']} "
          f"(cross-node: {lstats['dups_cross_node']}), "
          f"bad json: {lstats['bad_json']}, final: {lstats['rows_final']}")

    # --- 2. catalog: uid -> map ---
    uid2map, uids_by_sign, maps_by_sign = load_catalog(args.catalog, sign_set)
    run_uids = {r.get("scene_uid") for r in rows
                if normalize_sign(r.get("sign_code")
                                  or r.get("sign_slug")) in sign_set}
    run_uids.discard(None)
    hit = sum(1 for u in run_uids if u in uid2map)
    rate = hit / max(1, len(run_uids))
    print(f"catalog join: {hit}/{len(run_uids)} uids ({rate:.1%})")
    if rate < 0.90:
        sys.exit("ERROR: run scene_uids do not match the catalog — "
                 "make sure this is the same catalog the collection "
                 "manifests were built from")

    # --- 2b. geometry audit: keep only candidates that truly passed the sign ---
    if args.geometry_audit:
        import csv as _csv
        good = set()
        audited = 0
        for a in _csv.DictReader(open(args.geometry_audit, encoding="utf-8")):
            audited += 1
            try:
                inz = int(float(a.get("in_zone_steps") or 0))
                md = float(a.get("min_dist_m") or 1e9)
            except ValueError:
                continue
            if a.get("verdict", "").startswith("ERROR"):
                continue
            if inz > 0 and md <= args.geometry_max_dist:
                good.add((a["scene_uid"], a["policy"], a["variant"]))
        n_before = len(rows)
        rows = [r for r in rows
                if (r.get("scene_uid"), r.get("policy"), r.get("variant"))
                in good]
        print(f"geometry audit: {audited} audited episodes, "
              f"{len(good)} pass (in_zone>0 & min_dist<="
              f"{args.geometry_max_dist}m); candidate rows "
              f"{n_before} -> {len(rows)}")
        if not rows:
            sys.exit("ERROR: geometry audit filtered out every candidate — "
                     "check that the CSV matches this collection")

    # --- 3. selection: top-2 per scene_uid (rank 1 = the top-1 strategy) ---
    picks, scene_groups, _ = select_expert_per_scene(
        rows, sorted(sign_set), beta=args.beta, horizon=args.horizon,
        min_final_step=args.min_final_step, top_n=2,
        min_in_zone_steps=args.min_in_zone_steps)
    for p in picks:
        p["net_path"] = uid2map.get(p["scene_key"])

    # --- 4. map strategy: best rank-1 pick per (sign, map) ---
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

    # --- 5. verification ---
    print("\n=== Verification ===")
    n_win_mismatch = n_invariant = 0
    for p in rank1:
        sign = p["sign"]
        tclass = SIGN_CLASS_MAP[sign]
        group = scene_groups[(sign, p["scene_key"])]
        ref = reference_winner(group, tclass, args.beta, args.min_final_step,
                               args.min_in_zone_steps)
        if ref is None:
            n_win_mismatch += 1
            print(f"  [MISMATCH] {sign} {p['scene_key']}: pick exists but "
                  f"the reference found no candidates")
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
                  f"f1={p['f1_score']:.6f} vs ref f1={ref_f1:.6f} "
                  f"top={[(r.get('policy'), r.get('variant')) for r in ref_top]}")
        win_rows = [r for r in group
                    if r.get("policy") == p["winner_policy"]
                    and r.get("variant") == p["winner_variant"]]
        if not win_rows or not _ref_passes(win_rows[-1], tclass,
                                           args.min_final_step,
                                           args.min_in_zone_steps):
            n_invariant += 1
            print(f"  [INVARIANT] {sign} {p['scene_key']}: the winner row "
                  f"does not pass the filter")
    print(f"winner recomputation: {len(rank1) - n_win_mismatch}/{len(rank1)} "
          f"matched; invariant failures: {n_invariant}")

    n_missing_files = 0
    if args.check_files:
        old, new = (args.path_map.split("=", 1) if args.path_map
                    else (None, None))
        for p in rank1:
            for fkey in ("pkl_path", "sidecar_path"):
                fp = p.get(fkey)
                if not fp:
                    continue
                if old:
                    fp = fp.replace(old, new)
                if not Path(fp).exists():
                    n_missing_files += 1
                    print(f"  [NO FILE] {fkey}: {fp}")
        print(f"pkl/sidecar files: missing {n_missing_files}")

    # --- 6. coverage + report ---
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

    # --- 7. per-strategy metrics ---
    def metrics_table(label, sel):
        print(f"\n=== Selection metrics: {label} ({len(sel)} trajectories) ===")
        print(f"{'sign':<7}{'traj':>6}{'scenes':>7}{'cat uids':>9}{'cov':>6}"
              f"{'avg F1':>8}{'avg t':>7}{'avg c':>7}{'avg fs':>7}{'avg v0':>7}")
        def line(sign, ps, cat_n):
            scenes = {p["scene_key"] for p in ps}
            if ps:
                af1 = sum(p["f1_score"] for p in ps) / len(ps)
                at = sum(p["time_eff"] for p in ps) / len(ps)
                ac = sum(p["comfort"] for p in ps) / len(ps)
                afs = sum(p["final_step"] for p in ps) / len(ps)
                av0 = sum(p["initial_speed_mps"] for p in ps) / len(ps)
            else:
                af1 = at = ac = afs = av0 = 0.0
            print(f"{sign:<7}{len(ps):>6}{len(scenes):>7}{cat_n:>9}"
                  f"{len(scenes) / max(1, cat_n):>6.0%}"
                  f"{af1:>8.4f}{at:>7.3f}{ac:>7.3f}{afs:>7.0f}{av0:>7.2f}")
        for sign in sorted(sign_set):
            line(sign, [p for p in sel if p["sign"] == sign],
                 len(uids_by_sign.get(sign, set())))
        line("total", sel, sum(len(u) for u in uids_by_sign.values()))
        dist = collections.Counter(
            (p["sign"], f"{p['winner_policy']}_{p['winner_variant']}")
            for p in sel)
        print("winners:")
        for (sign, pol), n in sorted(dist.items()):
            print(f"  {sign:<7}{pol:<38}{n:>6}")

    metrics_table("top-1 (best per scene_uid)", rank1)
    metrics_table("top-2 (up to 2 experts per scene_uid)", picks)
    n2 = sum(1 for p in rank1 if p["n_emitted_per_scene"] >= 2)
    print(f"\nscenes with a second expert: {n2}/{len(rank1)} "
          f"({n2 / max(1, len(rank1)):.0%}); rank-2 by policy:")
    dist2 = collections.Counter(
        (p["sign"], f"{p['winner_policy']}_{p['winner_variant']}")
        for p in picks if p["rank"] == 2)
    for (sign, pol), n in sorted(dist2.items()):
        print(f"  {sign:<7}{pol:<38}{n:>6}")
    metrics_table("map (best per map)", map_picks)

    # --- 8. output files ---
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
    # Merged deduped source rows — the input downstream metric tables
    # (make_oracle_metrics_table.py) should consume instead of the raw
    # per-node all_runs files, which contain resume duplicates.
    dump("all_runs_dedup.jsonl", rows)
    json.dump({"load": lstats, "join_rate": rate,
               "n_top1": len(rank1), "n_top2": len(picks),
               "n_map": len(map_picks),
               "winner_mismatches": n_win_mismatch,
               "invariant_failures": n_invariant,
               "missing_files": n_missing_files,
               "uncovered_uids": uncovered_uids,
               "uncovered_maps": uncovered_maps},
              open(out_dir / "coverage_report.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"  {out_dir / 'coverage_report.json'}")

    if n_win_mismatch or n_invariant:
        sys.exit("VERIFICATION FAILED: see MISMATCH/INVARIANT above")
    print("\nVerification passed: winners confirmed by independent "
          "recomputation, invariants hold.")


if __name__ == "__main__":
    main()
