#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from select_experts import (    # noqa: E402
    SIGN_CLASS_MAP, NO_ENTRY_SIGNS, HORIZON_DEFAULT, BETA_DEFAULT,
    MIN_ROUTE_COMPLETION, MIN_FINAL_STEP,
    normalize_sign,
    select_expert_per_scene,
)
from select_experts_by_scene_id import select_expert_per_scene_id    # noqa: E402


POLICIES = ("comprehensive", "rule_compliant", "carl", "plant2")
COMP_VARIANTS = ("comprehensive_default", "comprehensive_s1", "comprehensive_s2",
                 "comprehensive_s3", "comprehensive_s4")
SHORT_TO_VARIANT = {
    "comprehensive_default": "default",
    "comprehensive_s1": "s1",
    "comprehensive_s2": "s2",
    "comprehensive_s3": "s3",
    "comprehensive_s4": "s4",
}

EXPERT_DISPLAY = {
    "comprehensive_default": "idm_default",
    "comprehensive_s1": "idm_s1",
    "comprehensive_s2": "idm_s2",
    "comprehensive_s3": "idm_s3",
    "comprehensive_s4": "idm_s4",
    "rule_compliant": "ppo",
    "carl": "carl",
    "plant2": "plant2",
}


def display_policy(pol, var=None):
    """Render policy[+variant] for presentation.
    'comprehensive'+'s2' -> 'idm_s2'; 'rule_compliant' -> 'ppo'; 'carl' -> 'carl'.
    """
    if pol == "comprehensive":
        return f"idm_{var or 'default'}"
    if pol == "rule_compliant":
        return "ppo"
    return pol or ""


def sign_slug_from_path(parts):
    """Extract sign slug from path components.
    Layout: <out_base>/<policy>/<sign>/by_sign/<sign>/by_scene/<scene_uid>/<expert>/replay.json
    The sign right after by_sign is the most reliable.
    Fallback: the component right after policy at the start of the path.
    """
    for i, p in enumerate(parts):
        if p == "by_sign" and i + 1 < len(parts):
            sign = parts[i + 1]
            if sign not in ("_filt", "by_scene", "_tmp_manifests"):
                return sign
    for i, p in enumerate(parts):
        if p in POLICIES and i + 1 < len(parts):
            cand = parts[i + 1]
            if cand and cand != "_filt" and ("_" in cand or cand.isdigit()):
                return cand
    return ""


def parse_sidecar_to_row(sc_path, cache_key=None):
    """Parse sidecar.json into a row dict (the format expected by select_experts).
    Returns the row, or None if anything goes wrong.
    """
    parts = sc_path.parts
    # Extract policy/sign/expert/scene_uid from the path:
    # .../by_scene/<scene_uid>/<expert>/replay.json
    try:
        expert = parts[-2]
        scene_uid = parts[-3]
    except IndexError:
        return None
    sign_slug = sign_slug_from_path(parts)
    sign_code = sign_slug.replace("_", ".") if "_" in sign_slug and "." not in sign_slug else sign_slug

    # Mapping expert dirname -> (policy, variant). Extended with the new run_names
    # from run_experts_test_chunk.sh (idm_*, carl_rule, plant2_rule, plant2_artem,
    # ppo_metadrive). All of them write a sidecar in the expert_replay layout.
    IDM_VARIANTS = ("idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4")
    EXTRA_EXPERTS = ("ppo_metadrive", "carl_rule", "plant2_rule", "plant2_artem")
    SUFFIXED_POLICIES = ("carl", "carl_rule", "plant2", "plant2_rule",
                         "rule_compliant", "ppo_metadrive", "plant2_artem")
    if expert in COMP_VARIANTS:
        policy = "comprehensive"
        variant = SHORT_TO_VARIANT[expert]
    elif expert in IDM_VARIANTS:
        policy = "idm"
        variant = expert.replace("idm_", "")
    elif expert in ("rule_compliant", "carl", "plant2") + EXTRA_EXPERTS:
        policy = expert
        variant = None
    elif "_" in expert and any(expert.startswith(f"{p}_") for p in SUFFIXED_POLICIES):
        # Sort by length descending so longer prefixes (carl_rule) are checked
        # before shorter ones (carl); otherwise "carl_rule_default" would match "carl_".
        policy = None
        for p in sorted(SUFFIXED_POLICIES, key=len, reverse=True):
            if expert == f"{p}_default" or expert.startswith(f"{p}_"):
                policy = p
                suffix = expert[len(p) + 1:]
                variant = suffix if suffix and suffix != "default" else None
                break
        if policy is None:
            return None
    else:
        return None

    # Read the sidecar first — we need its data.valid flag to determine validity
    # without requiring a pkl (the new pipeline only writes statistics).
    try:
        data = json.loads(sc_path.read_text())
    except Exception:
        return None

    pkl = sc_path.parent / "replay.pkl"
    if pkl.exists():
        pkl_valid = pkl.stat().st_size > 0
    else:
        # No pkl on disk -> fall back to the sidecar's own `valid` flag (the new
        # pipeline writes `valid: True` for successful rollouts without saving the trajectory).
        pkl_valid = bool(data.get("valid", True))

    m = data.get("metrics") or {}
    scene_id = data.get("scene_id")
    if not scene_id:
        return None

    row = {
        "valid": pkl_valid,
        "policy": policy,
        "variant": variant,
        "sign_code": sign_code,
        "sign_slug": sign_slug,
        "scene_id": str(scene_id),
        "scene_uid": scene_uid,
        # Metrics needed for passes_filter and F1
        "crashed": bool(m.get("crashed", False)),
        "out_of_road": bool(m.get("out_of_road", False)),
        "arrived_dest": bool(m.get("arrived_dest", False)),
        "violations_by_class": m.get("violations_by_class") or {},
        "final_step": int(m.get("final_step") or 0),
        "route_completion": float(m.get("route_completion") or 0.0),
        "frame_smooth_ratio": float(m.get("frame_smooth_ratio") or 0.0),
        "initial_speed_mps": float(m.get("initial_speed_mps") or 0.0),
        # Paths used by output writers
        "pkl_path": str(pkl) if pkl_valid else None,
        "sidecar_path": str(sc_path),
        "gif_path": None,
    }
    return row


def parse_old_index_row(r):
    """OLD index format -> list of rows (one per agent).
    Index entry layout:
        {scene_id, sign_code, agents: {agent_name: {valid, pkl, metrics}}}
    """
    rows = []
    sid = r.get("scene_id")
    sign_code = (r.get("sign_code") or r.get("sign_slug") or "")
    if "_" in sign_code and "." not in sign_code:
        sign_code = sign_code.replace("_", ".")
    sign_slug = sign_code.replace(".", "_") if sign_code else ""
    if not sid or not sign_code:
        return rows

    IDM_VARIANTS = ("idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4")
    EXTRA_EXPERTS = ("ppo_metadrive", "carl_rule", "plant2_rule", "plant2_artem")

    agents = r.get("agents") or {}
    for agent_name, info in agents.items():
        if agent_name in COMP_VARIANTS:
            policy = "comprehensive"
            variant = SHORT_TO_VARIANT[agent_name]
        elif agent_name in IDM_VARIANTS:
            policy = "idm"
            variant = agent_name.replace("idm_", "")
        elif agent_name in ("rule_compliant", "carl", "plant2") + EXTRA_EXPERTS:
            policy = agent_name
            variant = None
        else:
            continue

        pkl_valid = bool(info.get("valid")) and bool(info.get("pkl"))
        # The OLD style may have metrics either directly in info, or nested under "metrics"
        m = info.get("metrics") or info

        rows.append({
            "valid": pkl_valid,
            "policy": policy,
            "variant": variant,
            "sign_code": sign_code,
            "sign_slug": sign_slug,
            "scene_id": str(sid),
            "scene_uid": str(sid),    # OLD has no scene_uid -> reuse scene_id
            "crashed": bool(m.get("crashed", False)),
            "out_of_road": bool(m.get("out_of_road", False)),
            "arrived_dest": bool(m.get("arrived_dest", False)),
            "violations_by_class": m.get("violations_by_class") or {},
            "final_step": int(m.get("final_step") or 0),
            "route_completion": float(m.get("route_completion") or 0.0),
            "frame_smooth_ratio": float(m.get("frame_smooth_ratio") or 0.0),
            "initial_speed_mps": float(m.get("initial_speed_mps") or 0.0),
            "pkl_path": info.get("pkl"),
            "sidecar_path": info.get("sidecar"),
            "gif_path": info.get("gif"),
        })
    return rows


# === Cache ===
CACHE_SCHEMA_VERSION = 1


def load_cache(cache_file):
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text())
        if data.get("_meta", {}).get("schema") != CACHE_SCHEMA_VERSION:
            return {}
        return data.get("entries", {})
    except Exception:
        return {}


def save_cache(cache_file, cache):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "_meta": {"schema": CACHE_SCHEMA_VERSION, "saved_at": time.time(),
                   "n_entries": len(cache)},
        "entries": cache,
    }
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(cache_file)


def scan_base(base, label, cache, force=False):
    """Scan a replay-source directory and return a list of rows.
    Uses the cache to skip files that have already been parsed (keyed by mtime).
    """
    if not base.is_dir():
        print(f"  [warn] {label}: {base} does not exist, skipping")
        return []

    rows = []
    n_total = 0
    n_cached = 0
    n_fresh = 0
    n_skipped = 0
    seen_paths = set()
    base_str = str(base)

    t0 = time.time()
    for sc in base.glob("*/*/by_sign/*/by_scene/*/*/replay.json"):
        n_total += 1
        sc_str = str(sc)
        seen_paths.add(sc_str)
        try:
            mtime = sc.stat().st_mtime
        except OSError:
            n_skipped += 1
            continue

        cached = cache.get(sc_str)
        if not force and cached and cached.get("mtime") == mtime and "row" in cached:
            n_cached += 1
            rows.append(cached["row"])
            continue

        n_fresh += 1
        row = parse_sidecar_to_row(sc)
        if row is None:
            n_skipped += 1
            continue
        cache[sc_str] = {"mtime": mtime, "row": row}
        rows.append(row)

    # Drop stale entries from the cache
    stale = [p for p in cache if p.startswith(base_str) and p not in seen_paths]
    for p in stale:
        del cache[p]

    print(f"  [{label}] {n_total:>6} sidecars  "
          f"(cached={n_cached:,} fresh={n_fresh:,} skipped={n_skipped} "
          f"stale_removed={len(stale)})  {time.time()-t0:.1f}s")
    return rows


def scan_old_index(old_base, label, cache):
    """Scan an OLD index file and return a list of rows."""
    if not old_base.is_dir():
        print(f"  [warn] {label}: {old_base} does not exist, skipping")
        return []
    idx = old_base / "_merged" / "scenes_by_sign.jsonl"
    if not idx.exists():
        print(f"  [warn] {label}: {idx} not found, skipping")
        return []

    # A line-offset cache makes no sense — the index is written atomically. Cache by mtime.
    try:
        mtime = idx.stat().st_mtime
    except OSError:
        return []

    cache_key = f"_OLD_INDEX_:{idx}"
    cached = cache.get(cache_key)
    if cached and cached.get("mtime") == mtime and "rows" in cached:
        rows = cached["rows"]
        print(f"  [{label}] cached: {len(rows):,} rows from {idx.name}")
        return rows

    rows = []
    n_lines = 0
    t0 = time.time()
    with open(idx) as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.extend(parse_old_index_row(r))
    cache[cache_key] = {"mtime": mtime, "rows": rows}
    print(f"  [{label}] {n_lines:,} lines -> {len(rows):,} rows  "
          f"{time.time()-t0:.1f}s (fresh)")
    return rows


def compute_validation_stats(all_rows, args):
    """Compute aggregated stats tables (mirrors count_attempted_no_valid.sh):
      - per_expert: attempts, passes, pass%
      - per_sign x per_expert: counts of valid (sign, scene_id) pairs

    Uses the same passes_filter as the selection step.
    """
    from select_experts import passes_filter, SIGN_CLASS_MAP

    EXPERTS = ("comprehensive_default", "comprehensive_s1", "comprehensive_s2",
               "comprehensive_s3", "comprehensive_s4",
               "rule_compliant", "carl", "plant2")
    EXPERTS_SET = set(EXPERTS)

    sign_filter = set(args.signs) if args.signs else None

    # Per-expert: attempts and passes
    expert_attempt = Counter()
    expert_pass = Counter()
    # Per-sign x per-expert: set of valid scene_ids
    sign_expert_valid = defaultdict(lambda: defaultdict(set))
    sign_expert_attempt = defaultdict(lambda: defaultdict(set))
    sign_to_sids_attempted = defaultdict(set)
    sign_to_sids_valid = defaultdict(set)

    for r in all_rows:
        sign_code = r.get("sign_code") or ""
        if not sign_code: continue
        if sign_filter and sign_code not in sign_filter: continue
        sign_slug = r.get("sign_slug") or sign_code.replace(".", "_")
        sid = r.get("scene_id")
        if not sid: continue

        # Map policy/variant -> expert_dir_name
        pol = r.get("policy")
        var = r.get("variant")
        if pol == "comprehensive":
            expert = f"comprehensive_{var}" if var else "comprehensive_default"
        else:
            expert = pol
        if expert not in EXPERTS_SET: continue

        target_class = SIGN_CLASS_MAP.get(sign_code)
        if target_class is None: continue

        expert_attempt[expert] += 1
        sign_expert_attempt[sign_slug][expert].add(sid)
        sign_to_sids_attempted[sign_slug].add(sid)

        ok = passes_filter(r, sign_code, target_class, args.horizon,
                            args.min_route_completion, args.min_final_step)
        if ok:
            expert_pass[expert] += 1
            sign_expert_valid[sign_slug][expert].add(sid)
            sign_to_sids_valid[sign_slug].add(sid)

    # Compute full coverage 8/8 per sign
    sign_full = {}
    for sign_slug, exp_sids in sign_expert_valid.items():
        # For each sid collect the set of experts with a valid replay
        sid_to_exp = defaultdict(set)
        for exp, sids in exp_sids.items():
            for sid in sids:
                sid_to_exp[sid].add(exp)
        sign_full[sign_slug] = sum(1 for ex_set in sid_to_exp.values()
                                    if ex_set >= EXPERTS_SET)

    return {
        "experts": EXPERTS,
        "expert_attempt": dict(expert_attempt),
        "expert_pass": dict(expert_pass),
        "sign_expert_valid": {s: {e: len(sids) for e, sids in exps.items()}
                                for s, exps in sign_expert_valid.items()},
        "sign_to_sids_attempted": {s: len(sids) for s, sids in sign_to_sids_attempted.items()},
        "sign_to_sids_valid": {s: len(sids) for s, sids in sign_to_sids_valid.items()},
        "sign_full": sign_full,
    }


def write_validation_tables(stats, out_dir):
    """Write 2 CSVs: per_expert_pass_rate.csv + per_sign_per_expert.csv"""
    out_dir = Path(out_dir)
    EXPERTS = stats["experts"]

    # === per_expert_pass_rate.csv ===
    p1 = out_dir / "per_expert_pass_rate.csv"
    with open(p1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expert", "attempts", "passes", "pass_pct"])
        for ex in EXPERTS:
            a = stats["expert_attempt"].get(ex, 0)
            p = stats["expert_pass"].get(ex, 0)
            pct = round(100 * p / a, 2) if a else 0
            w.writerow([EXPERT_DISPLAY.get(ex, ex), a, p, pct])
    print(f"  -> {p1}  ({len(EXPERTS)} rows)")

    # === per_sign_per_expert.csv ===
    p2 = out_dir / "per_sign_per_expert.csv"
    short = ["idm_def", "idm_s1", "idm_s2", "idm_s3", "idm_s4", "ppo", "carl", "plnt"]
    fields = ["sign"] + short + ["total_attempted", "full_8", "pct_full"]
    all_signs = sorted(stats["sign_to_sids_attempted"].keys())
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sign in all_signs:
            row = {"sign": sign}
            for ex, sh in zip(EXPERTS, short):
                row[sh] = stats["sign_expert_valid"].get(sign, {}).get(ex, 0)
            n_att = stats["sign_to_sids_attempted"].get(sign, 0)
            n_full = stats["sign_full"].get(sign, 0)
            row["total_attempted"] = n_att
            row["full_8"] = n_full
            row["pct_full"] = round(100 * n_full / n_att, 1) if n_att else 0
            w.writerow(row)
    print(f"  -> {p2}  ({len(all_signs)} rows)")


def compute_full_coverage_pairs(all_rows, args):
    """Return the set of (sign_code, scene_id) where ALL 8 experts pass passes_filter.

    Used both by oracle metrics and by splitting picks into strategies.
    """
    from select_experts import passes_filter, SIGN_CLASS_MAP

    EXPERTS = ("comprehensive_default", "comprehensive_s1", "comprehensive_s2",
               "comprehensive_s3", "comprehensive_s4",
               "rule_compliant", "carl", "plant2")
    EXPERTS_SET = set(EXPERTS)
    sign_filter = set(args.signs) if args.signs else None
    sid_passing_experts = defaultdict(lambda: defaultdict(set))

    for r in all_rows:
        sc = r.get("sign_code") or ""
        if not sc: continue
        if sign_filter and sc not in sign_filter: continue
        target_class = SIGN_CLASS_MAP.get(sc)
        if not target_class: continue
        pol = r.get("policy")
        var = r.get("variant")
        if pol == "comprehensive":
            ex = f"comprehensive_{var}" if var else "comprehensive_default"
        else:
            ex = pol
        if ex not in EXPERTS_SET: continue
        if passes_filter(r, sc, target_class, args.horizon,
                          args.min_route_completion, args.min_final_step):
            sid_passing_experts[sc][r.get("scene_id")].add(ex)

    return {(sc, sid)
            for sc, sid_to_exps in sid_passing_experts.items()
            for sid, exps in sid_to_exps.items()
            if exps >= EXPERTS_SET}


def split_picks_by_strategy(picks, full_coverage_pairs):
    """Split picks into 4 strategies (cumulative):
      - top1:    rank=1 picks (one best pick per scene)
      - top2:    rank=1 + rank=2 (two best trajectories per scene — superset of top1)
      - top1_fc: top1 restricted to full-covered scenes
      - top2_fc: top2 restricted to full-covered scenes
    Returns: dict[strategy_name] -> list of picks.
    """
    out = {"top1": [], "top2": [], "top1_fc": [], "top2_fc": []}
    for p in picks:
        rank = p.get("rank", 1)
        if rank not in (1, 2): continue
        is_fc = (p.get("sign"), p.get("scene_id")) in full_coverage_pairs
        # rank=1 goes into both (top1 + top2); rank=2 only into top2
        if rank == 1:
            out["top1"].append(p)
            out["top2"].append(p)
            if is_fc:
                out["top1_fc"].append(p)
                out["top2_fc"].append(p)
        else:
            out["top2"].append(p)
            if is_fc:
                out["top2_fc"].append(p)
    return out


def write_strategy_outputs(picks, out_dir, jsonl_dir, full_coverage_pairs):
    """For each of the 4 strategies, write:
      - <jsonl_dir>/expert_picks_<strategy>.jsonl  — all picks for the strategy
      - <out_dir>/expert_pkl_list_<strategy>.txt   — pkl paths of the winners

    Returns: dict[strategy] -> (n_picks, jsonl_path, pkl_list_path)
    """
    out_dir = Path(out_dir)
    jsonl_dir = Path(jsonl_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    by_strategy = split_picks_by_strategy(picks, full_coverage_pairs)
    summary = {}
    for strat, strat_picks in by_strategy.items():
        jsonl_path = jsonl_dir / f"expert_picks_{strat}.jsonl"
        with open(jsonl_path, "w") as f:
            for p in strat_picks:
                f.write(json.dumps(p, default=str) + "\n")
        pkl_path = out_dir / f"expert_pkl_list_{strat}.txt"
        n_pkls = 0
        with open(pkl_path, "w") as f:
            for p in strat_picks:
                pkl = p.get("pkl_path")
                if pkl:
                    f.write(f"{pkl}\n")
                    n_pkls += 1
        summary[strat] = (len(strat_picks), jsonl_path, pkl_path, n_pkls)
        print(f"  -> [{strat}]  {len(strat_picks):>6,} picks  "
              f"{n_pkls:>6,} pkls  ->  {jsonl_path.name}, {pkl_path.name}")
    return summary


def compute_oracle_metrics(all_rows, picks, args, full_coverage_pairs=None):
    """Per-sign x per-expert raw metrics + ORACLE column.
    The output format mirrors oracle_metrics_summary.md.

    Returns: dict[sign_code or "__all__"] -> dict[expert] -> dict[metric] -> value
    """
    from select_experts import (passes_filter, time_eff, comfort, f1_score,
                                 recompute_dest, SIGN_CLASS_MAP)

    EXPERTS = ("comprehensive_default", "comprehensive_s1", "comprehensive_s2",
               "comprehensive_s3", "comprehensive_s4",
               "rule_compliant", "carl", "plant2")
    EXPERTS_SET = set(EXPERTS)
    sign_filter = set(args.signs) if args.signs else None

    # Group rows by (sign_code, expert) and at the same time track scene_min/max for time_eff
    grouped = defaultdict(lambda: defaultdict(list))
    scene_minmax = defaultdict(lambda: [10**9, 0])

    for r in all_rows:
        sign_code = r.get("sign_code") or ""
        if not sign_code: continue
        if sign_filter and sign_code not in sign_filter: continue
        pol = r.get("policy")
        var = r.get("variant")
        if pol == "comprehensive":
            expert = f"comprehensive_{var}" if var else "comprehensive_default"
        else:
            expert = pol
        if expert not in EXPERTS_SET: continue
        grouped[sign_code][expert].append(r)

    # scene_minmax is collected from passing trajectories (used for time_eff normalization).
    # In parallel we collect, per scene, the set of experts that passed — needed
    # either to compute full_coverage_pairs (if not supplied) or as a skip filter.
    sid_passing_experts = defaultdict(lambda: defaultdict(set))
    for sign_code, exp_rows in grouped.items():
        target_class = SIGN_CLASS_MAP.get(sign_code)
        if not target_class: continue
        for expert, rows in exp_rows.items():
            for r in rows:
                if passes_filter(r, sign_code, target_class, args.horizon,
                                  args.min_route_completion, args.min_final_step):
                    sid = r.get("scene_id")
                    fs = max(1, int(r.get("final_step") or 1))
                    mm = scene_minmax[sid]
                    mm[0] = min(mm[0], fs)
                    mm[1] = max(mm[1], fs)
                    sid_passing_experts[sign_code][sid].add(expert)
    # Full-coverage scene set: (sign, sid) where all 8 experts passed
    if full_coverage_pairs is None:
        full_coverage_pairs = {(sc, sid)
                               for sc, sid_to_exps in sid_passing_experts.items()
                               for sid, exps in sid_to_exps.items()
                               if exps >= EXPERTS_SET}

    # ORACLE rows: split into rank=1 (top1) and rank=2 (top2) — both are mapped
    # to a row via pkl_path. In parallel we collect the _fc variants — only picks
    # belonging to full-covered scenes. ORACLE_top2 is non-empty only when --top-n >= 2.
    pkl_to_row = {r["pkl_path"]: r for r in all_rows if r.get("pkl_path")}
    oracle_top1_per_sign = defaultdict(list)
    oracle_top2_per_sign = defaultdict(list)
    oracle_top1_fc_per_sign = defaultdict(list)
    oracle_top2_fc_per_sign = defaultdict(list)
    for p in picks:
        rank = p.get("rank", 1)
        if rank not in (1, 2): continue
        sign_code = p.get("sign")
        if sign_filter and sign_code not in sign_filter: continue
        row = pkl_to_row.get(p.get("pkl_path"))
        if row is None: continue
        is_fc = (sign_code, p.get("scene_id")) in full_coverage_pairs
        # top2 is cumulative: rank=1 goes into both top1 and top2
        if rank == 1:
            oracle_top1_per_sign[sign_code].append(row)
            oracle_top2_per_sign[sign_code].append(row)
            if is_fc:
                oracle_top1_fc_per_sign[sign_code].append(row)
                oracle_top2_fc_per_sign[sign_code].append(row)
        else:
            oracle_top2_per_sign[sign_code].append(row)
            if is_fc:
                oracle_top2_fc_per_sign[sign_code].append(row)

    def metrics_for_rows(rows, sign_code_for_compliance=None):
        n = len(rows)
        if n == 0:
            return {"n": 0, "n_passing": 0, "dest_rate": 0, "crash_rate": 0,
                    "out_of_road_rate": 0, "avg_violations": 0,
                    "compliance_rate": 0, "compliance_arrived_n": 0,
                    "compliance_arrived_pct": 0, "avg_comfort": 0,
                    "avg_time_eff_passing": 0, "avg_f1_passing": 0,
                    "pass_rate": 0,
                    "avg_total_violations": 0, "compliance_rate_total": 0}
        n_passing = n_arrived = n_crashed = n_oor = 0
        sum_viols = n_compliant = n_arrived_compliant = 0
        sum_total_viols = n_compliant_total = 0
        sum_comfort = sum_te = sum_f1 = 0.0
        for r in rows:
            sc = sign_code_for_compliance or r.get("sign_code")
            tc = SIGN_CLASS_MAP.get(sc) if sc else None
            # dest_rate (recomputed) — accounts for NO_ENTRY+compliance and horizon-timeout
            arrived = (recompute_dest(r, sc, tc, args.horizon)
                        if tc else bool(r.get("arrived_dest")))
            crashed = bool(r.get("crashed"))
            oor = bool(r.get("out_of_road"))
            vbc = r.get("violations_by_class") or {}
            target_viols = int(vbc.get(tc, 0) or 0) if tc else 0
            total_viols = sum(int(v or 0) for v in vbc.values())
            compliant = (target_viols == 0)
            compliant_total = (total_viols == 0)
            if arrived: n_arrived += 1
            if crashed: n_crashed += 1
            if oor: n_oor += 1
            if compliant: n_compliant += 1
            if compliant_total: n_compliant_total += 1
            if arrived and compliant: n_arrived_compliant += 1
            sum_viols += target_viols
            sum_total_viols += total_viols
            sum_comfort += comfort(r)
            if tc and passes_filter(r, sc, tc, args.horizon,
                                     args.min_route_completion, args.min_final_step):
                n_passing += 1
                mm = scene_minmax.get(r.get("scene_id"), [1, 1])
                t = time_eff(r, mm[0], mm[1], args.time_eff_formula)
                c = comfort(r)
                sum_te += t
                sum_f1 += f1_score(t, c, args.beta)
        return {
            "n": n,
            "n_passing": n_passing,
            "dest_rate": n_arrived / n,
            "crash_rate": n_crashed / n,
            "out_of_road_rate": n_oor / n,
            "avg_violations": sum_viols / n,
            "compliance_rate": n_compliant / n,
            "compliance_arrived_n": n_arrived,
            "compliance_arrived_pct": (n_arrived_compliant / n_arrived) if n_arrived else 0,
            "avg_comfort": sum_comfort / n,
            "avg_time_eff_passing": (sum_te / n_passing) if n_passing else 0,
            "avg_f1_passing": (sum_f1 / n_passing) if n_passing else 0,
            "pass_rate": n_passing / n,
            # Total-violations metrics (across ALL classes, not only the target)
            "avg_total_violations": sum_total_viols / n,
            "compliance_rate_total": n_compliant_total / n,
        }

    results = {}
    for sign_code in sorted(grouped.keys()):
        per_exp = {ex: metrics_for_rows(grouped[sign_code].get(ex, []), sign_code)
                    for ex in EXPERTS}
        per_exp["ORACLE_top1"] = metrics_for_rows(
            oracle_top1_per_sign.get(sign_code, []), sign_code)
        per_exp["ORACLE_top2"] = metrics_for_rows(
            oracle_top2_per_sign.get(sign_code, []), sign_code)
        per_exp["ORACLE_top1_fc"] = metrics_for_rows(
            oracle_top1_fc_per_sign.get(sign_code, []), sign_code)
        per_exp["ORACLE_top2_fc"] = metrics_for_rows(
            oracle_top2_fc_per_sign.get(sign_code, []), sign_code)
        results[sign_code] = per_exp

    # All-signs aggregate (use per-row sign_code for compliance)
    agg = {}
    for ex in EXPERTS:
        rows_all = []
        for sc in grouped:
            rows_all.extend(grouped[sc].get(ex, []))
        agg[ex] = metrics_for_rows(rows_all, None)
    rows_top1_all = [r for sc in oracle_top1_per_sign for r in oracle_top1_per_sign[sc]]
    rows_top2_all = [r for sc in oracle_top2_per_sign for r in oracle_top2_per_sign[sc]]
    rows_top1_fc_all = [r for sc in oracle_top1_fc_per_sign for r in oracle_top1_fc_per_sign[sc]]
    rows_top2_fc_all = [r for sc in oracle_top2_fc_per_sign for r in oracle_top2_fc_per_sign[sc]]
    agg["ORACLE_top1"] = metrics_for_rows(rows_top1_all, None)
    agg["ORACLE_top2"] = metrics_for_rows(rows_top2_all, None)
    agg["ORACLE_top1_fc"] = metrics_for_rows(rows_top1_fc_all, None)
    agg["ORACLE_top2_fc"] = metrics_for_rows(rows_top2_fc_all, None)
    results["__all__"] = agg
    # Also return full_coverage_pairs for use by write_*
    results["__meta__"] = {"full_coverage_pairs": full_coverage_pairs}

    return results


def write_oracle_metrics_md(metrics, picks, out_dir, args):
    """Write oracle_metrics_summary.md in the format used by the reference doc."""
    out_dir = Path(out_dir)
    md_path = out_dir / "oracle_metrics_summary.md"

    EXPERTS = ("comprehensive_default", "comprehensive_s1", "comprehensive_s2",
               "comprehensive_s3", "comprehensive_s4",
               "rule_compliant", "carl", "plant2")
    SHORT = dict(EXPERT_DISPLAY)
    SHORT.update({
        "ORACLE_top1": "oracle_top1",
        "ORACLE_top2": "oracle_top2",
        "ORACLE_top1_fc": "oracle_top1_fc",
        "ORACLE_top2_fc": "oracle_top2_fc",
    })
    expert_cols = (list(EXPERTS) +
                   ["ORACLE_top1", "ORACLE_top2", "ORACLE_top1_fc", "ORACLE_top2_fc"])

    full_coverage_pairs = (metrics.get("__meta__") or {}).get("full_coverage_pairs", set())

    # Oracle picks per (sign, expert) — used by the `oracle picks (of N)` row.
    # Top1 — rank=1; top2 — rank=2; *_fc — only picks for full-covered scenes.
    oracle_pick_counts_top1 = defaultdict(lambda: defaultdict(int))
    oracle_pick_counts_top2 = defaultdict(lambda: defaultdict(int))
    oracle_pick_counts_top1_fc = defaultdict(lambda: defaultdict(int))
    oracle_pick_counts_top2_fc = defaultdict(lambda: defaultdict(int))
    for p in picks:
        rank = p.get("rank", 1)
        if rank not in (1, 2): continue
        sc = p.get("sign", "")
        pol = p.get("winner_policy")
        var = p.get("winner_variant")
        ex = (f"comprehensive_{var}" if (pol == "comprehensive" and var)
              else "comprehensive_default" if pol == "comprehensive"
              else pol)
        is_fc = (sc, p.get("scene_id")) in full_coverage_pairs
        # top2 is cumulative (rank=1 goes into both top1 and top2)
        if rank == 1:
            oracle_pick_counts_top1[sc][ex] += 1
            oracle_pick_counts_top1["__all__"][ex] += 1
            oracle_pick_counts_top2[sc][ex] += 1
            oracle_pick_counts_top2["__all__"][ex] += 1
            if is_fc:
                oracle_pick_counts_top1_fc[sc][ex] += 1
                oracle_pick_counts_top1_fc["__all__"][ex] += 1
                oracle_pick_counts_top2_fc[sc][ex] += 1
                oracle_pick_counts_top2_fc["__all__"][ex] += 1
        else:
            oracle_pick_counts_top2[sc][ex] += 1
            oracle_pick_counts_top2["__all__"][ex] += 1
            if is_fc:
                oracle_pick_counts_top2_fc[sc][ex] += 1
                oracle_pick_counts_top2_fc["__all__"][ex] += 1

    def row(label, fmt_fn):
        return "| " + label + " | " + " | ".join(
            fmt_fn(ex) for ex in expert_cols) + " |\n"

    with open(md_path, "w") as f:
        # __meta__ is not a sign — exclude it from the iteration
        all_signs = ["__all__"] + sorted(s for s in metrics
                                          if s not in ("__all__", "__meta__"))
        for sign in all_signs:
            section = "all signs" if sign == "__all__" else sign
            f.write(f"## {section}\n\n")
            f.write("| metric | " + " | ".join(SHORT[e] for e in expert_cols) + " |\n")
            f.write("| --- | " + " | ".join(["---"] * len(expert_cols)) + " |\n")
            data = metrics[sign]
            top1_total = data.get("ORACLE_top1", {}).get("n", 0)
            top2_total = data.get("ORACLE_top2", {}).get("n", 0)
            top1_fc_total = data.get("ORACLE_top1_fc", {}).get("n", 0)
            top2_fc_total = data.get("ORACLE_top2_fc", {}).get("n", 0)

            # n (episodes)
            f.write(row("n (episodes)",
                        lambda ex: f"{data.get(ex, {}).get('n', 0)}"))
            # n_passing
            f.write(row("n_passing",
                        lambda ex: f"{data.get(ex, {}).get('n_passing', 0)}"))
            # dest_rate
            f.write(row("dest_rate (recomputed)",
                        lambda ex: f"{data.get(ex, {}).get('dest_rate', 0):.2f}"))
            # crash_rate
            f.write(row("crash_rate",
                        lambda ex: f"{data.get(ex, {}).get('crash_rate', 0):.2f}"))
            # out_of_road_rate
            f.write(row("out_of_road_rate",
                        lambda ex: f"{data.get(ex, {}).get('out_of_road_rate', 0):.2f}"))
            # avg_violations
            f.write(row("avg_violations (target)",
                        lambda ex: f"{data.get(ex, {}).get('avg_violations', 0):.2f}"))
            # avg_total_violations (across all violation classes)
            f.write(row("avg_total_violations (all classes)",
                        lambda ex: f"{data.get(ex, {}).get('avg_total_violations', 0):.2f}"))
            # compliance_rate
            f.write(row("compliance_rate (target)",
                        lambda ex: f"{data.get(ex, {}).get('compliance_rate', 0):.2f}"))
            # compliance_rate_total (no violations of any class)
            f.write(row("compliance_rate_total (all classes)",
                        lambda ex: f"{data.get(ex, {}).get('compliance_rate_total', 0):.2f}"))
            # compliance among arrived
            f.write(row("compliance among arrived",
                        lambda ex: (
                            f"{data.get(ex, {}).get('compliance_arrived_pct', 0):.2f} "
                            f"({data.get(ex, {}).get('compliance_arrived_n', 0)})")))
            # avg_comfort
            f.write(row("avg_comfort",
                        lambda ex: f"{data.get(ex, {}).get('avg_comfort', 0):.3f}"))
            # avg_time_eff (passing)
            f.write(row("avg_time_eff (passing)",
                        lambda ex: f"{data.get(ex, {}).get('avg_time_eff_passing', 0):.3f}"))
            # F1 beta
            f.write(row(f"F1 beta={args.beta} (passing)",
                        lambda ex: f"{data.get(ex, {}).get('avg_f1_passing', 0):.3f}"))
            # pass_rate
            f.write(row("pass_rate",
                        lambda ex: f"{data.get(ex, {}).get('pass_rate', 0):.2f}"))
            # oracle picks — 4 separate rows, one per strategy.
            # Under each expert column: how many times that expert was the winner
            # under this strategy (count + % within the strategy).
            # The ORACLE column is filled only for its own strategy (the total).
            strategy_data = [
                ("top1",    "ORACLE_top1",    top1_total,    oracle_pick_counts_top1),
                ("top2",    "ORACLE_top2",    top2_total,    oracle_pick_counts_top2),
                ("top1_fc", "ORACLE_top1_fc", top1_fc_total, oracle_pick_counts_top1_fc),
                ("top2_fc", "ORACLE_top2_fc", top2_fc_total, oracle_pick_counts_top2_fc),
            ]
            for label, oracle_col, total, counts in strategy_data:
                def _cell(ex, _oc=oracle_col, _t=total, _c=counts):
                    if ex == _oc:
                        return str(_t)
                    if ex.startswith("ORACLE_"):    # a different ORACLE column
                        return "—"
                    n = _c[sign].get(ex, 0)
                    pct = 100 * n / _t if _t else 0
                    return f"{n} ({pct:.0f}%)" if n else "0"
                f.write(row(f"oracle picks ({label})", _cell))
            f.write("\n")

    n_signs = sum(1 for s in metrics if s not in ("__all__", "__meta__"))
    print(f"  -> {md_path}  ({n_signs} signs + 'all signs')")


def write_markdown_report(picks, out_dir, group_by, args, n_rows_input,
                            scan_summary, stats=None, strategy_summary=None):
    """Write a markdown report with tables for presentations / sharing.

    Layout:
      1. Run parameters
      2. Picks summary (overall)
      3. Per-sign distribution (winner_policy)
      4. Per-policy aggregate (avg F1, time, etc.)
      5. Top-N scenes with the highest F1
      6. Bottom-N scenes (low F1) — for debugging
      7. Coverage (when group_by=scene_id)
    """
    from datetime import datetime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "REPORT.md"

    with open(md_path, "w") as f:
        # === Header ===
        f.write(f"# Expert Selection Report\n\n")
        f.write(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

        # === 1. Run parameters ===
        f.write(f"## 1. Run parameters\n\n")
        f.write(f"| Parameter | Value |\n|---|---|\n")
        f.write(f"| group_by | `{args.group_by}` |\n")
        f.write(f"| signs | `{', '.join(args.signs)}` |\n")
        f.write(f"| beta | `{args.beta}` |\n")
        f.write(f"| horizon | `{args.horizon}` |\n")
        f.write(f"| top_n | `{args.top_n}` |\n")
        f.write(f"| min_route_completion | `{args.min_route_completion}` "
                f"({'soft success enabled' if args.min_route_completion > 0 else 'OFF'}) |\n")
        f.write(f"| min_final_step | `{args.min_final_step}` "
                f"({'antibug' if args.min_final_step > 0 else 'OFF'}) |\n")
        f.write(f"| idm_pick | `{args.idm_pick}` |\n")
        f.write(f"| time_eff_formula | `{args.time_eff_formula}` |\n")
        if args.group_by == "scene_id":
            f.write(f"| per_agent_best | `{args.per_agent_best}` |\n")
        f.write(f"\n### Data sources\n\n")
        for label, path, n in scan_summary:
            f.write(f"- **{label}**: `{path}` — {n:,} sidecars/rows\n")
        f.write(f"\n**Total input rows**: {n_rows_input:,}\n\n")

        # === 2. Summary ===
        f.write(f"## 2. Picks summary\n\n")
        rank1 = [p for p in picks if p.get("rank", 1) == 1]
        unique_scenes = len({(p["sign"], p.get("scene_id"))
                              if group_by == "scene_id" else
                              (p["sign"], p.get("scene_uid"))
                              for p in rank1})
        avg_f1 = sum(p["f1_score"] for p in rank1) / max(1, len(rank1))
        avg_steps = sum(p["final_step"] for p in rank1) / max(1, len(rank1))
        avg_rc = sum(p.get("route_completion", 0) for p in rank1) / max(1, len(rank1))
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Picks total | {len(picks):,} |\n")
        f.write(f"| Unique scenes (rank=1) | {unique_scenes:,} |\n")
        f.write(f"| Avg F1 score | {avg_f1:.3f} |\n")
        f.write(f"| Avg final_step | {avg_steps:.1f} |\n")
        f.write(f"| Avg route_completion | {avg_rc:.3f} |\n")
        f.write(f"\n")

        # === 3. Per-sign distribution ===
        f.write(f"## 3. Per-sign x winner_policy\n\n")
        by_sign_policy = defaultdict(lambda: defaultdict(int))
        sign_totals = defaultdict(int)
        for p in rank1:
            sig = p["sign"]
            pol_disp = display_policy(p["winner_policy"], p.get("winner_variant"))
            by_sign_policy[sig][pol_disp] += 1
            sign_totals[sig] += 1

        all_policies = sorted({p for sig_pols in by_sign_policy.values()
                                for p in sig_pols})
        if all_policies:
            f.write(f"| sign | total | " + " | ".join(all_policies) + " |\n")
            f.write(f"|---|---|" + "|".join(["---"] * len(all_policies)) + "|\n")
            for sig in sorted(by_sign_policy):
                row = [sig, str(sign_totals[sig])]
                for pol in all_policies:
                    n = by_sign_policy[sig].get(pol, 0)
                    pct = 100 * n / max(1, sign_totals[sig])
                    cell = f"{n} ({pct:.0f}%)" if n else "—"
                    row.append(cell)
                f.write("| " + " | ".join(row) + " |\n")
        f.write(f"\n")

        # === 4. Per-policy aggregate ===
        f.write(f"## 4. Per-policy: quality of selected trajectories\n\n")
        by_policy = defaultdict(list)
        for p in rank1:
            pol_disp = display_policy(p["winner_policy"], p.get("winner_variant"))
            by_policy[pol_disp].append(p)
        f.write(f"| policy | n_winners | avg F1 | avg final_step | avg route_completion | avg time_eff | avg comfort |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for pol in sorted(by_policy, key=lambda p: -len(by_policy[p])):
            items = by_policy[pol]
            n = len(items)
            af1 = sum(p["f1_score"] for p in items) / n
            asteps = sum(p["final_step"] for p in items) / n
            arc = sum(p.get("route_completion", 0) for p in items) / n
            ate = sum(p.get("time_eff", 0) for p in items) / n
            ac = sum(p.get("comfort", 0) for p in items) / n
            f.write(f"| `{pol}` | {n:,} | {af1:.3f} | {asteps:.1f} | "
                    f"{arc:.3f} | {ate:.3f} | {ac:.3f} |\n")
        f.write(f"\n")

        # === 5. Top-20 scenes by F1 ===
        f.write(f"## 5. Top-20 scenes by F1 score\n\n")
        rank1_sorted = sorted(rank1, key=lambda p: -p["f1_score"])
        f.write(f"| sign | scene_id | winner | F1 | route_compl | final_step |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for p in rank1_sorted[:20]:
            pol = display_policy(p["winner_policy"], p.get("winner_variant"))
            sid = p.get("scene_id", "—")
            f.write(f"| {p['sign']} | `{sid}` | `{pol}` | "
                    f"{p['f1_score']:.3f} | {p.get('route_completion', 0):.3f} | "
                    f"{p['final_step']} |\n")
        f.write(f"\n")

        # === 6. Bottom-10 scenes by F1 (for debugging) ===
        f.write(f"## 6. Bottom-10 scenes by F1 (for debugging)\n\n")
        f.write(f"| sign | scene_id | winner | F1 | route_compl | final_step |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for p in rank1_sorted[-10:]:
            pol = display_policy(p["winner_policy"], p.get("winner_variant"))
            sid = p.get("scene_id", "—")
            f.write(f"| {p['sign']} | `{sid}` | `{pol}` | "
                    f"{p['f1_score']:.3f} | {p.get('route_completion', 0):.3f} | "
                    f"{p['final_step']} |\n")
        f.write(f"\n")

        # === 7. Best IDM variant distribution (only when idm picks exist) ===
        comp_picks = [p for p in rank1 if p.get("best_idm_variant")]
        if comp_picks:
            f.write(f"## 7. Best IDM variant distribution\n\n")
            f.write(f"_Which idm variant became best_idm on scenes that had idm among the candidates_\n\n")
            idm_dist = Counter(p["best_idm_variant"] for p in comp_picks)
            f.write(f"| variant | n | % |\n|---|---|---|\n")
            total_idm = sum(idm_dist.values())
            for v, n in sorted(idm_dist.items(), key=lambda x: -x[1]):
                f.write(f"| `idm_{v}` | {n:,} | {100*n/total_idm:.1f}% |\n")
            f.write(f"\n")

        # === 8. Pass rate by expert (aggregated) ===
        if stats:
            EXPERTS = stats["experts"]
            f.write(f"## 8. Pass rate by expert\n\n")
            f.write(f"_passes_filter applied to ALL trajectories from the 3 sources "
                    f"(NEW + ADD + OLD), filtered by --signs._\n\n")
            f.write(f"| expert | attempts | passes | pass% |\n")
            f.write(f"|---|---|---|---|\n")
            for ex in EXPERTS:
                a = stats["expert_attempt"].get(ex, 0)
                p = stats["expert_pass"].get(ex, 0)
                pct = 100 * p / a if a else 0
                f.write(f"| `{EXPERT_DISPLAY.get(ex, ex)}` | {a:,} | {p:,} | {pct:.1f}% |\n")
            f.write(f"\n")

            # === 9. Per-sign x per-expert (count of valid scene_ids) ===
            f.write(f"## 9. Coverage per-sign x per-expert\n\n")
            f.write(f"_Number of **unique scene_ids** with a valid replay for each "
                    f"(sign, expert) pair. Column `total` = total attempted scene_ids, "
                    f"`full` = scene_ids covered by all 8 experts._\n\n")
            short = ["idm_def", "idm_s1", "idm_s2", "idm_s3", "idm_s4", "ppo", "carl", "plnt"]
            header_cols = ["sign"] + short + ["total", "full", "%full"]
            f.write(f"| " + " | ".join(header_cols) + " |\n")
            f.write(f"|" + "|".join(["---"] * len(header_cols)) + "|\n")
            all_signs = sorted(stats["sign_to_sids_attempted"].keys())
            for sign in all_signs:
                row = [sign]
                for ex in EXPERTS:
                    n_v = stats["sign_expert_valid"].get(sign, {}).get(ex, 0)
                    row.append(str(n_v))
                n_att = stats["sign_to_sids_attempted"].get(sign, 0)
                n_full = stats["sign_full"].get(sign, 0)
                pct_full = 100 * n_full / n_att if n_att else 0
                row.extend([str(n_att), str(n_full), f"{pct_full:.0f}%"])
                f.write(f"| " + " | ".join(row) + " |\n")
            f.write(f"\n")

        # === 10. Output files ===
        f.write(f"## 10. Output files\n\n")
        f.write(f"- [`expert_picks.jsonl`](../{Path(args.output).name}) — raw picks JSONL ({len(picks):,} rows)\n")
        f.write(f"- [`picks_detail.csv`](picks_detail.csv) — every pick as a flat table\n")
        f.write(f"- [`picks_summary.csv`](picks_summary.csv) — per (sign x policy) aggregates\n")
        f.write(f"- [`per_scene_table.csv`](per_scene_table.csv) — per scene: winner + candidates\n")
        f.write(f"- [`expert_pkl_list.txt`](expert_pkl_list.txt) — list of pkl paths for the winners ({len(rank1):,} files)\n")
        if stats:
            f.write(f"- [`per_expert_pass_rate.csv`](per_expert_pass_rate.csv) — pass rate per expert\n")
            f.write(f"- [`per_sign_per_expert.csv`](per_sign_per_expert.csv) — per-sign x per-expert valid scene_id counts\n")
        f.write(f"- [`oracle_metrics_summary.md`](oracle_metrics_summary.md) — per-sign x per-expert raw metrics + ORACLE\n")

        # === 11. Per-strategy expert picks ===
        if strategy_summary:
            f.write(f"\n## 11. Per-strategy expert picks\n\n")
            f.write(f"_Picks split into 4 strategies. Each one comes with a dedicated JSONL "
                    f"(rows from selection) plus a list of pkl paths._\n\n")
            f.write(f"| strategy | n_picks | n_pkls | jsonl | pkl_list |\n")
            f.write(f"|---|---|---|---|---|\n")
            for strat, (n_picks, jsonl_path, pkl_path, n_pkls) in strategy_summary.items():
                f.write(f"| `{strat}` | {n_picks:,} | {n_pkls:,} | "
                        f"[`{jsonl_path.name}`](../{jsonl_path.name}) | "
                        f"[`{pkl_path.name}`]({pkl_path.name}) |\n")
            f.write(f"\nLegend (top2 is cumulative — superset of top1):\n")
            f.write(f"- `top1` — rank=1 (one best trajectory per scene)\n")
            f.write(f"- `top2` — rank=1 + rank=2 (two best trajectories per scene "
                    f"when at least 2 candidates exist)\n")
            f.write(f"- `top1_fc` — top1 restricted to full-covered scenes (all 8 experts passed)\n")
            f.write(f"- `top2_fc` — top2 restricted to full-covered scenes\n")

    print(f"  -> {md_path}")


def write_tables(picks, out_dir, group_by="scene_id"):
    """Write 4 tables for convenient analytics:
      1. picks_detail.csv     — every pick (1 row per pick)
      2. picks_summary.csv    — per (sign, winner_policy): count, avg F1
      3. per_scene_table.csv  — per (sign, scene): top-1 winner + all candidates
      4. expert_pkl_list.txt  — plain list of pkl paths for the winners
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1. picks_detail.csv (flat table) ===
    detail_path = out_dir / "picks_detail.csv"
    if picks:
        # Every key except the nested 'all_candidates'
        skip_keys = {"all_candidates"}
        all_keys = sorted({k for p in picks for k in p.keys()
                            if k not in skip_keys})
        with open(detail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for p in picks:
                w.writerow({k: p.get(k, "") for k in all_keys})
    print(f"  -> {detail_path}  ({len(picks):,} rows)")

    # === 2. picks_summary.csv (per sign x policy aggregate) ===
    summary_path = out_dir / "picks_summary.csv"
    rank1_picks = [p for p in picks if p.get("rank", 1) == 1]
    by_sign_policy = defaultdict(lambda: defaultdict(list))
    for p in rank1_picks:
        sig = p["sign"]
        pol = p["winner_policy"]
        if pol == "comprehensive" and p.get("winner_variant"):
            pol = f"comprehensive_{p['winner_variant']}"
        by_sign_policy[sig][pol].append(p)

    rows = []
    for sig in sorted(by_sign_policy):
        sig_picks = sum((picks for picks in by_sign_policy[sig].values()),
                          [])
        # ... [TRUNCATED — please send the rest of write_tables() and any code that follows]
