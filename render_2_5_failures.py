"""Identify N 'failed' scenes for any PDD sign (no compliant pick across all
agents) and produce a small manifest + bash command to render them as GIFs.

A scene is "failed" if NO episode passed Strategy F's filter (compliant + arrived
+ no crash + no oor).

Usage:
    python3 render_2_5_failures.py                    # default: 2.5, 10 scenes
    SIGN=5.11.1 python3 render_2_5_failures.py        # any sign
    SIGN=3.20 N_SCENES=20 python3 render_2_5_failures.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from collections import defaultdict

# EDIT or pass via env var
SIGN = os.environ.get("SIGN", "2.5")
N_SCENES = int(os.environ.get("N_SCENES", "10"))
RUN_ROOT = Path(os.environ.get(
    "RUN_ROOT",
    "/Users/victoria_s/sdc_new_signs/sign_experts_20260428_174805"))

SIGN_DIR = SIGN.replace(".", "_")
ALL_RUNS = RUN_ROOT / "all_runs.jsonl"
MANIFEST_SRC = Path(
    f"/Users/victoria_s/sdc_new_signs/mini_new/{SIGN_DIR}/sumo/sumo_manifest.jsonl"
)
GIF_MANIFEST = RUN_ROOT / f"{SIGN_DIR}_gif_manifest.jsonl"


def is_compliant(r):
    return int(r.get("total_violations", 0) or 0) == 0


def is_arrived(r):
    return bool(r.get("arrived_dest"))


def is_disqualified(r):
    return bool(r.get("crashed")) or bool(r.get("out_of_road"))


def passes_F(r):
    if is_disqualified(r):
        return False
    if not is_compliant(r):
        return False
    return is_arrived(r)


def normalize_sign(s):
    return s.replace("_", ".") if isinstance(s, str) else s


def main():
    # 1) Group all_runs by scene
    rows = [json.loads(l) for l in ALL_RUNS.read_text().splitlines() if l.strip()]
    valid = [r for r in rows if r.get("valid")]
    by_scene = defaultdict(list)
    for r in valid:
        if normalize_sign(r.get("sign_code") or r.get("sign_slug")) != SIGN:
            continue
        by_scene[r.get("scene_id")].append(r)

    # 2) Find scenes where NO episode passes Strategy F
    failed_scenes = sorted(
        sid for sid, eps in by_scene.items()
        if not any(passes_F(r) for r in eps)
    )
    print(f"Found {len(failed_scenes)} 'failed' scenes for sign {SIGN}")
    print(f"  (out of {len(by_scene)} total)")
    print()

    # 3) Pick first N scenes deterministically
    pick_scenes = failed_scenes[:N_SCENES]
    print(f"Picked {len(pick_scenes)} scenes for GIF rendering:")
    for s in pick_scenes:
        print(f"  {s}")
    print()

    # 4) Build manifest from sidecar JSON (each recorded episode has source_row)
    selected_rows = []
    seen = set()
    by_sign_dir = ALL_RUNS.parent / "by_sign" / SIGN.replace(".", "_") / "by_scene"
    for scene_uid_dir in by_sign_dir.iterdir():
        if not scene_uid_dir.is_dir():
            continue
        # scene_uid format: <scene_id>_lane<N>_seed<N>
        scene_id_part = scene_uid_dir.name.split("_lane")[0]
        if scene_id_part not in pick_scenes or scene_id_part in seen:
            continue
        # Pick any variant's sidecar (they all have same source_row)
        for variant_dir in scene_uid_dir.iterdir():
            sc = variant_dir / "replay.json"
            if not sc.exists():
                continue
            try:
                sidecar = json.loads(sc.read_text())
            except Exception:
                continue
            sr = sidecar.get("source_row")
            if sr:
                selected_rows.append(sr)
                seen.add(scene_id_part)
                break
    print(f"Reconstructed {len(selected_rows)} manifest rows from sidecar source_row")
    if len(selected_rows) < N_SCENES:
        print(f"  (expected {N_SCENES}; some scenes missing pkl/sidecar)")

    GIF_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(GIF_MANIFEST, "w") as f:
        for r in selected_rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote: {GIF_MANIFEST}")

    # 5) Suggest commands to render
    print()
    print("=" * 80)
    print("Команда для рендера GIF на сервере (или локально с conda env metadrive_signs):")
    print("=" * 80)
    out_run = f"/Users/victoria_s/sdc_new_signs/sign_experts_20260428_174805_{SIGN_DIR}_gifs"
    print(f"""
RUN={out_run}

conda run -n plant2 --no-capture-output python3 \\
    /Users/victoria_s/sdc_new_signs/sdc/pdd-bench/scripts/per_sign_bench/expert_replay.py \\
    --manifest    {GIF_MANIFEST} \\
    --code        {SIGN} \\
    --backend     sumo \\
    --policy      comprehensive \\
    --output-dir  $RUN \\
    --count       9999 \\
    --max-steps   600 \\
    --save-gifs

# GIF файлы появятся в:
# $RUN/by_sign/{SIGN_DIR}/by_scene/<scene_uid>/comprehensive_default/replay.gif
""")


if __name__ == "__main__":
    main()
