"""Render up to N random scene GIFs for a single sign code (visual validation).
Usage: python render_samples.py <code> <out_root> [n=10] [seed=7]
"""
import os, sys, json, random
os.environ["PER_SIGN_USE_DESTINATION"]="1"
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..",".."))
sys.path.insert(0, os.path.join(ROOT,"scripts/per_sign_bench"))
sys.path.insert(0, os.path.join(ROOT,"scripts/per_sign_bench/sumo_space"))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import sumo_space.sumo_runner as R
from sumo_space.sumo_catalog import build_catalog

code=sys.argv[1]; out_root=sys.argv[2]
n=int(sys.argv[3]) if len(sys.argv)>3 else 10
seed=int(sys.argv[4]) if len(sys.argv)>4 else 7
out_dir=os.path.join(out_root, code); os.makedirs(out_dir, exist_ok=True)
try:
    rows=build_catalog(os.path.join(ROOT,"scenes"), n_per_category=100000,
                       n_variations=1, sign_categories=[code], seed=42)
except Exception as e:
    print(f"[{code}] catalog build failed: {e}"); sys.exit(0)
by_scene={}
for r in rows:
    sid=str(r["sign_id"])
    if sid not in by_scene: by_scene[sid]=r
ids=sorted(by_scene)
random.Random(seed).shuffle(ids)
pick=ids[:n]
ok=0
for sid in pick:
    gif=os.path.join(out_dir, f"sumo_{code}_{sid}.gif")
    try:
        R.materialize_sumo_scene(by_scene[sid], drive_agent=True, agent_policy="idm",
                                 save_gif=gif, max_steps=150)
        if os.path.exists(gif): ok+=1
    except Exception as e:
        print(f"[{code}/{sid}] render error: {str(e)[:120]}")
print(f"[{code}] rendered {ok}/{len(pick)} gifs -> {out_dir}")
