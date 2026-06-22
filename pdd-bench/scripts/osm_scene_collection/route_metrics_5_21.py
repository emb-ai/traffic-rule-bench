"""Per-scene runtime route-metrics probe for 5.21 (entry-realism verification).

Usage: python route_metrics_5_21.py <sign_id> [<sign_id> ...]
Prints one JSON line per scene with: sign edge highway class, whether the route
starts on a big road, the sign is mid-route, the route continues into the
courtyard, and whether it degenerates. Exit 0 always; per-scene errors are JSON.
"""
import os, sys, json
os.environ["PER_SIGN_USE_DESTINATION"]="1"
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..",".."))
sys.path.insert(0, os.path.join(ROOT,"scripts/per_sign_bench"))
sys.path.insert(0, os.path.join(ROOT,"scripts/per_sign_bench/sumo_space"))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import sumo_space.sumo_runner as R
from sumo_space.sumo_catalog import build_catalog
import xml.etree.ElementTree as ET
from sign_edge_orientation import _hwy_class
BIG={"primary","secondary","tertiary","trunk","motorway","primary_link","secondary_link","tertiary_link","trunk_link","motorway_link"}
def edge_of(li):
    s=str(li); s=s[5:] if s.startswith("lane_") else s
    if s.startswith(":"): return None
    p=s.split("_")
    while len(p)>1 and p[-1].isdigit(): p=p[:-1]
    return "_".join(p)
def etypes(net):
    r=ET.parse(net).getroot(); return {e.get("id"):_hwy_class(e.get("type")) for e in r.findall("edge") if e.get("function")!="internal"}
want=set(sys.argv[1:])
rows=build_catalog(os.path.join(ROOT,"scenes"), n_per_category=10000, n_variations=1, sign_categories=["5.21"], seed=42)
seen={}
for r in rows:
    sid=str(r["sign_id"])
    if sid in want and sid not in seen: seen[sid]=r
for sid in sys.argv[1:]:
    if sid not in seen:
        print(json.dumps({"scene":sid,"error":"no_row"})); continue
    row=seen[sid]
    try:
        net=os.path.join(ROOT,"scenes",row["net_path"]); et=etypes(net); se=str(row["road_id"])
        res=R.materialize_sumo_scene(row, drive_agent=True, agent_policy="idm", max_steps=80)
        fp=res.get("fingerprint",{}) or {}
        ck=[edge_of(c) for c in fp.get("route_checkpoints",[])]; ck=[c for c in ck if c]
        sign_hwy=et.get(se)
        sign_mid = se in ck and ck.index(se) not in (0,len(ck)-1)
        after=ck[ck.index(se)+1:] if se in ck else []
        out={"scene":sid,"sign_edge":se,"sign_hwy":sign_hwy,
             "sign_on_residential": sign_hwy not in BIG,
             "start_big": bool(ck) and et.get(ck[0]) in BIG,
             "sign_mid": bool(sign_mid),
             "continues_courtyard": any(et.get(c) not in BIG for c in after),
             "degenerate": len(ck)<=2,
             "n_ckpt": len(ck), "routed": bool(res.get("routed_through_sign"))}
    except Exception as e:
        out={"scene":sid,"error":str(e)[:200]}
    print(json.dumps(out))
