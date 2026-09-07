#!/usr/bin/env bash
# Render N sample GIFs on the TEST catalog of one benchmark.
#
# run_benchmark.py writes the GIF inside a `try/except: pass`, so a failed
# render leaves no file and no message. This script therefore feeds it more
# candidate scenes than requested and reports which ones actually landed.
#
# Usage: make_gifs.sh <detour|stop> <ckpt> <out_dir> <gpu> [n_wanted]
set -euo pipefail
TRB=/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench
SM=/home/jovyan/shares/SR006.nfs2/smirnova
PY=/home/jovyan/.mlspace/envs/arbelyaev-plant2/bin/python
BENCH="$TRB/pdd-bench/scripts/per_sign_bench"

KIND="${1:?usage: make_gifs.sh <detour|stop> <ckpt> <out_dir> <gpu> [n]}"
CKPT="${2:?}"; OUT="${3:?}"; GPU="${4:?}"; WANT="${5:-5}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PLANT2_SIGN_RANGE_M=90

if [ "$KIND" = detour ]; then
  MANIFEST="$SM/traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl"
  SCENES="$SM/sdc/pdd-bench/scenes"
  export PLANT2_DUMP_SIGN_CLASSES="4.2.1,4.2.2,4.2.3"
else
  MANIFEST="$TRB/stop_data/output/ts_test/real_manifest.jsonl"
  SCENES="$TRB/stop_data/scenes"
  export PLANT2_DUMP_SIGN_CLASSES="2.5"
fi

mkdir -p "$OUT"
SUB="$OUT/manifest.jsonl"
# Take 2x the wanted count so failed renders can be dropped.
"$PY" - "$MANIFEST" "$SUB" "$KIND" "$((WANT * 2))" <<'PYEOF'
import json, sys
src, dst, kind, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(src) if l.strip()]
picks = []
if kind == "detour":
    # one per sign code with cones first, then without (false-positive check)
    for cones in (True, False):
        for code in ("4.2.1", "4.2.2", "4.2.3"):
            for r in rows:
                if r["sign_code"] == code and r.get("detour_cones") is cones:
                    picks.append(r); break
    seen = {id(p) for p in picks}
    picks += [r for r in rows if id(r) not in seen]
else:
    picks = rows
with open(dst, "w") as f:
    for r in picks[:n]:
        f.write(json.dumps(r) + "\n")
print(f"кандидатов: {min(n, len(picks))}")
PYEOF

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$BENCH/run_benchmark.py" \
  --policy plant2 --model-path "$CKPT" --run-name gifs --ego-variant default \
  --manifest "$SUB" --scenes-root "$SCENES" --benchmark-output "$OUT" \
  --backends sumo --plant2-action-mode pid --save-gifs --gif-dir "$OUT/gifs"

n=$(find "$OUT/gifs" -name '*.gif' 2>/dev/null | wc -l)
echo "== отрисовано GIF: $n (просили $WANT)"
find "$OUT/gifs" -name '*.gif' -printf '%f  %s байт\n' 2>/dev/null | sort | head -20
