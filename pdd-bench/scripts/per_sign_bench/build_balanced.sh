#!/usr/bin/env bash
# build_balanced.sh — make a balanced scene set and its ×10 manifest:
#   1) snapshot scenes_uniq -> scenes_balanced (never touch the original; the
#      running eval reads .net.xml straight from scenes_uniq)
#   2) (idempotent) persist 5.21/3.24 placement fixes on the copy
#   3) redistribute (cap mode): bring 4.6 & 5.31 UP to N, donors 3.24 & 5.21
#      give only what's needed and keep their surplus (>= N); N is the max equal
#      count = min((|3.24|+|4.6|)//2, (|5.21|+|5.31|)//2). No scenes dropped.
#   4) build the SUMO manifest with --n-per-category N (caps the donor surplus,
#      e.g. 5.21 & 5.22, so all four codes end at N) and --n-variations 10.
#
#   nohup bash build_balanced.sh > /tmp/build_balanced.log 2>&1 & disown
#   tail -f /tmp/build_balanced.log
set -euo pipefail
export PYTHONUNBUFFERED=1                      # unbuffered python output
export SDL_AUDIODRIVER=dummy                   # silence ALSA/audio noise (headless)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null; chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

REPO=/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench
PY=/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/plant2/bin/python   # env with metadrive
SRC=scenes_uniq            # pristine source (left untouched)
SC=scenes_balanced         # working copy we redistribute in place
OUT=benchmark_output_speed/balanced
CODES=3.24,3.25,5.21,5.22,5.31,4.6
MIN_ROAD_KMH=45            # 3.24->4.6 donors need road OSM speed >= this, so the
                           # enforced min (road-10, cap 40/60) lands >= 35 and sits
                           # ABOVE base cruise (~30) -> base violates, compliant obeys

log(){ echo "[$(date '+%F %T')] $*"; }

# 0) latest code with the catalog clamp + redistribute script
cd "$REPO"
log "git pull dev_v_s ..."
git fetch origin && git checkout dev_v_s && git pull origin dev_v_s
cd "$REPO/pdd-bench"

# 1) snapshot (fresh copy each run so the redistribution is reproducible)
if [ -e "$SC" ]; then
  log "removing stale $SC ..."
  rm -rf "$SC"
fi
log "copy $SRC -> $SC ..."
cp -r "$SRC" "$SC"
log "scene counts in $SC (before):"
for d in "$SC"/*/; do
  printf '    %-8s %s\n' "$(basename "$d")" \
    "$(find "$d" -maxdepth 1 -type d -name 'sign_*' | wc -l)"
done | sort

# 2) persist placement fixes on the COPY (idempotent — scenes_uniq is already done)
log "reorient 5.21 (apply) ..."
"$PY" scripts/osm_scene_collection/reorient_zone_signs.py --scenes-root "$SC" --codes 5.21 --apply
log "reorient 3.24 (apply) ..."
"$PY" scripts/osm_scene_collection/reorient_zone_signs.py --scenes-root "$SC" --codes 3.24 --apply

# 3) redistribute (cap mode): recipients 4.6/5.31 -> N, donors kept >= N.
#    dry-run first for the log, capture N, then apply.
log "redistribute (dry-run preview) ..."
"$PY" scripts/osm_scene_collection/redistribute_scenes.py \
  --root "$SC" --mode cap --target auto --min-road-kmh "$MIN_ROAD_KMH" --seed 42
N=$("$PY" scripts/osm_scene_collection/redistribute_scenes.py \
  --root "$SC" --mode cap --target auto --print-n)
log "max equal N=$N"
log "redistribute (apply) ..."
"$PY" scripts/osm_scene_collection/redistribute_scenes.py \
  --root "$SC" --mode cap --target auto --min-road-kmh "$MIN_ROAD_KMH" --seed 42 --apply

# 4) manifest: catalog + materialize, ×10 variations, cap each code to N
#    (caps the donor surplus 5.21/5.22 so the four codes end at N; 3.25 stays
#    natural if < N).
log "build manifest -> $OUT/sumo_manifest.jsonl (×10, cap N=$N) ..."
"$PY" scripts/per_sign_bench/sumo_space/sumo_pipeline.py \
  --scenes-root "$SC" \
  --n-per-category "$N" --n-variations 10 \
  --sign-categories "$CODES" \
  --output-dir "$OUT" --n-workers 16 --seed 42

log "DONE. manifest: $REPO/pdd-bench/$OUT/sumo_manifest.jsonl"
log "next: run_eval_8gpu.sh with MANIFEST=$OUT/sumo_manifest.jsonl SCENES=$SC OUT=$OUT/eval"
