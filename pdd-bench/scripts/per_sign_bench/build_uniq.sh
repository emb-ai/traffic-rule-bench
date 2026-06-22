#!/usr/bin/env bash
# build_uniq.sh — pull code, reorient scenes_uniq (persist road_id/offset fixes),
# then build the SUMO manifest (catalog + materialize, applying the runtime route
# fixes). Run detached with nohup; output is unbuffered.
#
#   nohup bash build_uniq.sh > /tmp/build_uniq.log 2>&1 & disown
#   tail -f /tmp/build_uniq.log
set -euo pipefail
export PYTHONUNBUFFERED=1                      # unbuffered python output
export SDL_AUDIODRIVER=dummy                   # silence ALSA/audio noise (headless)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null; chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

REPO=/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench
PY=/home/jovyan/shares/SR006.nfs2/smirnova/.conda-envs/plant2/bin/python   # env with metadrive
SC=scenes_uniq
OUT=benchmark_output/uniq
CODES=3.24,3.25,5.21,5.22,5.31,4.6   # 5.31/4.6 may be absent in a set (skipped, harmless)

log(){ echo "[$(date '+%F %T')] $*"; }

# 0) code with the fixes
cd "$REPO"
log "git pull dev_v_s ..."
git fetch origin && git checkout dev_v_s && git pull origin dev_v_s
cd "$REPO/pdd-bench"

# (optional) keep the base pristine — work on a copy: uncomment below
# SC=scenes_uniq_fixed; [ -d "$SC" ] || cp -r scenes_uniq "$SC"

# scene counts per code (before building)
log "scene counts in $SC:"
for d in "$SC"/*/; do
  printf '    %-8s %s\n' "$(basename "$d")" \
    "$(find "$d" -maxdepth 1 -type d -name 'sign_*' | wc -l)"
done | sort

# 1) persist fixes in meta.json: 5.21 orientation + resnap (5.21->courtyard, 3.24->street)
log "reorient 5.21 (dry-run report) ..."
"$PY" scripts/osm_scene_collection/reorient_zone_signs.py --scenes-root "$SC" --codes 5.21 --report
log "reorient 5.21 (apply) ..."
"$PY" scripts/osm_scene_collection/reorient_zone_signs.py --scenes-root "$SC" --codes 5.21 --apply
log "reorient 3.24 (apply) ..."
"$PY" scripts/osm_scene_collection/reorient_zone_signs.py --scenes-root "$SC" --codes 3.24 --apply

# 2) manifest: catalog + materialize (runtime route / 4.6 fixes applied here)
log "build manifest -> $OUT/sumo_manifest.jsonl ..."
"$PY" scripts/per_sign_bench/sumo_space/sumo_pipeline.py \
  --scenes-root "$SC" \
  --n-per-category 500 --n-variations 1 \
  --sign-categories "$CODES" \
  --output-dir "$OUT" --n-workers 16 --seed 42

# 3) sanity route check: auto-pick 5 ids per code (route <=3 edges past sign, sign on correct road)
for code in 5.21 3.24; do
  ids=$(ls -d "$SC/$code"/sign_* 2>/dev/null | head -5 | sed 's#.*/sign_##')
  if [ -n "$ids" ]; then
    log "route_metrics $code: $ids"
    "$PY" scripts/osm_scene_collection/route_metrics.py "$code" $ids || log "route_metrics $code failed (non-fatal)"
  fi
done

log "DONE. manifest: $REPO/pdd-bench/$OUT/sumo_manifest.jsonl"
