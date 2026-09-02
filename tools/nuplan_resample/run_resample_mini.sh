#!/usr/bin/env bash
# Recompute nuplan_statistics from the nuPlan v1.1 mini split, end to end.
#
# Expects both archives already downloaded and unpacked under $NUPLAN_ROOT:
#   $NUPLAN_ROOT/nuplan-v1.1/splits/mini/*.db   (nuplan-v1.1_mini.zip)
#   $NUPLAN_ROOT/maps/...                       (nuplan-maps-v1.0.zip)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NUPLAN_ROOT=${NUPLAN_ROOT:?set NUPLAN_ROOT to the unpacked nuPlan dataset}
PY=${PY:-python3}
OUT=${OUT:-$NUPLAN_ROOT/nuplan_statistics_v2}
OLD=${OLD:-}
WORKERS=${WORKERS:-32}

DATA=$(find "$NUPLAN_ROOT" -type d -name mini -path "*splits*" | head -1)
MAPS=$(find "$NUPLAN_ROOT" -maxdepth 3 -type d -name maps | head -1)
[ -n "$DATA" ] || { echo "mini split not found under $NUPLAN_ROOT"; exit 1; }
echo "data: $DATA"
echo "maps: ${MAPS:-NOT FOUND (lane changes will be skipped)}"

$PY "$HERE/extract_nuplan_statistics.py" \
    --data-root "$DATA" \
    ${MAPS:+--maps-root "$MAPS"} \
    --out "$OUT" --workers "$WORKERS" "$@"

# Comparing against a previous set is optional: point OLD at one to get a report
# of what moved.
if [ -n "$OLD" ] && [ -d "$OLD" ]; then
  $PY "$HERE/compare_nuplan_statistics.py" \
      --old "$OLD" --new "$OUT" --out "$OUT/comparison_report.md"
fi
