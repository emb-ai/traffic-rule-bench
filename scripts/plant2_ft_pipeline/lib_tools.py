"""Small helpers shared across scripts/plant2_ft_pipeline/tools/*.py.

Deliberately NOT part of lib/ (that package covers data/train/eval/shell/shims
plumbing and is edited elsewhere); this module only holds bits of logic that
were hand-copied between debug/analysis scripts under tools/.

Importing this module requires `plant2/PlanT` on sys.path (for
`util.sign_id`) -- callers already insert that path before importing
lib_tools, matching the existing convention in tools/*.py. `util.sign_id`
itself has no torch/beartype dependency, so importing lib_tools is safe from
environments that lack those (see tools/hist_sign_planT_dataset.py
--no-torch). cv2/numpy (needed only for `class_colors_bgr()`) are imported
lazily inside that function for the same reason -- don't pay for them just to
call `boxes_has_sign()`.
"""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

from util.sign_id import SIGN_CODES

# --------------------------------------------------------------------------
# "sign survives PlanTDataset filtering" rule.
#
# Identical logic was hand-copied in:
#   - tools/print_plant_batch.py        (boxes_has_class)
#   - tools/hist_sign2_5_planT_dataset.py (_boxes_has_class)
#   - tools/hist_sign2_5_planT_dataset_no_torch.py (_boxes_has_class)
# All three: sign must be within `range_m` in xy, |z| <= range_m, and
# affects_ego must be True.
#
# NOTE: tools/validate_dump_sample.py's boxes_to_x_objs() has a *similar but
# not identical* sign-range check (xy radius only, no |z| <= range_m term) --
# see the discrepancy note in that file. It is deliberately NOT wired to this
# helper so as not to silently change its filtering behavior.
# --------------------------------------------------------------------------

SIGN_RANGE_M = 30.0
SIGN_LIKE_CLASSES: frozenset[str] = frozenset(SIGN_CODES) | {"stop_sign"}


def sign_survives_filter(obj: dict, *, range_m: float = SIGN_RANGE_M) -> bool:
    """True if a sign-like box dict would survive PlanTDataset's sign filter.

    Rule: within `range_m` meters in xy, |z| <= range_m, and affects_ego is
    True. Does not look at obj["class"] -- callers filter by class first.
    """
    pos = obj.get("position") or [0.0, 0.0, 0.0]
    px, py = float(pos[0]), float(pos[1])
    pz = float(pos[2]) if len(pos) > 2 else 0.0
    if px * px + py * py > range_m ** 2 or abs(pz) > range_m:
        return False
    return bool(obj.get("affects_ego"))


def boxes_has_sign(path: str | Path, sign_class: str, *, range_m: float = SIGN_RANGE_M) -> bool:
    """True if `sign_class` (a PDD sign code, e.g. "2.5", or "stop_sign")
    survives the PlanTDataset sign filter somewhere in boxes/NNNN.json.gz.

    Only meaningful for sign-like classes (see SIGN_LIKE_CLASSES): for
    anything else this always returns False. Callers that need an
    unfiltered/non-sign class presence check (e.g.
    tools/print_plant_batch.py's boxes_has_class for "car") implement that
    themselves.
    """
    want = str(sign_class)
    if want not in SIGN_LIKE_CLASSES:
        return False
    with gzip.open(path, "rt", encoding="utf-8") as f:
        boxes = json.load(f)
    if not isinstance(boxes, list) or len(boxes) < 2:
        return False
    for obj in boxes[1:]:  # skip ego
        if str(obj.get("class")) != want:
            continue
        if sign_survives_filter(obj, range_m=range_m):
            return True
    return False


# --------------------------------------------------------------------------
# BGR color table for x_objs type_id, shared by validate_dump_sample.py's
# debug BEV overlay and viz_train_global_gif.py's route GIFs.
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def class_colors_bgr() -> dict[float, tuple[int, int, int]]:
    """BGR color per x_objs type_id (cv2/numpy imported lazily -- see module
    docstring). Cached: callers must not mutate the returned dict."""
    import cv2
    import numpy as np

    colors: dict[float, tuple[int, int, int]] = {
        1.0: (0, 0, 220),      # car
        2.0: (0, 220, 220),    # walker
        3.0: (180, 180, 180),  # static
        4.0: (220, 0, 220),    # stop_sign
        5.0: (0, 0, 255),      # traffic_light
        6.0: (0, 140, 255),    # emergency
    }
    for i, _code in enumerate(SIGN_CODES):
        hue = int(180 * i / max(len(SIGN_CODES), 1))
        bgr = cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors[float(7 + i)] = tuple(int(x) for x in bgr)
    return colors
