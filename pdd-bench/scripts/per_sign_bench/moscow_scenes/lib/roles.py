"""Dual-path slots and sign → slot matching (sign-free harvest atoms).

A slot is an exact ``(baseline_dir, compliant_dir)`` pair. Multi-dir sign
roles (e.g. 4.1.4 ``l`` vs ``s|r``) expand to several slots at allocate time.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Optional, Sequence, Set, Tuple

# Closed catalog: all ordered pairs among {l,s,r} with b != c.
SLOTS: Tuple[str, ...] = ("l_s", "l_r", "r_s", "r_l", "s_l", "s_r")
SLOT_SET: FrozenSet[str] = frozenset(SLOTS)
_CARDINAL = frozenset({"l", "s", "r"})


def slot_name(baseline_dir: str, compliant_dir: str) -> str:
    b = str(baseline_dir).strip().lower()
    c = str(compliant_dir).strip().lower()
    if b not in _CARDINAL or c not in _CARDINAL or b == c:
        raise ValueError(f"Invalid dual-path dirs baseline={baseline_dir!r} compliant={compliant_dir!r}")
    return f"{b}_{c}"


def parse_slot(slot: str) -> Tuple[str, str]:
    s = str(slot).strip().lower()
    if s not in SLOT_SET:
        raise ValueError(f"Unknown slot {slot!r}; expected one of {', '.join(SLOTS)}")
    b, c = s.split("_", 1)
    return b, c


# Sign → allowed atomic slots (+ shape policy lives in sign_shape_policy).
SIGN_TO_SLOTS: Dict[str, FrozenSet[str]] = {
    "5.7.1": frozenset({"l_s", "l_r"}),
    "5.7.2": frozenset({"r_s", "r_l"}),
    "3.18.1": frozenset({"r_s", "r_l"}),  # no right
    "3.18.2": frozenset({"l_s", "l_r"}),  # no left
    "4.1.1": frozenset({"l_s", "r_s"}),
    "4.1.2": frozenset({"s_r", "l_r"}),
    "4.1.3": frozenset({"s_l", "r_l"}),
    "4.1.4": frozenset({"l_s", "l_r"}),
    "4.1.5": frozenset({"r_s", "r_l"}),
    "4.1.6": frozenset({"s_r", "s_l"}),
    "3.1": SLOT_SET,
}


def sign_to_slots(pdd_code: str) -> FrozenSet[str]:
    code = str(pdd_code).strip()
    if code not in SIGN_TO_SLOTS:
        raise ValueError(
            f"No dual-path slot map for {code!r}; known: {', '.join(sorted(SIGN_TO_SLOTS))}"
        )
    return SIGN_TO_SLOTS[code]


def sign_shape_policy(pdd_code: str) -> FrozenSet[str]:
    """Allowed junction shapes for dual_path allocation."""
    code = str(pdd_code).strip()
    if code in ("5.7.1", "5.7.2"):
        return frozenset({"T"})
    if code in ("4.1.1", "4.1.4", "4.1.5"):
        return frozenset({"X"})  # needs a straight option
    if code in SIGN_TO_SLOTS:
        return frozenset({"T", "X"})
    raise ValueError(f"No shape policy for {code!r}")


def requires_t_stem(pdd_code: str) -> bool:
    return str(pdd_code).strip() in ("5.7.1", "5.7.2")


def requires_carriageway_pair(pdd_code: str) -> bool:
    return str(pdd_code).strip() in ("5.7.1", "5.7.2")


def scenario_matches_sign(
    meta: dict,
    pdd_code: str,
    *,
    require_stem: Optional[bool] = None,
    require_carriageway: Optional[bool] = None,
) -> bool:
    """True if a harvested dual_path meta can serve ``pdd_code``."""
    code = str(pdd_code).strip()
    try:
        slots = sign_to_slots(code)
        shapes = sign_shape_policy(code)
    except ValueError:
        return False

    shape = str(meta.get("shape") or "").upper()
    if shape and shape not in shapes:
        return False

    slot = meta.get("slot")
    if not slot:
        b = meta.get("baseline_dir") or (meta.get("dual_path") or {}).get("baseline_dir")
        c = meta.get("compliant_dir") or (meta.get("dual_path") or {}).get("compliant_dir")
        if not b or not c:
            return False
        try:
            slot = slot_name(str(b), str(c))
        except ValueError:
            return False
    if str(slot) not in slots:
        return False

    stem_needed = requires_t_stem(code) if require_stem is None else require_stem
    if stem_needed and not bool(meta.get("ego_is_t_stem")):
        return False

    cw_needed = (
        requires_carriageway_pair(code)
        if require_carriageway is None
        else require_carriageway
    )
    if cw_needed and not bool(meta.get("carriageway_pair")):
        return False

    return True


def slots_from_iterable(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not values:
        return SLOTS
    out: list[str] = []
    for v in values:
        parse_slot(v)  # validate
        if v not in out:
            out.append(str(v).strip().lower())
    return tuple(out)
