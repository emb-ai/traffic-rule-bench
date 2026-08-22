"""Shim — implementation is ``traffic_bench.eval.signs.dual_path.spec``."""

from __future__ import annotations

from typing import FrozenSet

from traffic_bench.eval.signs.dual_path.spec import (
    DualPathSignSpec,
    dual_path_role_dirs,
    get_spec,
    resolve_sign_class,
)

ONE_WAY_SIGN_CODES: tuple[str, ...] = ("5.7.1", "5.7.2")
DEFAULT_PDD_CODE = "5.7.1"
OneWaySignSpec = DualPathSignSpec


def get_one_way_sign_spec(pdd_code: str | None = None) -> DualPathSignSpec:
    return get_spec(pdd_code, family="one_way")


def dirs_allowed_by_sign(pdd_code: str) -> FrozenSet[str]:
    return get_one_way_sign_spec(pdd_code).allowed_dirs


def dirs_forbidden_by_sign(pdd_code: str) -> FrozenSet[str]:
    return frozenset({get_one_way_sign_spec(pdd_code).forbidden_dir})
