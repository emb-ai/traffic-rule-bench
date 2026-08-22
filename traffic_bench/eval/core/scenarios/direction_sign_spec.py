"""Shim — implementation is ``traffic_bench.eval.signs.dual_path.spec``."""

from traffic_bench.eval.signs.dual_path.spec import (
    DualPathSignSpec,
    dual_path_role_dirs,
    get_spec,
    resolve_sign_class,
)

DIRECTION_SIGN_CODES: tuple[str, ...] = (
    "4.1.1",
    "4.1.2",
    "4.1.3",
    "4.1.4",
    "4.1.5",
    "4.1.6",
)
DEFAULT_PDD_CODE = "4.1.1"
DirectionSignSpec = DualPathSignSpec


def get_direction_sign_spec(pdd_code: str | None = None) -> DualPathSignSpec:
    return get_spec(pdd_code, family="direction")
