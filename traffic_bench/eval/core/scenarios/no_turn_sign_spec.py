"""Shim — implementation is ``traffic_bench.eval.signs.dual_path.spec``."""

from traffic_bench.eval.signs.dual_path.spec import (
    DualPathSignSpec,
    dual_path_role_dirs,
    get_spec,
    resolve_sign_class,
)

NO_TURN_SIGN_CODES: tuple[str, ...] = ("3.18.1", "3.18.2")
DEFAULT_PDD_CODE = "3.18.1"
NoTurnSignSpec = DualPathSignSpec


def get_no_turn_sign_spec(pdd_code: str | None = None) -> DualPathSignSpec:
    return get_spec(pdd_code, family="no_turn")
