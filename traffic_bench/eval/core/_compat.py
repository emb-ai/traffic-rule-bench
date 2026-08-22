"""Re-export a moved module from a ``core`` shim, including private names."""

from __future__ import annotations

import importlib


def reexport(module_name: str, dest: dict) -> None:
    impl = importlib.import_module(module_name)
    dest.update({n: getattr(impl, n) for n in dir(impl) if not n.startswith("__")})
