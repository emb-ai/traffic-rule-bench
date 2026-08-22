"""Shim — Hydra entry is ``traffic_bench.eval.manifest.run``."""

import traffic_bench.eval.manifest.run as _impl

globals().update({n: getattr(_impl, n) for n in dir(_impl) if not n.startswith("__")})


if __name__ == "__main__":
    main()
