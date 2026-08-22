"""Shim — implementation is ``traffic_bench.eval.pipeline.run``."""

import traffic_bench.eval.pipeline.run as _impl

globals().update({n: getattr(_impl, n) for n in dir(_impl) if not n.startswith("__")})


if __name__ == "__main__":
    main()
