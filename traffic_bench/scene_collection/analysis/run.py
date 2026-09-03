"""CLI router: inventory and overlap analysis experiments."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    import sys

    raw = list(sys.argv[1:] if argv is None else argv)
    # Back-compat: bare `analysis` → inventory.
    if not raw or raw[0].startswith("-"):
        from traffic_bench.scene_collection.analysis.inventory.run import main as inv_main

        return inv_main(raw)
    cmd = raw[0]
    rest = raw[1:]
    if cmd in {"inventory", "inv"}:
        from traffic_bench.scene_collection.analysis.inventory.run import main as inv_main

        return inv_main(rest)
    if cmd in {"overlap", "ov"}:
        from traffic_bench.scene_collection.analysis.overlap.run import main as ov_main

        return ov_main(rest)
    if cmd in {"assign", "assign-verify", "assign_verify", "verify"}:
        from traffic_bench.scene_collection.analysis.assign_verify import main as av_main

        return av_main(rest)
    # Unknown token: treat as inventory flags (old callers).
    from traffic_bench.scene_collection.analysis.inventory.run import main as inv_main

    return inv_main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
