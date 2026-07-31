#!/usr/bin/env python3
"""Compress GIFs for the TrafficRuleBench docs site / GitHub Pages.

Shrinks frame size, optionally subsamples frames, and re-encodes with an
adaptive palette so multi‑MB MetaDrive dumps fit under a target size.

Examples:
    # Compress everything under docs/static/gifs/
    python tools/compress_gifs.py

    # One file / directory, custom caps
    python tools/compress_gifs.py path/to/file.gif --max-side 640 --max-mb 4

    # In-place overwrite (default writes *.min.gif next to source unless --in-place)
    python tools/compress_gifs.py static/gifs/pairs/3.1 --in-place
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageSequence


DOCS = Path(__file__).resolve().parents[1]


def _iter_gifs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = p.expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".gif":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.gif")))
    return out


def compress_gif(
    src: Path,
    dst: Path,
    *,
    max_side: int = 720,
    max_frames: int | None = 120,
    colors: int = 128,
    duration_ms: int | None = None,
    max_duration_ms: int = 80,
) -> tuple[int, int, int]:
    """Return (n_frames_out, bytes_in, bytes_out).

    Always writes via a temp file first so in-place overwrite cannot truncate
    the source while Pillow still has it open.

    When frames are subsampled, per-frame duration is scaled by the step so
    wall-clock length stays similar — but clamped to ``max_duration_ms`` so
    playback never drops below ~12 FPS (avoids laggy expert dumps with 1000+ frames).
    Pass ``duration_ms`` to force a fixed per-frame delay (no step scaling).
    """
    bytes_in = src.stat().st_size
    im = Image.open(src)
    frames_in = [frame.copy() for frame in ImageSequence.Iterator(im)]
    im.close()
    n_in = len(frames_in)

    step = 1
    if max_frames is not None and n_in > max_frames:
        step = max(1, (n_in + max_frames - 1) // max_frames)

    out_frames: list[Image.Image] = []
    durations: list[int] = []
    for i, frame in enumerate(frames_in):
        if i % step != 0:
            continue
        rgb = frame.convert("RGBA")
        w, h = rgb.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        pal = rgb.convert("RGB").quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        out_frames.append(pal)
        if duration_ms is not None:
            durations.append(max(20, int(duration_ms)))
        else:
            d = int(frame.info.get("duration", 40) or 40)
            durations.append(max(20, min(int(max_duration_ms), d * step)))

    if not out_frames:
        raise ValueError(f"No frames in {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish replace: never write on top of an open source path.
    with tempfile.NamedTemporaryFile(
        suffix=".gif", delete=False, dir=dst.parent
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        out_frames[0].save(
            tmp_path,
            save_all=True,
            append_images=out_frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
            disposal=2,
        )
        tmp_path.replace(dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return len(out_frames), bytes_in, dst.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress docs GIFs for GitHub Pages.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DOCS / "static" / "gifs"],
        help="GIF files or directories (default: docs/static/gifs)",
    )
    parser.add_argument("--max-side", type=int, default=720, help="Max width/height in px")
    parser.add_argument("--max-frames", type=int, default=120, help="Cap frame count (subsample)")
    parser.add_argument("--colors", type=int, default=128, help="GIF palette size")
    parser.add_argument("--duration-ms", type=int, default=None, help="Override frame duration")
    parser.add_argument(
        "--max-duration-ms",
        type=int,
        default=80,
        help="Cap per-frame duration after subsample scaling (default 80 ≈ 12.5 FPS)",
    )
    parser.add_argument("--max-mb", type=float, default=None, help="Skip if already under this size")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source files. Default writes <name>.min.gif",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gifs = _iter_gifs(list(args.paths))
    if not gifs:
        sys.exit("No GIF files found.")

    print(f"Found {len(gifs)} GIF(s)")
    total_in = total_out = 0
    for src in gifs:
        if src.name.endswith(".min.gif") and not args.in_place:
            continue
        mb = src.stat().st_size / (1024 * 1024)
        if args.max_mb is not None and mb <= args.max_mb:
            print(f"  skip  {src.name} ({mb:.1f} MB ≤ {args.max_mb} MB)")
            continue
        dst = src if args.in_place else src.with_name(src.stem + ".min.gif")
        if args.dry_run:
            print(f"  would compress {src} → {dst}")
            continue
        try:
            n, bin_, bout = compress_gif(
                src,
                dst,
                max_side=args.max_side,
                max_frames=args.max_frames,
                colors=args.colors,
                duration_ms=args.duration_ms,
                max_duration_ms=args.max_duration_ms,
            )
        except Exception as exc:
            print(f"  FAIL  {src.name}: {exc}")
            continue
        total_in += bin_
        total_out += bout
        print(
            f"  OK    {src.name}: {bin_/1e6:.1f} MB → {bout/1e6:.1f} MB "
            f"({n} frames) → {dst.name}"
        )

    if total_in:
        print(
            f"\nTotal: {total_in/1e6:.1f} MB → {total_out/1e6:.1f} MB "
            f"({100 * total_out / total_in:.0f}%)"
        )


if __name__ == "__main__":
    main()
