#!/usr/bin/env python3
"""Build a simple HTML gallery for failure_cases GIFs."""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/runs/failure_cases"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    index_csv = root / "index.csv"
    out_html = root / "gifs_index.html"

    rows: list[dict[str, str]] = []
    if index_csv.is_file():
        with index_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    cards: list[str] = []
    for row in rows:
        rel = Path(row["path"])
        gif = root / rel / row["baseline"] / "replay.gif"
        if not gif.is_file():
            continue
        gif_rel = gif.relative_to(root).as_posix()
        title = (
            f"{row['category']} | {row['sign_group']} | {row['baseline']} | "
            f"{row['scene_uid']}"
        )
        cards.append(
            f"<section class='card'>"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>compliant={html.escape(str(row.get('target_compliant_event','')))} "
            f"arrived={html.escape(str(row.get('arrived_dest','')))} "
            f"viol_events={html.escape(str(row.get('violations_event_count','')))}</p>"
            f"<img src='{html.escape(gif_rel)}' loading='lazy' />"
            f"<p><a href='{html.escape(gif_rel)}'>open gif</a></p>"
            f"</section>"
        )

    # Also pick up any GIFs not listed in index.csv
    listed = {c for c in cards}
    for gif in sorted(root.rglob("replay.gif")):
        gif_rel = gif.relative_to(root).as_posix()
        marker = f"<img src='{html.escape(gif_rel)}'"
        if any(marker in c for c in cards):
            continue
        cards.append(
            f"<section class='card'><h3>{html.escape(gif_rel)}</h3>"
            f"<img src='{html.escape(gif_rel)}' loading='lazy' /></section>"
        )

    body = "\n".join(cards)
    out_html.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>failure_cases GIFs</title>
<style>
body {{ font-family: sans-serif; margin: 1rem; background: #111; color: #eee; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1rem; }}
.card {{ background: #1b1b1b; padding: 0.75rem; border-radius: 8px; }}
.card img {{ width: 100%; height: auto; border-radius: 4px; }}
h1 {{ margin-bottom: 0.25rem; }}
</style></head><body>
<h1>failure_cases GIF gallery</h1>
<p>{len(cards)} GIF(s)</p>
<div class='grid'>
{body}
</div>
</body></html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {out_html} ({len(cards)} cards)")


if __name__ == "__main__":
    main()
