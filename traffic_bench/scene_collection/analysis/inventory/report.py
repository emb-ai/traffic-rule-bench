"""Write README.md + summary.json for the harvest inventory experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traffic_bench.scene_collection.analysis.inventory.harvest import (
    HarvestSnapshot,
    summary_dict,
)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join((head, sep, body))


def write_report(
    snap: HarvestSnapshot,
    out_dir: Path,
    *,
    figures_rel: str = "figures",
) -> Path:
    """Write ``summary.json`` + ``README.md`` under the inventory package dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summary_dict(snap)
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    fam_rows = [[name, t["on_disk"]] for name, t in stats["families"].items()]
    total = sum(t["on_disk"] for t in stats["families"].values())
    fam_rows.append(["total", total])
    j_rows = [
        [
            shape,
            stats["junction_by_shape"].get(shape, 0),
            len(snap.train_ids.get(shape, [])),
            len(snap.test_ids.get(shape, [])),
        ]
        for shape in ("T", "X", "O")
    ]

    fr = figures_rel.rstrip("/")
    md = f"""# Harvest inventory

Sign-free cropped SUMO nets on disk (`maps/crops/`).
Per-sign quota **N** = 80 train + 20 test is sampled later (`assign`);
it is not the harvest size.

## Reproduce

```bash
python -m traffic_bench.scene_collection analysis inventory
```

Outputs: this README, `summary.json`, PNGs under [`{fr}/`]({fr}/).

## Inventory

{_md_table(["Family", "On disk"], fam_rows)}

![Harvest inventory]({fr}/inventory.png)

## Junctions (T / X / O)

{_md_table(["Shape", "On disk", "Train ids", "Test ids"], j_rows)}

Place identity (`junction_id` / `scene_id`) is split 80/20 *before*
allocation to signs, stratified by shape.

![Junction topology and split]({fr}/junction_shapes.png)

## Dual-path atoms

Each cell is one `(baseline, compliant)` slot among {{l, s, r}}.
The same junction may contribute at most one atom per slot.

![Dual-path slot counts and detour gain]({fr}/dual_path.png)

## Segments (corridors)

Incoming edges cropped so the scene ends 10 m before the junction.
Gates: length ≥ 150 m; **straight** chord/arc ≥ 0.99; **curved** in [0.97, 0.99).

- distinct OSM ways: {stats["segment"]["n_osm_ways"]}
- `pass_right_ok`: {stats["segment"]["pass_right_ok"]}
- `pass_left_ok`: {stats["segment"]["pass_left_ok"]}

![Segment length, straightness, lanes]({fr}/segment_diversity.png)

## Geographic coverage

Points are cropped nets on disk. Dual-path locations are unique parent junctions.

![Geographic coverage]({fr}/geo_coverage.png)

## Example crops

One net per cell. Labels are `family/group` and `scene_id`.

![Example cropped maps]({fr}/examples.png)
"""
    path = out_dir / "README.md"
    path.write_text(md, encoding="utf-8")
    return path
