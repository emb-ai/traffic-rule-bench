"""Write inventory.md + summary.json next to the figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traffic_bench.scene_collection.analysis.inventory import HarvestSnapshot, summary_dict


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join((head, sep, body))


def write_report(snap: HarvestSnapshot, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summary_dict(snap)
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    fam_rows = [
        [name, t["index"], t["on_disk"], f"{100 * t['coverage']:.1f}%"]
        for name, t in stats["families"].items()
    ]
    j_rows = [
        [
            shape,
            stats["junction_by_shape"]["index"].get(shape, 0),
            stats["junction_by_shape"]["on_disk"].get(shape, 0),
            len(snap.train_ids.get(shape, [])),
            len(snap.test_ids.get(shape, [])),
        ]
        for shape in ("T", "X", "O")
    ]

    md = f"""# Harvest inventory

Sign-free crops from the BBBike Moscow OSM extract. **P** is the catalog
(`maps/index/`). **H** is cropped SUMO nets on disk (`maps/crops/`).
Per-sign quota **N** = 80 train + 20 test is sampled later (`assign`);
it is not the harvest size.

Regenerate:

```bash
python -m traffic_bench.scene_collection analysis
```

## Inventory

{_md_table(["Family", "P (index)", "H (on disk)", "H / P"], fam_rows)}

![Harvest inventory](inventory.png)

## Junctions (T / X / O)

{_md_table(["Shape", "Index", "On disk", "Train ids", "Test ids"], j_rows)}

Place identity (`junction_id` / `scene_id`) is split 80/20 *before*
allocation to signs, stratified by shape.

![Junction topology and split](junction_shapes.png)

## Dual-path atoms

Each cell is one `(baseline, compliant)` slot among {{l, s, r}}.
The same junction may contribute at most one atom per slot.

![Dual-path slot counts and detour gain](dual_path.png)

## Segments (corridors)

Incoming edges cropped so the scene ends 10 m before the junction.
Gates: length ≥ 150 m; **straight** chord/arc ≥ 0.99; **curved** in [0.97, 0.99).

- distinct OSM ways: {stats["segment"]["n_osm_ways"]}
- `pass_right_ok`: {stats["segment"]["pass_right_ok"]}
- `pass_left_ok`: {stats["segment"]["pass_left_ok"]}

![Segment length, straightness, lanes](segment_diversity.png)

## Geographic coverage

Points are harvest candidates (index), not the N-per-sign protocol sample.
Dual-path locations are unique parent junctions.

![Geographic coverage](geo_coverage.png)

## Example crops

One net per cell. Labels are `family/group` and `scene_id`.

![Example cropped maps](examples.png)
"""
    path = out_dir / "inventory.md"
    path.write_text(md, encoding="utf-8")
    return path
