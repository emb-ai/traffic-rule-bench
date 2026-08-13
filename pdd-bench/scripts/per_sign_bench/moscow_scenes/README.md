# moscow_scenes — Moscow map harvest (junction-only + dual_path)

Sign-free scene harvest for Moscow. Scenes are keyed by **SUMO `junction_id`**
(or roundabout fingerprint), not by traffic-sign database IDs.

Formerly `moscow_junctions`. Renamed because the pool now includes **dual_path**
crops (path-union bbox), not only junction-only stubs.

> Naming: use **dual_path**, not “detour” (PDD **4.2.1** is the detour sign).

## Layout

```
moscow_scenes/
├── README.md
├── raw/ … nets/ … index/
├── lib/                         # dual_path discovery + roles + stem
├── scenes/
│   ├── {T,X,O}/                 # junction-only (~80 m arms)
│   └── dual_path/{T,X}/{slot}/  # path-union crops (l_s, l_r, …)
├── splits/
│   ├── signs.yaml / signs.json
│   ├── train_ids.json / test_ids.json
│   └── sign_allocations.json
└── scripts/
    ├── build_net.py / enumerate_junctions.py / crop_scenes.py
    ├── crop_dual_path_scenes.py
    ├── make_junction_split.py / allocate_sign_scenes.py
    └── run_pipeline.py
```

## Two crop kinds

| Kind | Path | Used by |
|------|------|---------|
| `junction` | `scenes/{T,X,O}/` | 2.x, 3.2, 4.3, … |
| `dual_path` | `scenes/dual_path/{T,X}/{slot}/` | 5.7, 3.18, 4.1, 3.1 |

### Dual-path slots

Atomic slot = exact `(baseline_dir, compliant_dir)` among `{l,s,r}`:

`l_s` `l_r` `r_s` `r_l` `s_l` `s_r`

**Pool size:** at most **500** scenes per `(shape, slot)` (default `--max-per-slot 500`).

Same shared-pool idea as junction `T/X/O` (many signs reuse train/test maps).
Do **not** size the pool as one sign’s `n_train+n_test`. Cap exists only because
each dual_path crop is a path-union netconvert (heavier than junction-only); 500
is roughly X-inventory scale per bucket and leaves headroom under 80/20 split and
slot/stem filters. At most **one** atom per junction per slot; shuffle `--seed 42`;
early-stop when buckets are full.

Sign → slots (allocate filter): see `lib/roles.py` (`SIGN_TO_SLOTS`).
Extra gates: **5.7** → T only, `ego_is_t_stem`, `carriageway_pair`.

## Pipeline

```bash
cd traffic-rule-bench/pdd-bench/scripts/per_sign_bench/moscow_scenes

# Junction-only harvest (existing)
python scripts/run_pipeline.py --skip-download --skip-netconvert

# Dual-path harvest (among the same enumerated T/X junctions)
# Default: 500 scenes per (shape, slot) — shared pool, not one-sign quota
python scripts/crop_dual_path_scenes.py --max-per-slot 500 --skip-existing
# smoke:  python scripts/crop_dual_path_scenes.py --max-per-slot 5 --discover-only
# subset: python scripts/crop_dual_path_scenes.py --max-per-slot 500 --slots l_s,l_r

python scripts/make_junction_split.py
python scripts/allocate_sign_scenes.py
```

## Train / test

1. Shared pool — not owned by any sign.
2. Prefer split by **`junction_id`** so junction-only and dual_path crops of the
   same junction stay on the same side (no road leakage across train/test).
3. Signs sample from the matching `crop_kind` with shape / slot filters.

Quotas: `splits/signs.yaml` (`n_train`, `n_test`, `seed`, `crop_kind`, …).

## Provenance

BBBike Moscow OSM extract → `nets/moscow.net.xml`. See `raw/DOWNLOAD_META.json`.
