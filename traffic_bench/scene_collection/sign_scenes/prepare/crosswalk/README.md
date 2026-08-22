# prepare/crosswalk/ — mid-block zebra (PDD 5.19)

Injects one pedestrian crossing in the **middle** of a copied segment, in place
under `data/scenes/crosswalk/`. Does not write `_cw_<pos>` variants.

- `add_zebra.py` — walk live scene dirs, skip nets that already have a crossing
- `inject.py` — split the edge, add sidewalks, `netconvert` crossing
