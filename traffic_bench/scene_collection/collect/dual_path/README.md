# collect/dual_path/ — two-route crops

A dual-path crop is a **different net** (union of a short baseline and a longer compliant route), not a tag on a T/X junction crop.

- `graph.py` — discover `(baseline_dir, compliant_dir)` atoms on the city net
- `roles.py` — PDD code → slots (`r_s`, `l_r`, …); 5.7 stem / carriageway gates
- `stem.py` — T-junction stem approach
- `crop.py` — write `maps/crops/dual_path/{T,X}/{slot}/<scene_id>/`

Used by 3.1, 3.18, 4.1, 5.7.
