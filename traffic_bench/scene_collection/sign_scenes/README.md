# sign_scenes/ — per-sign folders under data/scenes/

Takes allocations from `assign/` and places maps into `data/scenes/<sign>/`.

| Path | Role |
| --- | --- |
| [`materialize/`](materialize/README.md) | Symlink or copy allocated crops into the sign folder |
| [`prepare/`](prepare/README.md) | Sign-specific surgery after materialize (currently 5.19 zebra) |
| [`filter/`](filter/README.md) | Reject unusable maps, visual review, keep/reject JSON |
