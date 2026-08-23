# `sign_scenes/`

Materializes and validates the per-sign scene datasets under `data/scenes/<sign>/`.

**Pipeline:**

`allocations → materialize → prepare → filter → data/scenes/<sign>/`


| Path           | Purpose                                                            |
| -------------- | ------------------------------------------------------------------ |
| `materialize/` | Materialize allocated crops as symlinks or copies                  |
| `prepare/`     | Apply sign-specific scene modifications (currently 5.19 crosswalk) |
| `filter/`      | Review scenes and reject unusable maps                             |


The final scene folders are stored in:

```
data/scenes/<sign>/
```

For packing and publishing the complete dataset, see `../publish/`.