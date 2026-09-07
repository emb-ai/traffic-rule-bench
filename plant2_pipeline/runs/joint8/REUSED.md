plan / dump / split are symlinks to runs/joint7.

Nothing in them depends on the sign radius: the dump was recorded at
DUMP_SIGN_RANGE_M (90 m) and the split only rebuilds the path target. What does
depend on it is the dataset CACHE, because dataset.py applies
model.training.sign_range_m when it loads a frame and the cache stores the
result — reusing joint7's cache would silently feed 30 m samples into a 90 m
run. joint8 therefore gets its own cache directory, derived from the run name.
