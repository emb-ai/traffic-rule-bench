# nuplan_resample

Recomputes `nuplan_statistics` from the nuPlan v1.1 mini split. Those files are
what `traffic_bench/eval/engine/traffic/nuplan_sampler.py` samples NPC speeds,
accelerations, following distances, route lengths and traffic density from, so
everything the benchmark's traffic does traces back to them.

## Why it was recomputed

The previous set was produced under different definitions, and one of them was
wrong by a wide margin: the lane-change rate counted every transition between
lane polygons, including driving forward from one stretch of road onto the next.
Only a change of lane *within* one lane group is a lane change. The rate it fed
into `IDMPolicy.LANE_CHANGE_FREQ` was overstated by roughly 50x, so NPCs changed
lanes far more often than nuPlan traffic does.

Two more definitions are worth knowing when reading the numbers:

- `routes.distance` is the path of a track **inside the annotation window**
  (~50-80 m around the ego), not the length of its trip. For a real route length
  use `ego_routes.csv`, which is not cut off by the window.
- density is counted within `DENSITY_RADIUS` = 150 m of the ego and divided by
  the lanes of the ego's road, because that is the quantity a simulator can be
  asked to reproduce: `traffic_density` fills spawn slots per lane, so a
  per-frame total is not comparable to it. The column to read is
  `count_moving_r150_per_lane`; `count_r150` is the same set without the
  moving filter, and it equals the old `count` exactly -- nuPlan's annotation
  window never reaches past 150 m, so the earlier 50 m radius was the only thing
  the two definitions differed by.

## Files

| file | what it does |
| --- | --- |
| `extract_nuplan_statistics.py` | the .db logs → all csv/json outputs; one function per statistic |
| `compare_nuplan_statistics.py` | two statistics directories → a markdown report of what moved |
| `calibrate_density_sumo.py` | sweeps `traffic_density` on the benchmark's own SUMO scenes and quantile-matches the result against nuPlan |
| `calibrate_density_metadrive.py` | the previous calibration, on PG maps under MetaDrive's trigger manager; superseded, kept for provenance |
| `plot_density_calibration.py`, `plot_old_vs_new.py` | figures |
| `run_resample_mini.sh` | extraction and comparison end to end |

## Running it

Needs the nuPlan mini split and the maps unpacked (`nuplan-v1.1_mini.zip`,
`nuplan-maps-v1.0.zip`):

```bash
NUPLAN_ROOT=/path/to/nuplan OUT=/path/to/nuplan_statistics_v2 \
    bash tools/nuplan_resample/run_resample_mini.sh
```

Set `OLD=/path/to/previous_statistics` to also get `comparison_report.md`.
Maps are optional — without them lane changes are skipped and everything else
still computes.

Density calibration is separate, because it measures the simulator rather than
nuPlan. It runs the benchmark, so it needs a node that can:

```bash
python3 tools/nuplan_resample/calibrate_density_sumo.py sweep \
    --manifest data/runs_v2/speed_limit/test --work $TMPDIR/dens_sweep

python3 tools/nuplan_resample/calibrate_density_sumo.py fit \
    --work $TMPDIR/dens_sweep \
    --densities-csv traffic_bench/eval/engine/traffic/nuplan_statistics/densities.csv.gz \
    --out traffic_bench/eval/engine/traffic/nuplan_statistics/density_calibration_sumo.json \
    --plot reports/density_calibration_sumo.png
```

`fit` writes the file the benchmark reads at scene-expansion time. Replacing it
changes the traffic of every scene generated afterwards, so it belongs in a
commit of its own.

## Output

`speeds.csv`, `acc_pos.csv`, `acc_neg.csv`, `following.csv`, `routes.csv`,
`densities.csv`, `lane_changes.csv`, `ego_routes.csv`, `metadrive_config.json`,
`density_calibration.json`, `statistics_report.json`.

`statistics_report.json` carries the provenance and the definition of each
statistic, so a set can be read back without this README.

Every threshold of the method is a constant at the top of
`extract_nuplan_statistics.py` — speed and acceleration cutoffs, the following
geometry, the density radius, the lane-change merge window.

## What ships in the repository

The current set lives where the benchmark reads it, not in a separate archive:
`traffic_bench/eval/engine/traffic/nuplan_statistics/`. `NuPlanSampler` draws
NPC speeds, accelerations and following distances from those files at run time,
and `traffic_density_levels.py` reads the calibration from the same directory.

It is stored gzipped -- 46 MB instead of 113 -- and `pandas.read_csv` opens
`.csv.gz` with no argument:

```python
import pandas as pd
d = pd.read_csv("traffic_bench/eval/engine/traffic/nuplan_statistics/densities.csv.gz")
d["count_moving_r150_per_lane"].describe()
```

That directory is held against the repository's blanket `*.csv` / `*.json`
ignores by explicit negations at the end of `.gitignore`; a new file there needs
the negation to cover it, or it is silently untracked -- which is how the whole
set came to be missing from git in the first place.
