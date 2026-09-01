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
- `densities.count` is every annotated car in the frame, kept for compatibility.
  What corresponds to "traffic around the ego" is `count_moving_r50`.

## Files

| file | what it does |
| --- | --- |
| `extract_nuplan_statistics.py` | the .db logs → all csv/json outputs; one function per statistic |
| `compare_nuplan_statistics.py` | two statistics directories → a markdown report of what moved |
| `calibrate_density_metadrive.py` | measures MetaDrive's density curve and solves for the levels matching nuPlan's p25/50/75 |
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

Density calibration is separate, because it needs MetaDrive rather than nuPlan:

```bash
python3 tools/nuplan_resample/calibrate_density_metadrive.py --out $OUT
```

## Output

`speeds.csv`, `acc_pos.csv`, `acc_neg.csv`, `following.csv`, `routes.csv`,
`densities.csv`, `lane_changes.csv`, `ego_routes.csv`, `metadrive_config.json`,
`density_calibration.json`, `statistics_report.json`.

`statistics_report.json` carries the provenance and the definition of each
statistic, so a set can be read back without this README.

Every threshold of the method is a constant at the top of
`extract_nuplan_statistics.py` — speed and acceleration cutoffs, the following
geometry, the density radius, the lane-change merge window.
