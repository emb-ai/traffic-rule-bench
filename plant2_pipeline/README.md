# plant2_pipeline

One command per stage of the loop that produces a PlanT2 sign-compliance model:

```
dump  ->  split  ->  train  ->  eval
```

```bash
python -m plant2_pipeline all --name my_run --gpu 0
```

Each stage can also be run alone, in order:

```bash
python -m plant2_pipeline dump   --name my_run
python -m plant2_pipeline split  --name my_run
python -m plant2_pipeline train  --name my_run --gpu 0 --epochs 24
python -m plant2_pipeline eval   --name my_run --gpu 0
python -m plant2_pipeline report --name my_run          # re-print the table
```

Everything for one run lands under `plant2_pipeline/runs/<name>/`:
`dump/`, `split/`, `logs/`, `eval/summary.txt`. Checkpoints go where PlanT2
puts them, `plant2/PlanT/checkpoints_ft/<name>/`.

## What each stage does

**dump** replays the privileged expert over the training scenes and records
what the model trains on: object boxes, measurements, and a semantic BEV per
step. One directory per benchmark, so either can be re-dumped alone.

**split** divides by *scene* (not by route — the old split put the same scene
in train and val, which made `val/loss_path` meaningless), then rebuilds the
path target from the trajectory the expert actually drove. The rebuild is not
optional: the dumper writes the same array into `route_original` (the model's
input) and `route` (its target), so copying the input is exactly optimal and
that is what the model learns. Heavy files are symlinked, so the source dump —
shared with other people's experiments — is never modified.

**train** fine-tunes from the CARLA pretrain checkpoint. That checkpoint on its
own scores 0.000 on detour and hits the obstacle in 100% of runs, so it is a
real baseline.

**eval** scores the checkpoint on every benchmark and prints one table.

## Things that will silently give you wrong numbers

These are not hypothetical; each one produced a plausible, wrong result at some
point, and each is now handled in `config.py` rather than left to the caller.

| variable | why it matters |
|---|---|
| `PLANT2_SIGN_RANGE_M` | sign visibility radius. Must match between dump and eval, or the model meets a world at eval it never saw in training. 30 m is too short for 4.2.x — the sign becomes visible after the deadline for the lane change. |
| `PLANT2_DUMP_SIGN_CLASSES` | which PDD codes may become `x_objs` tokens. A code not listed is invisible. Dumping a joint model with only `4.2.x` gives frames with no stop sign at all. |
| `PLANT2_YLEFT` | the dump writes route/path/waypoint targets as y=LEFT and object boxes as y=RIGHT. With inference told otherwise, route following still works but every obstacle is passed on the wrong side: honest compliance 0.000 vs 0.879 on the same weights. |

## Reading the metrics

`metrics.py` is the single definition. For the detour signs it requires
`reached_zone` in addition to zero violation steps, because the benchmark's own
`sign_compliance` can be satisfied without performing the manoeuvre — most
often by ending the episode before the zone, since violations are only
evaluated inside it. The checkpoint with the best headline number (0.814)
reached the zone in 42% of runs. `in_zone_total_steps` does not reveal this: it
counts the approach lookahead as in-zone.

`metrics.py` refuses to report a number for an eval directory produced before
that flag existed, rather than printing one that cannot be compared.

Two asymmetries worth knowing when quoting numbers:

- **Detour and stop run different code.** `per_sign_bench/run_benchmark.py` and
  `priority_bench/run_benchmark.py` are separate files; a fix to one does not
  reach the other. The stop path also enriches manifest rows with the
  experiment config (the NPC convoy at the stop line), so calling its runner
  directly instead of through `eval_pipeline.py` silently produces an easier
  scene.
- **The stop denominator is 42, not 50.** Eight test rows fail to build a scene
  (`Invalid route: spawn and destination are the same or unreachable`). Over
  all 50 the same run scores 0.620 / 0.720 instead of 0.738 / 0.857.
- **`sign_compliance` for 2.5 measures giving way, not stopping.** `StopSign`
  extends `YieldSign` and checks the yield rule first; in a full test run all
  11 violations were yield violations and none were stop-line violations. The
  stopping itself is good — compliant runs reach 0.0–0.24 m/s before the line —
  but it is not what the headline number is counting.

## Choosing a checkpoint

Always `last`, never `best`. `best` is selected by `val/loss_all`, which is
dominated by the ego-speed term: that term rises monotonically while the path
loss plateaus, so the "best" epoch is an almost-untrained one. Measured, `best`
checkpoints score stop compliance 1.000 at success 0.000 and efficiency 8.8 —
a car that barely moves satisfies a stop sign perfectly.

## Layout

```
config.py    paths, benchmark definitions, the Experiment dataclass
shell.py     subprocess helpers; non-zero exit always raises
stages.py    dump / split / train / evaluate
metrics.py   how a run is scored, and why it is scored that way
__main__.py  the CLI
```

No `try/except` anywhere in this package. A stage either produces its output or
stops with the failing command printed.
