# plant2_ft_pipeline

PlanT2 finetuning on dumped frames: split → train → training-curve report.

Closed-loop eval does not live here. It is `traffic_bench/eval`, run as
`python -m traffic_bench.eval run …`.

## What is here

| file | purpose |
| --- | --- |
| `data/make_train_val_split_fv_experts_signs.py` | dumps → a train/val split of hardlinks |
| `train/run_plant2_finetune.py` | launches the finetune |
| `lib/env.py` | paths and interpreter |
| `lib/finetune.py` | builds the command and runs it through `shims/run_lit_finetune.py` |
| `tools/report_ft_metrics.py` | training curves out of the CSVLogger |

## Environment

No default points outside the checkout.

| variable | what it sets | default |
| --- | --- | --- |
| `SPLIT_SRCS` | split sources, `tag=/abs/path` separated by `;` | **required** |
| `SPLIT_OUT` | where to build the split | **required** |
| `ORACLE_ROOT` | a collection run: `<root>/<family>/experts/` | unset — the sign is read from the frames |
| `VERIFY_GZ` | `0` skips the readability check | `1` |
| `TRB_ROOT` | repository root | this checkout |
| `CKPT0` | checkpoint to start from | `<repo>/checkpoints/plant2_pretrain/epoch=029_final_3.ckpt` |
| `METRICS_ROOT` | where metrics are written | `<repo>/data/plant2_ft_metrics` |
| `PYTHON` | interpreter for training | the running one |

## Building a split

```bash
cd scripts/plant2_ft_pipeline
SPLIT_SRCS="speed_limit=$DUMP/speed_limit;min_speed=$DUMP/min_speed" \
SPLIT_OUT=$WORK/plant2_splits/speed \
ORACLE_ROOT=$WORK/traj_full/<run> \
PYTHONPATH=.:$TRB_ROOT python data/make_train_val_split_fv_experts_signs.py
```

The builder reads every frame back and drops routes that fail: a dump worker
killed mid-write leaves a `.json.gz` of the right name and size whose contents
are not gzip, and without this check the run dies with `BadGzipFile` inside a
DataLoader worker minutes into an epoch.

## Finetuning

```bash
SPLIT=$WORK/plant2_splits/speed DS=$SPLIT/train DS_VAL=$SPLIT/val \
CHECKPOINT_ADDON=speed CKPT_EVERY_N_EPOCHS=2 \
LEARNING_RATE=1e-4 MAX_EPOCHS=20 BATCH_SIZE=128 \
PATH_TARGET=future PATH_HORIZON_FRAMES=40 TS_LOOKAHEAD=1 \
PYTHONPATH=.:$TRB_ROOT python train/run_plant2_finetune.py
```

`TS_LOOKAHEAD=1` changes the target-speed label from the posted number to the
speed the expert actually drove. The difference is not small: labelled with the
plate, the model holds it as a setpoint and is over it on about half the in-zone
steps (compliance 0.000); labelled with the expert's speed, compliance is
0.95–0.99 on the ceiling plates.

Selecting a checkpoint by validation loss is unreliable here: in a measured run
it fell monotonically for all 20 epochs while closed-loop destination rate was
higher at epoch 7 than at epoch 19 (0.743 vs 0.661). Save checkpoints often and
pick by eval, not by `best_*`.

## Curves

```bash
PYTHONPATH=.:$TRB_ROOT python tools/report_ft_metrics.py $PLANT/log/ft_<addon>_1 --every 4
```
