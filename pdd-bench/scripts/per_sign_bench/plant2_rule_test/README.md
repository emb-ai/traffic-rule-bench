# plant2_rule test eval (for colleague)

Evaluate a **plant2_rule** checkpoint on the fixed **test** splits of selected PDD signs.
Manifests are already prepared; this package only runs eval and collects metrics.

## Signs


| Label         | Source catalog                                                                                                                            | rows |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| 2.1           | `main_sign/.../final_metrics_v1/catalog_test20.jsonl`                                                                                     | 153  |
| 2.3.1-2.3.3   | `secondary_sign/.../final_metrics_v1/catalog_test20.jsonl`                                                                                | 183  |
| 2.4           | `yield_sign/.../final_metrics_v1/catalog_test20.jsonl`                                                                                    | 249  |
| 2.5           | `stop_sign/.../final_metrics_v1/catalog_test20.jsonl`                                                                                     | 153  |
| 3.1-3.2       | `no_entry_signs/.../combined/catalog_test20.jsonl`                                                                                        | 303  |
| 4.3           | `roundabout_sign/.../test_metrics/test20_batch/4_3/eval_out/input_manifest.jsonl`                                                         | 210  |
| 5.7.1-5.7.2   | `_batch_test_manifests/test20_batch/catalog_test_5_7_1_5_7_2.jsonl` (no clean `test_metrics/`; same 216 test rows, nets already prefixed) | 216  |
| 5.15.1-5.15.2 | `lane_direction_signs/.../test_metrics/test20_batch/5_15_1_5_15_2/eval_out/input_manifest.jsonl`                                          | 185  |
| 5.19          | `crosswalk_sign/.../test_metrics/test20_batch/5_19/eval_out/input_manifest.jsonl`                                                         | 170  |


Symlinks live in `[manifests/](manifests/)`.

## 1. Setup

```bash
conda activate zinkovich-sdc   
cd pdd-bench/scripts/per_sign_bench/plant2_rule_test
```

## 2. Run eval on your checkpoint

Full test:

```bash
python eval_checkpoint_on_test.py \
    --model-paths plant2_rule:/ABS/PATH/TO/your_checkpoint.ckpt \
    --jobs 8 \
    --keep-going
```

Only first **N** rows from each sign's manifest (`--n-scenes` / `--max-scenes`):

```bash
python eval_checkpoint_on_test.py \
    --model-paths plant2_rule:/ABS/PATH/TO/your_checkpoint.ckpt \
    --n-scenes 2 \
    --only 2.1,4.3,3.1-3.2
```

Dry-run (print commands only):

```bash
python eval_checkpoint_on_test.py \
    --model-paths plant2_rule:/ABS/PATH/TO/your_checkpoint.ckpt \
    --dry-run
```

Outputs go to:

```
plant2_rule_test/output/<run-name>/<sign_slug>/
  input_catalog_test.jsonl
  source_catalog.txt
  eval_out/
    metrics_per_episode.csv
    reports/report_cumulative.md
    reports/cumulative.json
```

Default `--run-name` is `plant2_rule_test`.

## 3. Summarize all signs

`eval_checkpoint_on_test.py` writes markdown summaries automatically at the end.
To regenerate:

```bash
python summarize_reports.py --run-name plant2_rule_test
```

Writes under `output/<run-name>/_summary/`:

- `run_summary.md` — run status (ok / failed / skipped) + metrics table
- `summary.md` — per-sign metrics table only
- `summary.csv` — same metrics as CSV

## Notes

- **Do not regenerate manifests** — use the packaged test splits above.
- `--n-scenes N` takes the first N rows of each job's catalog (not unique `scene_id`s).
- Per-sign benches write into this folder’s `output/`; each job calls that bench’s `eval_pipeline.py`.
- Optional: `--policies plant2_rule,plant2` with matching `--model-paths` entries.

