Download scenes for a specific sign

```
python build_sign_scenes_from_osm_async.py \
  --csv ../../data/data-cleaned.csv \
  --sign-types "4.3" \
  --scenes_dir ../../scenes \
  --max-signs-per-type 250 \
  --max-concurrent 5
```

which writes:

```
pdd-bench/scenes/4.3/sign_72752/
├── meta.json
└── sign_72752.net.xml
```