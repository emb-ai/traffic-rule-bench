The sequence of commands that are needed to be run to filter scenes: 

1. Filter catalog scenes by junction arm count (3- and/or 4-arm)
```
# 4-arm only (default) -> scenes/2.4_four_arm
python tools/filter_scenes/filter_catalog_by_junction.py

# 3-arm (T junction) only -> scenes/2.4_three_arm
python tools/filter_scenes/filter_catalog_by_junction.py --arms 3

# 4-arm preferred, else 3-arm -> scenes/2.4_junction_3_4arm
python tools/filter_scenes/filter_catalog_by_junction.py --arms 4 3
```

2. [Additional] Import filtered scenes and create png map for each scene for visual purposes 
```
python tools/filter_scenes/import_catalog_scenes.py 
```

3. Crop scenes around the picked junction
```
python tools/filter_scenes/crop_junction_scene.py
```

Now your scenes for sign 2.4 are ready
