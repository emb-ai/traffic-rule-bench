import os
import time
from tutorials.utils.tutorial_utils import visualize_nuplan_scenarios

if __name__ == '__main__':
    NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '/home/jovyan/shares/SR006.nfs2/smirnova/datasets/nuplan_test/data/nuplan-v1.1')
    NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '/home/jovyan/shares/SR006.nfs2/smirnova/datasets/nuplan_test/nuplan-maps-v1.0')
    NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '/home/jovyan/shares/SR006.nfs2/smirnova/datasets/nuplan_test/data/nuplan-v1.1/splits/trainval/2021.07.24.22.45.30_veh-26_03518_03604.db')
    NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')

    visualize_nuplan_scenarios(
        data_root=NUPLAN_DATA_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_root=NUPLAN_MAPS_ROOT,
        map_version=NUPLAN_MAP_VERSION,
        bokeh_port=8000
    )

    print("✅ Bokeh server running at: http://localhost:8000/")
    print("Press Ctrl+C to stop it.")

    # Keep script alive so the Bokeh server stays up
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
