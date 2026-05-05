import os
import sqlite3
import numpy as np
import pandas as pd

def search_token_in_db(db_path, token):
    """Search for a given lidarpc_token in all tables of a .db file."""
    sample_db = sqlite3.connect(db_path)

    # Get list of tables
    tokens = pd.read_sql_query("SELECT lidar_token FROM lidar_pc;", sample_db)
    lidar_tokens = np.unique(tokens.values)
    for lidar_token in lidar_tokens:
        if lidar_token.hex() == token:
            sample_db.close()
            return True

    return False

def search_token_in_dir(data_dir, lidarpc_token):
    """Search all .db files in directory for a lidarpc_token."""
    results = []

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".db"):
                db_path = os.path.join(root, f)
                if search_token_in_db(db_path, lidarpc_token):
                    results.append((db_path))

    if results:
        print("\nFound token in:", results)
    else:
        print("❌ Token not found in any .db file.")


if __name__ == "__main__":
    # Example usage
    data_dir = "/home/jovyan/shares/SR006.nfs2/smirnova/datasets/nuplan_test/data/nuplan-v1.1/splits/test"  # change this
    lidarpc_token = '26ed9ba5470e5362'  # your token

    search_token_in_dir(data_dir, lidarpc_token)
