import os
import sqlite3
import pandas as pd


def load_data(config):
    db_path = config["database"]["path"]
    table = config["database"]["table"]
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    conn.close()
    
    return df


def load_latest_from_pointer(pointer_path):
    if not os.path.exists(pointer_path):
        return None
    with open(pointer_path) as f:
        path = f.read().strip()
    if not os.path.exists(path):
        return None
    print(f"Loading dataset from pointer: {path}")
    return pd.read_csv(path)

