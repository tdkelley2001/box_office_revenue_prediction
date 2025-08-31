import os
import sqlite3
import pandas as pd

POINTER_FILE = "output/latest_preprocessed.txt"

def load_data(config):
    db_path = config["database"]["path"]
    table = config["database"]["table"]
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    conn.close()
    
    return df

def load_latest_preprocessed():
    if not os.path.exists(POINTER_FILE):
        return None
    with open(POINTER_FILE) as f:
        latest_file = f.read().strip()
    if not os.path.exists(latest_file):
        return None
    print(f"Loading latest preprocessed dataset: {latest_file}")
    return pd.read_csv(latest_file)
