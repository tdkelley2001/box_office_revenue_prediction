import sqlite3
import pandas as pd

def load_movie_data(config):
    db_path = config["database"]["path"]
    table = config["database"]["table"]
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    conn.close()
    
    return df