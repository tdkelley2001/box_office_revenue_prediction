import os
from src.utils import ts


def save_cleaned_data(df, config, key="preprocessed_data_path"):
    base_path = config["output"][key]["path"]
    ts_path = f"{os.path.splitext(base_path)[0]}_{ts}.csv"
    
    os.makedirs(os.path.dirname(ts_path), exist_ok=True)
    df.to_csv(ts_path, index=False)
    print(f"Saved: {ts_path}")
    
    # Update pointer file dynamically from config
    pointer_path = config["output"][key]["pointer"]
    
    if pointer_path:
        with open(pointer_path, "w") as f:
            f.write(ts_path)
        print(f"Updated pointer: {pointer_path}")
