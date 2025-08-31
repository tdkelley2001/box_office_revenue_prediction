import os
from src.utils import ts

POINTER_PREPROCESSED = "output/latest_paths/latest_preprocessed.txt"
POINTER_MODEL = "output/latest_paths/latest_model_ready.txt"

def save_cleaned_data(df, config, key="preprocessed_data_path"):
    base_path = config["output"][key]
    
    root, ext = os.path.splitext(base_path)
    output_path = f"{root}_{ts}{ext}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Update pointer files
    if key == "preprocessed_data_path":
        with open(POINTER_PREPROCESSED, "w") as f:
            f.write(output_path)
    elif key == "model_data_path":
        with open(POINTER_MODEL, "w") as f:
            f.write(output_path)
