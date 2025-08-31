import os
from src.config import config
from src.loader import load_data, load_latest_preprocessed
from src.preprocessing import preprocess_data
from src.saver import save_cleaned_data
from src.eda import run_eda
import pandas as pd

# Load config
config = config("config/config.yaml")

# Try loading latest preprocessed
movies_cleaned = load_latest_preprocessed()
if movies_cleaned is None:
    print("No preprocessed dataset found, running preprocessing...")
    movies_raw = load_data(config)
    movies_cleaned = preprocess_data(movies_raw, config)
    save_cleaned_data(movies_cleaned, config, key="preprocessed_data_path")

# Run EDA
run_eda(movies_cleaned, config)


