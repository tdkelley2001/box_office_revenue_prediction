import os
from src.config import config
from src.loader import load_data, load_latest_from_pointer
from src.preprocessing import preprocess_data, run_feature_engineering
from src.saver import save_cleaned_data
from src.eda import run_eda_round1, run_eda_round2
from src.indicator_optimization import optimize_indicators_by_n
from src.postprocessing import postprocess_data
from src.sfa import run_sfa
import pandas as pd

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load config
config = config("config/config.yaml")


# ------------------------
# Step 1: Load preprocessed dataset
# ------------------------
movies_cleaned = load_latest_from_pointer(config["output"]["preprocessed_data"]["pointer"])
if movies_cleaned is None:
    print("No preprocessed dataset found, running preprocessing...")
    movies_raw = load_data(config)
    movies_cleaned = preprocess_data(movies_raw, config)
    if config["save_preprocessed_data"]:
        save_cleaned_data(movies_cleaned, config, key="preprocessed_data")


# ------------------------
# Step 2: Run Round 1 EDA
# ------------------------
if config.get("do_round1_eda", True):
    run_eda_round1(movies_cleaned, config)


# ------------------------
# Step 3: Optimize Indicators
# ------------------------
if config.get("do_indicator_optimization", True):
    optimize_indicators_by_n(movies_cleaned, config)


# ------------------------
# Step 4: Feature Engineering
# ------------------------
movies_fe = load_latest_from_pointer(config["output"]["feature_engineered_data"]["pointer"])
if movies_fe is None:
    print("No feature-engineered dataset found, running feature engineering...")
    movies_fe = run_feature_engineering(movies_cleaned, config)
    if config["save_feature_engineered_data"]:
        save_cleaned_data(movies_fe, config, key="feature_engineered_data")


# ------------------------
# Step 5: Run Round 2 EDA (on FE dataset)
# ------------------------
if config.get("do_round2_eda", True):
    run_eda_round2(movies_fe, config)


# ------------------------
# Step 6: Postprocessing / Modeling Prep
# ------------------------
movies_model = load_latest_from_pointer(config["output"]["model_data"]["pointer"])
if movies_model is None:
    print("No model dataset found, running postprocessing...")
    movies_model = postprocess_data(movies_fe, config)
    if config["save_modeling_data"]:
        save_cleaned_data(movies_model, config, key="model_data")


# ------------------------
# Step 7: Run Single-Factor Analysis
# ------------------------
if config.get("do_sfa", True):
    train_data = movies_model[movies_model["split"] == "train"]
    run_sfa(train_data, config)


# ------------------------
# Step 8: Run Multi-Factor Analysis
# ------------------------
# TODO: Develop MFA


print("Run complete!")