import pandas as pd
from sklearn.model_selection import train_test_split

def winsorize_df(df, config):
    """
    Winsorize numeric columns based on thresholds provided in config.
    Parameters:
        df: DataFrame
        config: configuration dictionary
    Returns:
        df with winsorized columns
    """
    for col, threshold in config["postprocessing"]["winsorization"].items():
        if col not in df.columns:
            continue
        if threshold < 1.0:  # Only apply if threshold < 1.0
            lower = df[col].quantile(1 - threshold)
            upper = df[col].quantile(threshold)
            df[col] = df[col].clip(lower, upper)
    return df


def impute_missing(df, numeric_cols, categorical_cols, 
                   numeric_method="median", categorical_method="most_frequent"
):
    """
    Impute missing values for numeric and categorical columns.
    Parameters:
        df: DataFrame
        numeric_cols: list of numeric columns
        categorical_cols: list of categorical/indicator columns
        numeric_method: "median" or "mean"
        categorical_method: "most_frequent"
    Returns:
        df with imputed values
    """
    df_proc = df.copy()
    
    # Numeric
    for col in numeric_cols:
        if df_proc[col].isnull().any():
            if numeric_method == "median":
                df_proc[col].fillna(df_proc[col].median(), inplace=True)
            elif numeric_method == "mean":
                df_proc[col].fillna(df_proc[col].mean(), inplace=True)
            else:
                raise ValueError(f"Unsupported numeric imputation method: {numeric_method}")
    
    # Categorical / indicator
    for col in categorical_cols:
        if df_proc[col].isnull().any():
            if categorical_method == "most_frequent":
                df_proc[col].fillna(df_proc[col].mode()[0], inplace=True)
            else:
                raise ValueError(f"Unsupported categorical imputation method: {categorical_method}")
    
    return df_proc


def split_train_test(df, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    Parameters:
        df: DataFrame
        test_size: fraction of data to use for testing
        random_state: for reproducibility
    Returns:
        df
    """
    df["split"] = "train"
    test_idx = df.sample(frac=test_size, random_state=random_state).index
    df.loc[test_idx, "split"] = "test"
    return df


def postprocess_data(df, config):
    """
    Wrapper for all postprocessing steps: winsorization, missing value imputation, train/test split.
    Parameters:
        df: DataFrame to process
        config: configuration dictionary, must include:
            - postprocessing.winsorization: dict {col: threshold}
            - postprocessing.numeric_missing: list of numeric columns to impute
            - postprocessing.categorical_missing: list of categorical columns to impute
            - postprocessing.train_test_split: dict with test_size, random_state
    Returns:
        dict: {
            "full": fully postprocessed DataFrame,
            "train": training set,
            "test": test set
        }
    """
    df_proc = df.copy()
    # ---------- Winsorization ----------
    df_proc = winsorize_df(df_proc, config)

    # ---------- Missing Imputation ----------
    numeric_cols = config["postprocessing"]["winsorization"].items()
    categorical_cols = config["postprocessing"].get("categorical_vars", [])
    df_proc = impute_missing(df_proc, numeric_cols, categorical_cols)

    # ---------- Train/Test Split ----------
    split_cfg = config["postprocessing"].get("train_test_split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    df_proc = split_train_test(df_proc, test_size=test_size, random_state=random_state)

    return df_proc
