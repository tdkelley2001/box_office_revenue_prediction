import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def parse_json_columns(df, cols):
    """Parse JSON-like string columns into lists of names."""
    for col in cols:
        def parse(x):
            try:
                return [g['name'] for g in ast.literal_eval(x)]
            except:
                return []
        df[f"{col}_list"] = df[col].apply(parse)
    return df


def process_dates(df, cols):
    """Ensure dates are datetime and extract useful date parts from release_date."""
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['release_year'] = df[col].dt.year
    df['release_month'] = df[col].dt.month
    df['release_quarter'] = df[col].dt.quarter
    return df


def drop_columns(df, cols):
    # Drop unusable colums
    df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

    return df


def preprocess_data(df, config):
    raw_vars = config["raw_vars"]

    # Parse JSON-like columns
    df = parse_json_columns(df, raw_vars["parse_cols"])

    # Process dates
    df = process_dates(df, raw_vars["date_cols"])

    # Drop unneeded columns
    df_filtered = drop_columns(df, raw_vars["drop_cols"])

    return df_filtered


def create_indicators(
    df: pd.DataFrame,
    var: str,
    top_n: int = None,
    cumulative_pct: float = None,
    list_col: bool = True,
    min_count: int = None
):
    """
    Create indicator columns for a categorical or list variable.

    Parameters:
        df: DataFrame
        var: Column name
        top_n: Keep top N levels (optional)
        cumulative_pct: Keep levels until cumulative coverage (optional)
        list_col: If True, the column contains lists
        min_count: Minimum number of occurrences for a level to be included (optional)

    Returns:
        df with new indicator columns
    """
    if var not in df.columns:
        return df

    # Flatten values for list columns
    if list_col:
        all_items = df[var].explode()
    else:
        all_items = df[var]

    counts = all_items.value_counts()

    # Apply min_count filter if specified
    if min_count is not None:
        counts = counts[counts >= min_count]

    # Determine which levels to keep
    if top_n is not None:
        keep_levels = counts.head(top_n).index.tolist()
    elif cumulative_pct is not None:
        cumsum = counts.cumsum() / counts.sum()
        keep_levels = cumsum[cumsum <= cumulative_pct].index.tolist()
    else:
        keep_levels = counts.index.tolist()

    # Create indicators
    for level in keep_levels:
        if list_col:
            df[f"{var}_{level}"] = df[var].apply(lambda x: int(level in x) if isinstance(x, list) else 0)
        else:
            df[f"{var}_{level}"] = (df[var] == level).astype(int)

    # Optional: create 'Other' column
    df[f"{var}_Other"] = 0
    if list_col:
        df[f"{var}_Other"] = df[var].apply(
            lambda x: int(any(i not in keep_levels for i in x)) if isinstance(x, list) else 0
        )
    else:
        df[f"{var}_Other"] = (~df[var].isin(keep_levels)).astype(int)

    return df


def add_basic_features(df):
    # Revenue log
    if "budget" in df.columns:
        df["budget_log"] = np.log(df["budget"])
    # Homepage indicator
    if "homepage" in df.columns:
        df["has_homepage"] = df["homepage"].notna().astype(int)
    
    # Title length
    if "original_title" in df.columns:
        df["title_length"] = df["original_title"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Overview length
    if "overview" in df.columns:
        df["overview_length"] = df["overview"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Tagline indicator
    if "tagline" in df.columns:
        df["has_tagline"] = df["tagline"].notna().astype(int)

    # Tagline length
    if "tagline" in df.columns:
        df["tagline_length"] = df["tagline"].apply(lambda x: len(x) if isinstance(x, str) else 0)

    # Summer release indicator
    df['is_summer'] = df['release_month'].isin([5,6,7,8]).astype(int)
    
    return df


def run_feature_engineering(df, config):
    feature_engineering = config["feature_engineering"]

    # Feature engineering for list vars
    for var, params in feature_engineering["categorical_vars"].items():
        df = create_indicators(
            df,
            var,
            top_n=params["top_n"],
            cumulative_pct=params["cumulative_pct"],
            list_col=params["list_col"],
            min_count=params["min_count"]
        )

    # Add other features
    df = add_basic_features(df)

    return df

