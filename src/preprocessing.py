import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils import safe_literal_eval

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


def get_counts(df, var, list_col=True):
    """Return value counts for a column, handling list vs categorical."""
    if list_col:
        df[var] = df[var].apply(safe_literal_eval)
        return df[var].explode().value_counts()
    else:
        return df[var].value_counts()


def select_levels(counts, top_n=None, cumulative_pct=None):
    """Select levels based on top_n or cumulative percentage coverage."""
    if top_n is not None:
        keep = counts.head(top_n).index.tolist()
    elif cumulative_pct is not None:
        cumsum = counts.cumsum() / counts.sum()
        keep = cumsum[cumsum <= cumulative_pct].index.tolist()
        # Guarantee at least one level
        if not keep and len(counts) > 0:
            keep = [counts.index[0]]
    else:
        keep = counts.index.tolist()
    return keep


def apply_min_count(counts, keep_levels, min_count=None):
    """Filter levels by minimum count."""
    if min_count is not None:
        keep_levels = [lvl for lvl in keep_levels if counts.get(lvl, 0) >= min_count]
    return keep_levels


def create_indicator_columns(df, var, keep_levels, list_col=True):
    """Create indicator columns for the selected levels, plus an 'Other' column."""
    new_cols = {}
    if list_col:
        for level in keep_levels:
            new_cols[f"{var}_{level}"] = df[var].apply(lambda x: int(level in x) if isinstance(x, list) else 0)
        
        # Only create 'Other' if any value falls outside keep_levels
        other_mask = df[var].apply(lambda x: any(i not in keep_levels for i in x) if isinstance(x, list) else False)
        if other_mask.any():
            new_cols[f"{var}_Other"] = other_mask.astype(int)

    else:
        for level in keep_levels:
            new_cols[f"{var}_{level}"] = (df[var] == level).astype(int)
        
        # Only create 'Other' if any value falls outside keep_levels
        other_mask = ~df[var].isin(keep_levels)
        if other_mask.any():
            new_cols[f"{var}_Other"] = other_mask.astype(int)

    # Add all new columns at once to reduce fragmentation
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df

def create_indicators(
    df,
    var,
    top_n=None,
    cumulative_pct=None,
    list_col=True,
    min_count=None
):
    """
    Main wrapper for indicator creation.
    Selects levels based on top_n / cumulative_pct,
    filters with min_count, and creates indicator columns.
    """
    if var not in df.columns:
        return df

    counts = get_counts(df, var, list_col=list_col)
    keep_levels = select_levels(counts, top_n=top_n, cumulative_pct=cumulative_pct)
    keep_levels = apply_min_count(counts, keep_levels, min_count=min_count)
    df = create_indicator_columns(df, var, keep_levels, list_col=list_col)

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

