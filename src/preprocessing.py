import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def parse_json_columns(df, columns):
    """Parse JSON-like string columns into lists of names."""
    for col in columns:
        def parse(x):
            try:
                return [g['name'] for g in ast.literal_eval(x)]
            except:
                return []
        df[f"{col}_list"] = df[col].apply(parse)
    return df

def one_hot_encode_lists(df, col, top_n=-1):
    """One-hot encode list columns with optional top-N limit."""
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df[col])
    classes = mlb.classes_

    if top_n > 0:
        # Get top N classes by frequency
        freq = pd.Series(encoded.sum(axis=0), index=classes).sort_values(ascending=False)
        keep = freq.index[:top_n]
        mask = [c in keep for c in classes]
        encoded = encoded[:, mask]
        classes = keep

    encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{c}" for c in classes], index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def encode_categorical(df, cols):
    """One-hot encode simple categorical variables."""
    return pd.get_dummies(df, columns=cols, drop_first=True)

def process_dates(df, col):
    """Extract useful date parts from release_date."""
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df['release_year'] = df[col].dt.year
    df['release_month'] = df[col].dt.month
    df['release_quarter'] = df[col].dt.quarter
    df['is_summer'] = df['release_month'].isin([5,6,7,8]).astype(int)
    return df

def preprocess_data(df, config):
    pp_cfg = config["preprocessing"]

    # Parse JSON-like columns
    df = parse_json_columns(df, pp_cfg["parse_columns"])

    # One-hot encode list columns
    for col, top_n in pp_cfg["one_hot_top_n"].items():
        list_col = f"{col}_list"
        if list_col in df.columns:
            df = one_hot_encode_lists(df, list_col, top_n)

    # Encode categorical
    df = encode_categorical(df, pp_cfg["categorical"])

    # Process dates
    df = process_dates(df, pp_cfg["date_column"])

    return df
