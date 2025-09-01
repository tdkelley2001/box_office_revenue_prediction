from datetime import datetime
import ast
import pandas as pd
import numpy as np

def get_timestamp(fmt="%Y%m%d_%H%M%S"):
    """Return current timestamp as a string for filenames/folders."""
    return datetime.now().strftime(fmt)

# Generate one timestamp per run, available globally
ts = get_timestamp()


def safe_literal_eval(x):
    # If x is array-like, convert to list
    if isinstance(x, (np.ndarray, pd.Series)):
        return x.tolist()
    # If scalar, check for null
    if np.isscalar(x) and pd.isnull(x):
        return []
    # If string, try to literal_eval
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    # If already a list
    if isinstance(x, list):
        return x
    # Any other unexpected type
    return []


def safe_sheet_name(name):
    """Ensure Excel sheet name is <= 31 characters."""
    # Remove invalid characters for sheet names as well
    invalid_chars = ['\\', '/', '*', '[', ']', ':', '?']
    for c in invalid_chars:
        name = name.replace(c, "_")
    # Truncate to 31 chars
    return name[:31]
