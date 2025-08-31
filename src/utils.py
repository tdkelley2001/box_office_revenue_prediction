from datetime import datetime

def get_timestamp(fmt="%Y%m%d_%H%M%S"):
    """Return current timestamp as a string for filenames/folders."""
    return datetime.now().strftime(fmt)

# Generate one timestamp per run, available globally
ts = get_timestamp()