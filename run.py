from src.config import config
from src.loader import load_movie_data
from src.eda import run_eda
from src.saver import save_cleaned_data

# Load config
config = config("config/config.yaml")

# Load movie data
movie_data = load_movie_data(config)

# Run EDA
run_eda(movie_data, config)

# Save cleaned dataset
save_cleaned_data(movie_data, config)
