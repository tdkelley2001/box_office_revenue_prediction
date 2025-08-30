import os

def save_cleaned_data(df, config):
    output_path = config["output"]["cleaned_data_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)