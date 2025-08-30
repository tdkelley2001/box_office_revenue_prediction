import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df, config):
    output_dir = config["output"]["plots_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Class balance
    sns.countplot(x="bomb", data=df)
    plt.title("Bomb Class Distribution")
    plt.savefig(f"{output_dir}/class_distribution.png")
    plt.clf()