import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def save_summary_stats(df, output_dir):
    summary = df.describe(include="all")
    summary.to_csv(f"{output_dir}/summary_stats.csv")

def save_null_counts(df, output_dir):
    nulls = df.isnull().sum().reset_index()
    nulls.columns = ["column", "null_count"]
    nulls.to_csv(f"{output_dir}/null_counts.csv", index=False)

def plot_target_distribution(df, output_dir):
    sns.countplot(x="bomb", data=df)
    plt.title("Bomb Class Distribution")
    plt.savefig(f"{output_dir}/class_distribution.png")
    plt.clf()

def plot_numeric_distributions(df, output_dir):
    for col in ["budget", "revenue"]:
        if col in df.columns:
            sns.histplot(df[col], bins=50, kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(f"{output_dir}/{col}_distribution.png")
            plt.clf()

def plot_return_ratio(df, output_dir):
    if "budget" in df.columns and "revenue" in df.columns:
        df["return_ratio"] = df["revenue"] / df["budget"]
        sns.histplot(df["return_ratio"], bins=50, kde=True)
        plt.title("Distribution of Return Ratio")
        plt.savefig(f"{output_dir}/return_ratio_distribution.png")
        plt.clf()

def plot_correlation_heatmap(df, output_dir):
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.clf()

def plot_release_year(df, output_dir):
    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        sns.countplot(x="release_year", data=df)
        plt.xticks(rotation=90)
        plt.title("Number of Movies by Release Year")
        plt.savefig(f"{output_dir}/release_year_counts.png")
        plt.clf()

def run_eda(df, config):
    output_dir = config["output"]["plots_dir"]
    os.makedirs(output_dir, exist_ok=True)

    save_summary_stats(df, output_dir)
    save_null_counts(df, output_dir)
    plot_target_distribution(df, output_dir)
    plot_numeric_distributions(df, output_dir)
    plot_return_ratio(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_release_year(df, output_dir)