import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ast
from src.utils import ts, safe_literal_eval, safe_sheet_name

def run_eda_round1(df, config):
    output_dir = config["output"]["eda_round1_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_cols = config["eda_round1"]["numeric_columns"]
    categorical_cols = config["eda_round1"]["categorical_columns"]
    list_cols = config["eda_round1"]["list_columns"]
    text_cols = config["eda_round1"]["text_columns"]
    target_col = config["eda_round1"]["target"]
    
    
    # Parse list columns if loaded from CSV
    for col in list_cols:
        df[col] = df[col].apply(safe_literal_eval)

    # Use a Unicode-supporting font globally
    matplotlib.rcParams['font.family'] = matplotlib.rcParamsDefault['font.family']
    
    # Timestamp for outputs
    excel_path = os.path.join(output_dir, f"eda_round1_{ts}.xlsx")
    pdf_path = os.path.join(output_dir, f"eda_plots_{ts}.pdf")
    
    pdf = PdfPages(pdf_path)
    writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    
    # ---------- Numeric ----------
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe(percentiles=config["percentiles"])
        numeric_summary.to_excel(writer, sheet_name=safe_sheet_name("numeric_summary"))
        
        for col in numeric_cols:
            plt.figure(figsize=(6,4))
            df[col].hist(bins=50)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            pdf.savefig()
            plt.close()
    
    # ---------- Missing ----------
    missing_summary = df.isnull().sum().reset_index()
    missing_summary.columns = ["column", "missing_count"]
    missing_summary["missing_pct"] = missing_summary["missing_count"] / len(df)
    missing_summary.to_excel(writer, sheet_name=safe_sheet_name("missing_summary"), index=False)
    
    # ---------- Categorical ----------
    for col in categorical_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        counts.to_excel(writer, sheet_name=safe_sheet_name(f"{col}_counts"), index=False)
        
        plt.figure(figsize=(6,4))
        counts.head(20).plot(kind="bar", x=col, y="count", legend=False)
        plt.title(f"Top 20 categories for {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # ---------- List ----------
    for col in list_cols:
        df[f"num_{col}"] = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df[[f"num_{col}"]].to_excel(writer, sheet_name=safe_sheet_name(f"{col}_lengths"), index=False)
        
        all_items = [item for sublist in df[col] for item in sublist]
        freq = pd.Series(all_items).value_counts().reset_index()
        freq.columns = [col, "count"]
        freq.to_excel(writer, sheet_name=safe_sheet_name(f"{col}_frequencies"), index=False)
        
        plt.figure(figsize=(6,4))
        freq.head(20).plot(kind="bar", x=col, y="count", legend=False)
        plt.title(f"Top 20 items for {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # ---------- Text ----------
    for col in text_cols:
        df[f"{col}_length"] = df[col].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df[[f"{col}_length"]].describe().to_excel(writer, sheet_name=safe_sheet_name(f"{col}_length_summary"))
        
        plt.figure(figsize=(6,4))
        df[f"{col}_length"].hist(bins=50)
        plt.title(f"Length distribution of {col}")
        plt.xlabel("Length")
        plt.ylabel("Count")
        pdf.savefig()
        plt.close()
    
    # ---------- Target ----------
    target_counts = df[target_col].value_counts().reset_index()
    target_counts.columns = [target_col, "count"]
    target_counts.to_excel(writer, sheet_name=safe_sheet_name(f"{target_col}_distribution"), index=False)
    
    plt.figure(figsize=(6,4))
    target_counts.plot(kind="bar", x=target_col, y="count", legend=False)
    plt.title(f"{target_col} class distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Save Excel and PDF
    writer.close()
    pdf.close()
    
    print(f"EDA Round 1 complete. Excel saved to {excel_path}, PDF saved to {pdf_path}")


def run_eda_round2(df, config):
    """
    Round 2 EDA:
      - Missing summary
      - Percentile distributions for numeric columns
      - Counts of indicator variables grouped by prefix
      - Optionally split counts by target (e.g., bomb)
    """

    output_dir = config["output"]["eda_round2_dir"]
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = config["sfa"]["numeric_columns"]
    indicator_prefixes = config["sfa"]["indicator_prefixes"]
    target_col = config["sfa"]["target"]

    # Timestamped output file
    excel_path = os.path.join(output_dir, f"eda_round2_{ts}.xlsx")
    writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")

    # ---------- Missing ----------
    missing_summary = df.isnull().sum().reset_index()
    missing_summary.columns = ["column", "missing_count"]
    missing_summary["missing_pct"] = missing_summary["missing_count"] / len(df)
    missing_summary.to_excel(writer, sheet_name=safe_sheet_name("missing_summary"), index=False)

    # ---------- Numeric ----------
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe(percentiles=config["percentiles"])
        numeric_summary.to_excel(writer, sheet_name=safe_sheet_name("numeric_summary"))

    # ---------- Indicator groups ----------
    for prefix in indicator_prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue

        # Total counts for each indicator
        counts = df[cols].sum().reset_index()
        counts.columns = ["indicator", "count_1"]
        counts["count_0"] = len(df) - counts["count_1"]
        counts.to_excel(writer, sheet_name=safe_sheet_name(f"{prefix}_counts"), index=False)

        # Split by target if available
        if target_col in df.columns:
            counts_by_target = (
                df.groupby(target_col)[cols]
                  .sum()
                  .T
                  .reset_index()
                  .rename(columns={"index": "indicator"})
            )
            counts_by_target.to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{prefix}_by_{target_col}"),
                index=False
            )

    # Save Excel
    writer.close()
    print(f"EDA Round 2 complete. Excel saved to {excel_path}")
