import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from src.utils import ts, safe_sheet_name

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no figure popping up)

def run_single_var_regression(df, var, target):
    """Run single-variable logistic regression and return stats + predictions."""
    temp = df[[var, target]].dropna()
    if temp[var].nunique() < 2:  # skip constants
        return None, None
    
    X = sm.add_constant(temp[[var]])
    y = temp[target]
    try:
        model = sm.Logit(y, X).fit(disp=False)
    except Exception:
        return None, None
    
    temp["pred"] = model.predict(X)
    auc = roc_auc_score(y, temp["pred"])
    gini = 2 * auc - 1
    stats = {
        "var": var,
        "n": len(temp),
        "missing_pct": df[var].isnull().mean(),
        "coef": model.params[var],
        "p_value": model.pvalues[var],
        "pseudo_r2": model.prsquared,
        "gini": gini,
        "mean": temp[var].mean(),
        "std": temp[var].std()
    }
    return model, stats, temp


def plot_sfa_variable(temp, var, target, pdf, n_bins=40):
    """Create combined distribution + actual vs predicted plot for a variable."""
    temp = temp.copy()
    temp["bin"] = pd.qcut(temp[var], q=n_bins, duplicates="drop")

    grouped = temp.groupby("bin").agg(
        count=(var, "size"),
        mean_var=(var, "mean"),
        mean_target=(target, "mean"),
        pred=("pred", "mean")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(7,5))

    # Bar plot: distribution (counts)
    ax1.bar(
        grouped["mean_var"],
        grouped["count"],
        width=0.01 * (grouped["mean_var"].max() - grouped["mean_var"].min()),
        alpha=0.4,
        color="gray",
        label="Count"
    )
    ax1.set_ylabel("Count")
    ax1.set_xlabel(var)

    # Second axis for rates
    ax2 = ax1.twinx()
    ax2.scatter(grouped["mean_var"], grouped["mean_target"], color="blue", label="Actual", marker="o")
    ax2.plot(grouped["mean_var"], grouped["pred"], color="red", label="Predicted", marker="x")
    ax2.set_ylabel("Target Rate")

    # Title + legends
    fig.suptitle(f"Distribution + Actual vs Predicted for {var}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    pdf.savefig(fig)
    plt.close(fig)


def run_sfa(df, config):
    """Run single factor analysis (SFA) with stats + plots."""
    output_dir = config["output"]["sfa_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Get variables to test
    numeric_vars = config["sfa"]["numeric_columns"].copy()
    indicator_prefixes = config["sfa"].get("indicator_prefixes", [])
    indicator_vars = [col for col in df.columns if any(col.startswith(p) for p in indicator_prefixes)]
    vars_to_test = numeric_vars.extend(indicator_vars)
    target = config["sfa"]["target"]

    stats_list = []
    pdf_path = os.path.join(output_dir, f"sfa_plots_{ts}.pdf")
    excel_path = os.path.join(output_dir, f"sfa_stats_{ts}.xlsx")

    pdf = PdfPages(pdf_path)

    for var in vars_to_test:
        model, stats, temp = run_single_var_regression(df, var, target)
        if model is None:
            continue
        stats_list.append(stats)
        if var in numeric_vars:
            plot_sfa_variable(temp, var, target, pdf)

    pdf.close()

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_excel(excel_path, sheet_name=safe_sheet_name("sfa_stats"), index=False)

    print(f"SFA complete. Excel saved to {excel_path}, PDF saved to {pdf_path}")
