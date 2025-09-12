import os
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from src.utils import ts, safe_sheet_name

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no figure popping up)

def run_single_var_regression(df, var, target, regularized):
    """Run single-variable logistic regression and return stats + predictions."""
    temp = df[[var, target]].dropna()
    if temp[var].nunique() < 2:  # skip constants
        return None, None, None
    
    X_const = sm.add_constant(temp[[var]])
    y = temp[target]
    warning_messages = []

    # Capture warnings during fitting
    warnings_str = ""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            if regularized:
                model = sm.Logit(y, X_const).fit_regularized(disp=False)
                pval = np.nan # fit_regularized does not provide p-values
            else:
                model = sm.Logit(y, X_const).fit(disp=False)
                pval = model.llr_pvalue
        except Exception:
            return None, None, None
        
        # If any warnings occurred, join them into a single string
        if w:
            warnings_str = "; ".join([str(wi.message) for wi in w])
    
    temp["pred"] = model.predict(X_const)
    auc = roc_auc_score(y, temp["pred"])
    gini = 2 * auc - 1
    stats = {
        "var": var,
        "n": len(temp),
        "missing_pct": df[var].isnull().mean(),
        "coef": model.params[var],
        "p_value": pval,
        "pseudo_r2": model.prsquared,
        "gini": gini,
        "mean": temp[var].mean(),
        "std": temp[var].std(),
        "warnings": warnings_str
    }

    warnings_str = "; ".join(warning_messages) if warning_messages else None

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
    vars_to_test = numeric_vars + indicator_vars
    target = config["target"]
    regularized = config["sfa"].get("regularized", None)

    stats_list = []
    pdf_path = os.path.join(output_dir, f"plots/sfa_plots_{ts}.pdf")
    excel_path = os.path.join(output_dir, f"raw/sfa_stats_{ts}.xlsx")

    pdf = PdfPages(pdf_path)

    for var in vars_to_test:
        model, stats, temp = run_single_var_regression(df, var, target, regularized)
        if model is None:
            continue
        stats_list.append(stats)
        if var in numeric_vars:
            plot_sfa_variable(temp, var, target, pdf)

    pdf.close()

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_excel(excel_path, sheet_name=safe_sheet_name("sfa_stats"), index=False)

    print(f"SFA complete. Excel saved to {excel_path}, PDF saved to {pdf_path}")
