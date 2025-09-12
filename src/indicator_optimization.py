import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
from src.preprocessing import create_indicators, get_counts
from src.utils import ts, safe_sheet_name


def evaluate_indicators(df, target, indicator_cols, regularized):
    """Fit univariate logistic regression on combined indicators, return Gini, p-value, and AIC."""
    try:
        X = df[indicator_cols]
        y = df[target]
        X_const = sm.add_constant(X)

        warnings_str = ""
        # Capture warnings during model fit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if regularized:
                model = sm.Logit(y, X_const).fit_regularized(disp=False)
                pval = np.nan # fit_regularized does not provide p-values
            else:
                model = sm.Logit(y, X_const).fit(disp=False)
                pval = model.llr_pvalue
            
            if w:
                warnings_str = "; ".join([str(wi.message) for wi in w])

        # Pseudo-Gini from ROC
        pred = model.predict(X_const)
        auc = roc_auc_score(y, pred)
        gini = 2 * auc - 1
        return gini, pval, model.aic, warnings_str
    except Exception:
        return np.nan, np.nan, np.inf, warnings_str


def plot_var_results(var, results_df, writer):
    """
    Create and save plots for a given variable:
    - Gini vs n
    - AIC vs n
    """
    var_results = results_df[results_df["var"] == var]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.set_title(f"Indicator Optimization: {var}")
    ax1.set_xlabel("n (levels kept)")

    # Gini on left y-axis
    ax1.plot(var_results["n"], var_results["gini"], marker="o", color="blue", label="Gini")
    ax1.set_ylabel("Gini", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # AIC on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(var_results["n"], var_results["aic"], marker="x", color="red", label="AIC")
    ax2.set_ylabel("AIC", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.tight_layout()

    # Save directly into Excel workbook as image
    sheet_name = safe_sheet_name(f"{var[:25]}_Plot")  # Excel sheet name limit
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    # Save temp image and insert
    img_path = f"output/modeling/eda/indicator_optimization/plots/{var}_opt.png"
    plt.savefig(img_path, bbox_inches="tight")
    worksheet.insert_image("B2", img_path)

    plt.close(fig)


def optimize_indicators_by_n(df, config):
    """
    For each categorical variable, loop across all possible n levels to keep.
    Selects the best n by AIC (with Gini as tiebreaker).
    Saves results + plots to Excel if configured.
    """
    all_results = []

    max_n_cap = config["optimization"]["indicator_search"].get("max_n")
    min_count = config["optimization"]["indicator_search"].get("min_count", None)
    regularized = config["optimization"]["indicator_search"].get("regularized", None)
    output_path = config["output"]["ind_opt_dir"]
    
    feature_engineering = config["feature_engineering"]
    target = config["target"]
    

    total_vars = len(feature_engineering["categorical_vars"])

    for var_idx, (var, params) in enumerate(feature_engineering["categorical_vars"].items(), 1):
        list_col = params.get("list_col", True)
        counts = get_counts(df, var, list_col=list_col)
        max_n = len(counts)
        if max_n_cap is not None:
            max_n = min(max_n, max_n_cap)

        print(f"Processing {var_idx}/{total_vars}: {var} (max n={max_n})")

        for n in tqdm(range(1, max_n + 1), desc=f"{var} n", leave=False):
            df_copy = create_indicators(
                df.copy(),
                var,
                top_n=n,
                min_count=min_count,
                list_col=list_col
            )
            indicator_cols = [c for c in df_copy.columns if c.startswith(f"{var}_")]
            if not indicator_cols:
                continue

            gini, pval, aic, warnings_str = evaluate_indicators(df_copy, target, indicator_cols, regularized)

            all_results.append({
                "var": var,
                "n": n,
                "n_indicators": len(indicator_cols),
                "gini": gini,
                "pval": pval,
                "aic": aic,
                "warnings": warnings_str
            })

    results_df = pd.DataFrame(all_results)

    # Select best per variable
    if regularized:
        best_results = (
            results_df
            .sort_values(["var", "gini", "n_indicators"], ascending=[True, False, True])
            .groupby("var")
            .head(1)
            .reset_index(drop=True)
        )
    else:
        best_results = (
            results_df
            .sort_values(["var", "aic", "gini"], ascending=[True, True, False])
            .groupby("var")
            .head(1)
            .reset_index(drop=True)
        )

    # Save if requested
    if output_path:
        excel_path = os.path.join(output_path, f"indicator_optimization_{ts}.xlsx")
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, sheet_name="All_Results", index=False)
            best_results.to_excel(writer, sheet_name="Best_Per_Var", index=False)

            # Add plots per variable
            for var, params in feature_engineering["categorical_vars"].items():
                if var in results_df["var"].unique():
                    plot_var_results(var, results_df, writer)
