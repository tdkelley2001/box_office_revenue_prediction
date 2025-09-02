import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import ts, safe_sheet_name


# ---------- Single-variable logistic regression ----------
def run_univariate_logit(df, var, target):
    missing_pct = df[var].isnull().mean()  # fraction of missing values

    X = df[[var]].copy()
    X = sm.add_constant(X)
    y = df[target]

    try:
        model = sm.Logit(y, X).fit(disp=0)
        coef = model.params[var]
        pval = model.pvalues[var]
        y_pred = model.predict(X)
        gini = 2 * roc_auc_score(y, y_pred) - 1
    except Exception as e:
        print(f"Warning: Could not run SFA for {var}: {e}")
        coef, pval, gini = None, None, None

    return {
        "variable": var,
        "missing_pct": missing_pct,
        "coefficient": coef,
        "p_value": pval,
        "gini": gini
    }

# ---------- Diagnostic plot for numeric variables ----------
def plot_binned_means(df, var, target, pdf, bins=10):
    plt.figure(figsize=(6,4))
    df['bin'] = pd.qcut(df[var], q=bins, duplicates='drop')
    bin_means = df.groupby('bin')[target].mean()
    bin_centers = [interval.mid for interval in bin_means.index]
    plt.plot(bin_centers, bin_means, marker='o')
    plt.title(f"{var} vs {target} (binned)")
    plt.xlabel(var)
    plt.ylabel(f"Mean {target}")
    plt.grid(True)
    pdf.savefig()
    plt.close()
    df.drop(columns='bin', inplace=True)

# ---------- Wrapper ----------
def run_sfa(df, config):
    output_dir = config["output"]["sfa_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Get variables to test
    vars_to_test = config["sfa"]["numeric_columns"].copy()
    indicator_prefixes = config["sfa"].get("indicator_prefixes", [])
    indicator_vars = [col for col in df.columns if any(col.startswith(p) for p in indicator_prefixes)]
    vars_to_test.extend(indicator_vars)
    target = config["sfa"]["target"]

    sfa_results = []
    pdf_path = os.path.join(output_dir, f"sfa_plots_{ts}.pdf")
    pdf = PdfPages(pdf_path)

    for var in vars_to_test:
        if var not in df.columns:
            continue
        
        # Run regression
        result = run_univariate_logit(df, var, target)
        sfa_results.append(result)

        # Plot only for numeric, non-indicator variables
        if var not in indicator_vars:
            plot_binned_means(df, var, target, pdf)

    results_df = pd.DataFrame(sfa_results)
    excel_path = os.path.join(output_dir, f"sfa_results_{ts}.xlsx")
    results_df.to_excel(excel_path, index=False)
    pdf.close()

    print(f"SFA complete. Results saved to {excel_path}, plots saved to {pdf_path}")
    return results_df

