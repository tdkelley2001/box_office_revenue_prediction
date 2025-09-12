import os
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from src.utils import ts


def generate_combos(block_names, max_blocks=None):
    """Generate all unique block name combinations once."""
    combos = []
    for r in range(1, len(block_names) + 1):
        if max_blocks is not None and r > max_blocks:
            break
        combos.extend(itertools.combinations(block_names, r))
    return combos


def evaluate_mfa_model(df, target, features, regularized):
    """Fit logistic regression on given features, return metrics + VIF."""
    warnings_str = ""
    vif_warnings = ""
    try:
        X = df[features].copy()
        y = df[target]

        X_const = sm.add_constant(X)

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

        # Predictions & metrics
        pred = model.predict(X_const)
        auc = roc_auc_score(y, pred)
        gini = 2 * auc - 1

        # VIF calculation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vif_df = pd.DataFrame()
            vif_df["variable"] = X_const.columns
            vif_df["VIF"] = [
                variance_inflation_factor(X_const.values, i)
                for i in range(X_const.shape[1])
            ]
            if w:
                vif_warnings = "; ".join(str(wi.message) for wi in w)

        return {
            "features": features,
            "aic": model.aic,
            "gini": gini,
            "pval": pval,
            "vif": vif_df.to_dict(orient="records"),
            "warnings": {"model": warnings_str, "vif": vif_warnings}
        }

    except Exception as e:
        return {
            "features": features,
            "aic": float("inf"),
            "gini": float("nan"),
            "pval": float("nan"),
            "vif": [],
            "warnings": {"model": warnings_str, "vif": vif_warnings},
            "error": str(e)
        }


def run_mfa(df, config):
    """
    Multi-Feature Analysis (MFA):
    - Loop across combinations of numeric + categorical blocks
    - Each block is all-or-none (categorical blocks grouped by prefix)
    - Evaluate with Logistic Regression
    """
    target = config["target"]
    single_vars = config["mfa"].get("single_vars", [])
    categorical_blocks = config["mfa"].get("categorical_blocks", {})
    max_blocks = config["mfa"].get("max_blocks", None)
    regularized = config["mfa"].get("regularized", None)
    output_path = config["output"]["mfa_dir"]

    # ---- Expand categorical blocks into full column lists
    block_map = {}

    # Numeric blocks: each variable is its own block
    for var in single_vars:
        if var in df.columns:
            block_map[var] = [var]

    # Categorical blocks: expand all columns with the prefix
    for prefix in categorical_blocks:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            block_map[prefix] = cols

    block_names = list(block_map.keys())

    # ---- Generate combos once
    combos = generate_combos(block_names, max_blocks=max_blocks)

    # ---- Loop through combinations of blocks
    all_results = []
    for combo in tqdm(combos, desc="MFA Combos", leave=False):
        # Flatten block → actual feature list
        features = [f for block in combo for f in block_map[block]]

        result = evaluate_mfa_model(df, target, features, regularized)
        result["blocks"] = combo
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # ---- Save results
    if output_path:
        excel_path = os.path.join(output_path, f"sfa_stats_{ts}.xlsx")
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, sheet_name="MFA_Results", index=False)

