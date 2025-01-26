import pandas as pd
from scipy.stats import spearmanr, pearsonr
import numpy as np
from typing import List

def compute_correlations(
    df: pd.DataFrame,
    include_categorical: bool = False,
    cat_cols: List[str] = None,
    drop_first: bool = True,
    sort_results: bool = True
) -> pd.DataFrame:
    """
    Compute pairwise correlations (Pearson or Spearman) dynamically based on data type.
    Handles categorical variables with optional one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric and/or categorical columns.
    include_categorical : bool, optional
        Whether to one-hot encode categorical variables before computing correlations.
        If True, Spearman correlation is used. If False, Pearson correlation is used.
        Default is False.
    drop_first : bool, optional
        If include_categorical=True, whether to drop the first dummy column
        to avoid multicollinearity. Default is True.
    sort_results : bool, optional
        Whether to sort the results by absolute correlation value in descending order.
        Default is True.

    Returns
    -------
    pd.DataFrame
        A structured DataFrame with the following columns:
        - "Variable 1"
        - "Variable 2"
        - "Correlation"
        - "p-value" (only for Spearman correlation)
        - "Method" (Pearson or Spearman)
        - "Error" (if any issues occur)
    """
    # Step 1: Handle categorical variables if include_categorical=True
    if include_categorical:
        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
        method = "spearman"
    else:
        method = "pearson"

    # Step 2: Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise ValueError("Not enough numeric columns to compute correlations.")
    df = df[numeric_cols]

    # Step 3: Compute correlations
    results = []
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i + 1:]:  # Avoid duplicate pairs (A, B) and (B, A)
            x, y = df[col1].dropna(), df[col2].dropna()
            common_idx = x.index.intersection(y.index)
            x, y = x.loc[common_idx], y.loc[common_idx]

            try:
                if method == "pearson":
                    corr, p_value = pearsonr(x, y)
                elif method == "spearman":
                    corr, p_value = spearmanr(x, y)
                else:
                    raise ValueError("Unsupported correlation method. Choose 'pearson' or 'spearman'.")

                results.append({
                    "Variable 1": col1,
                    "Variable 2": col2,
                    "Correlation": corr,
                    "p-value": p_value,
                    "Method": method,
                    "Error": None
                })
            except (ValueError, TypeError) as e:
                # Catch any issues with correlation calculation
                results.append({
                    "Variable 1": col1,
                    "Variable 2": col2,
                    "Correlation": None,
                    "p-value": None,
                    "Method": method,
                    "Error": str(e)
                })

    # Step 4: Create a results DataFrame
    results_df = pd.DataFrame(results)

    # Step 5: Sort results by absolute correlation (if requested)
    if sort_results:
        results_df["abs_correlation"] = results_df["Correlation"].abs()
        results_df.sort_values(by="p-value", ascending=True, inplace=True)
        results_df.drop(columns=["abs_correlation"], inplace=True)

    return results_df.reset_index(drop=True)
