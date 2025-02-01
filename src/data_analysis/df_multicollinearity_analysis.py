import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def compute_vif(data: pd.DataFrame, drop_constant: bool = True, drop_first: bool = True) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for all numeric and one-hot encoded features.
    """
    # Step 1: Make a copy of the data
    df = data.copy()
    # print("Initial DataFrame:")
    # print(df.head())w

    # Step 2: One-hot encode categorical variables (if any)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    # print(f"Categorical columns detected: {list(cat_cols)}")
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
        # print("DataFrame after one-hot encoding:")
        # print(df.head())

    # Step 3: Ensure all features are numeric
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
    # print("DataFrame after converting all columns to numeric:")
    # print(df.dtypes)

    # Step 4: Check for NaN or Infinite Values
    # print("Checking for NaN or Infinite values in the DataFrame:")
    # print("Number of NaN values per column:")
    # print(df.isnull().sum())
    # print("Number of Infinite values per column:")
    # print((df == np.inf).sum() + (df == -np.inf).sum())

    # Replace infinite values with NaN and drop rows with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    # print("DataFrame after handling NaN and Infinite values:")
    # print(df.head())

    # Step 5: Add a constant column for regression
    df_with_const = sm.add_constant(df, has_constant="add")
    # print("DataFrame with constant column added:")
    # print(df_with_const.head())

    # Step 6: Ensure all columns are numeric
    df_with_const = df_with_const.astype(float)  # Convert all columns to float64
    # print("DataFrame after ensuring all columns are numeric:")
    # print(df_with_const.dtypes)

    # Step 7: Compute VIF
    vif_data = []
    columns_for_vif = df_with_const.columns
    if drop_constant and "const" in columns_for_vif:
        columns_for_vif = columns_for_vif.drop("const")
        # print("Dropped constant column from VIF calculation.")

    for col in columns_for_vif:
        vif_value = variance_inflation_factor(df_with_const.values, df_with_const.columns.get_loc(col))
        # print(f"VIF for {col}: {vif_value:.2f}")
        vif_data.append({"Feature": col, "VIF": vif_value})

    # Step 8: Return results sorted by VIF in descending order
    vif_df = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False).reset_index(drop=True)
    # print("Final VIF DataFrame:")
    # print(vif_df)

    return vif_df

def compute_vif_vectorized(data: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for all numeric and one-hot encoded features
    using a vectorized approach via the inverse of the correlation matrix.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        drop_first (bool): Whether to drop the first category during one-hot encoding.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'Feature' and 'VIF', sorted by VIF descending.
    """
    # Step 1: Copy the data
    df = data.copy()

    # Step 2: One-hot encode categorical variables (if any)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)

    # Step 3: Ensure all features are numeric and handle non-numeric entries
    df = df.apply(pd.to_numeric, errors='coerce')

    # Step 4: Replace infinite values and drop rows with NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure all values are floats (this helps with numerical stability)
    df = df.astype(float)

    # (Optional) If you were previously adding a constant for regression,
    # you typically do NOT include the constant when computing VIF.
    # Thus, we leave it out here.

    # Step 5: Compute the correlation matrix
    corr_matrix = df.corr()

    # Step 6: Invert the correlation matrix
    # If the matrix is nearly singular, use the pseudo-inverse
    try:
        inv_corr_matrix = np.linalg.inv(corr_matrix.values)
    except np.linalg.LinAlgError:
        inv_corr_matrix = np.linalg.pinv(corr_matrix.values)

    # Step 7: Extract VIFs from the diagonal of the inverse correlation matrix
    vif_values = np.diag(inv_corr_matrix)

    # Step 8: Format the results in a DataFrame
    vif_df = pd.DataFrame({
        "Feature": corr_matrix.columns,
        "VIF": vif_values
    }).sort_values(by="VIF", ascending=False).reset_index(drop=True)

    return vif_df
