import pandas as pd
from scipy import stats
import numpy as np
import logging 

def run_univariate_tests(
    df: pd.DataFrame,
    categorical_cols: list,
    quantitative_cols: list,
    test_type: str = "anova",
    sort_results: bool = True
) -> pd.DataFrame:
    """
    Run univariate tests (one-way ANOVA or Kruskal-Wallis) across all
    categorical x quantitative variable pairs.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing both categorical and quantitative columns.
    categorical_cols : list
        A list of column names in df that are categorical.
    quantitative_cols : list
        A list of column names in df that are numeric/quantitative.
    test_type : str, optional
        The type of test to run: "anova" or "kruskal".
        Default is "anova".
    sort_results : bool, optional
        Whether to sort the final results by p-value ascending. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the test results for each categorical-quantitative pair.
        Columns include:
          - "Categorical Variable"
          - "Quantitative Variable"
          - "Statistic" (F-statistic or H-statistic)
          - "p-value"
          - "Error" (if applicable)
    """
    # Select the appropriate test function
    if test_type.lower() == "anova":
        test_func = stats.f_oneway
        stat_label = "F-statistic"
    elif test_type.lower() == "kruskal":
        test_func = stats.kruskal
        stat_label = "H-statistic"
    else:
        raise ValueError(f"Unsupported test_type: {test_type}. Must be 'anova' or 'kruskal'.")

    results = []

    # Loop through each categorical x quantitative pairing
    for cat_col in categorical_cols:
        for quant_col in quantitative_cols:

            # 1. Gather unique categories for this categorical variable
            categories = df[cat_col].dropna().unique()

            # 2. Group the quantitative data by category
            grouped_data_list = [
                df.loc[df[cat_col] == category, quant_col].dropna().values
                for category in categories
            ]

            # 3. Perform the test
            try:
                test_result = test_func(*grouped_data_list)
                results.append({
                    "Categorical Variable": cat_col,
                    "Quantitative Variable": quant_col,
                    stat_label: test_result.statistic,
                    "p-value": test_result.pvalue,
                    "Error": None
                })
            except ValueError as e:
                # Handle errors (like one group being empty)
                results.append({
                    "Categorical Variable": cat_col,
                    "Quantitative Variable": quant_col,
                    stat_label: None,
                    "p-value": None,
                    "Error": str(e)
                })

    # Create DataFrame and optionally sort by p-value
    results_df = pd.DataFrame(results)
    if sort_results and "p-value" in results_df.columns:
        results_df.sort_values("p-value", inplace=True, ascending=True)

    return results_df.reset_index(drop=True)

def run_one_sample_ttest(
    df: pd.DataFrame,
    categorical_cols: list,
    quantitative_cols: list,
    # Does acategory produce free trails significantly above or below the dataset's
    popmean: int = 0,  # Hypothesize that the true mean number of free trials is X. Usually use a "value" decided by the business or the mean/median of the entire dataset.
    alpha: float = 0.05,
    sort_results: bool = True,
    stat_type: str = "mean"
) -> pd.DataFrame:
    """
    Perform a one-sample t-test on each group of each categorical/quantitative pair,
    comparing to a hypothesized population mean (popmean).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing your data.
    categorical_cols : list
        List of column names in df that are categorical.
    quantitative_cols : list
        List of column names in df that are numeric/quantitative.
    popmean : float or dict, optional
        The hypothesized population mean for the one-sample test.
        - If float, the same popmean is used for all quantitative variables.
        - If dict, keys should be quantitative column names and values the hypothesized mean.
        Default is 0.
    alpha : float, optional
        Significance level used for deciding whether to reject the null hypothesis.
        Default is 0.05.
    sort_results : bool, optional
        Whether to sort the final DataFrame by p-value in ascending order.
        Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
            - "Categorical Variable"
            - "Category"
            - "Quantitative Variable"
            - "Sample Size"
            - "Sample Stat"
            - "T-statistic"
            - "p-value"
            - "Decision" (Reject H0 or Fail to Reject H0)
            - "Error" (if any issue occurred, e.g. empty group)
    """
    def _get_popmean_for_col(col: str):
        """Retrieve the popmean for the current quantitative variable."""
        if isinstance(popmean, dict):
            return popmean.get(col, 0)  # default to 0 if not found in dict
        return popmean  # if single float

    one_sample_results = []

    for cat_col in categorical_cols:
        for quant_col in quantitative_cols:
            # 1. Identify unique categories for this categorical variable
            categories = df[cat_col].dropna().unique()

            # 2. Group the data by category
            for category_value in categories:
                data_array = df.loc[df[cat_col] == category_value, quant_col].dropna().values

                # 3. Perform the one-sample t-test for this group vs. popmean
                try:
                    # If there's not enough data, stats.ttest_1samp can raise errors
                    t_result = stats.ttest_1samp(
                        data_array,
                        _get_popmean_for_col(quant_col)
                    )

                    # Basic descriptive stats for context
                    sample_size = len(data_array)

                    if stat_type == "mean":
                        sample_metric = data_array.mean() if sample_size > 0 else None
                    else:
                        logging.warning(f"Unknown stat_type: {stat_type}. Defaulting to mean.")
                        #sample_metric = np.median(data_array) if sample_size > 0 else None

                    # Determine reject/fail decision
                    decision = (
                        "Reject H0"
                        if (t_result.pvalue is not None and t_result.pvalue < alpha)
                        else "Fail to Reject H0"
                    )

                    one_sample_results.append({
                        "Categorical Variable": cat_col,
                        "Category": category_value,
                        "Quantitative Variable": quant_col,
                        "Sample Size": sample_size,
                        "Sample Mean": sample_metric,
                        "T-statistic": t_result.statistic,
                        "p-value": t_result.pvalue,
                        "Decision": decision,
                        "Error": None
                    })

                except ValueError as e:
                    # E.g. not enough data in the group
                    one_sample_results.append({
                        "Categorical Variable": cat_col,
                        "Category": category_value,
                        "Quantitative Variable": quant_col,
                        "Sample Size": None,
                        "Sample Stat": None,
                        "T-statistic": None,
                        "p-value": None,
                        "Decision": "N/A",
                        "Error": str(e)
                    })

    # Convert to DataFrame
    one_sample_results_df = pd.DataFrame(one_sample_results)

    # Optionally sort by p-value
    if sort_results:
        # Some rows might have p-value = None if there's an error -> we place them at the bottom
        one_sample_results_df.sort_values(
            by="p-value",
            ascending=True,
            na_position="last",
            inplace=True
        )

    # Reset index for clean display
    return one_sample_results_df.reset_index(drop=True)
