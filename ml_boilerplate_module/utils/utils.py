import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f, norm, rankdata


def find_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate the Pearson correlation coefficient between two columns
    in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.

    Returns:
    float: The Pearson correlation coefficient between the two columns,
    ranging from -1 to 1.

    Examples:
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]}
    >>> df = pd.DataFrame(data)
    >>> find_correlation(df, 'A', 'B')
    -1.0
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': [1, 2, 3, 4, 5]}
    >>> df = pd.DataFrame(data)
    >>> find_correlation(df, 'A', 'B')
    1.0
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': [2, 3, 4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> find_correlation(df, 'A', 'B')
    1.0
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': [5, 6, 7, 8, 10]}
    >>> df = pd.DataFrame(data)
    >>> find_correlation(df, 'A', 'B')
    0.970725343394151
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': [5, 5, 5, 5, 5]}
    >>> df = pd.DataFrame(data)
    >>> find_correlation(df, 'A', 'B')
    nan
    """
    col1_mean = df[col1].mean()
    col2_mean = df[col2].mean()
    col1_dev = df[col1] - col1_mean
    col2_dev = df[col2] - col2_mean
    numerator = np.sum(col1_dev * col2_dev)
    denominator = np.sqrt(np.sum(col1_dev**2) * np.sum(col2_dev**2))
    return np.nan if denominator == 0 else float(numerator / denominator)


def find_least_square_estimates(
    df: pd.DataFrame, x_col: str, y_col: str
) -> dict[str, float]:
    x_mean = df[x_col].mean()
    y_mean = df[y_col].mean()
    x_dev = df[x_col] - x_mean
    y_dev = df[y_col] - y_mean
    slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2)
    intercept = y_mean - slope * x_mean
    return {
        "slope_estimate": slope,
        "intercept_estimate": intercept,
    }


def find_residuals(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    estimates = find_least_square_estimates(df, x_col, y_col)
    slope_estimate = estimates["slope_estimate"]
    intercept_estimate = estimates["intercept_estimate"]
    residuals = np.sum(
        (df[y_col] - (slope_estimate * df[x_col] + intercept_estimate)) ** 2
    )
    return float(residuals)


# Sum of Squared Error Divided by degrees of freedom (n-2) - mean squared error
def find_residual_squared_error(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> float:
    estimates = find_least_square_estimates(df, x_col, y_col)
    slope_estimate = estimates["slope_estimate"]
    intercept_estimate = estimates["intercept_estimate"]
    residuals = np.sum(
        (df[y_col] - (slope_estimate * df[x_col] + intercept_estimate)) ** 2
    )
    return float(residuals / (df.shape[0] - 2))


# Sum of Squared Residuals - what regression explains
def find_sum_of_squared_residuals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> float:
    estimates = find_least_square_estimates(df, x_col, y_col)
    slope_estimate = estimates["slope_estimate"]
    intercept_estimate = estimates["intercept_estimate"]
    y_mean = df[y_col].mean()

    # fmt: off
    return float(
        np.sum(
            ((slope_estimate * df[x_col] + intercept_estimate) - y_mean) ** 2
        )
    )
    # fmt: on


def compute_f_statistic(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    sum_of_squared_residuals = find_sum_of_squared_residuals(df, x_col, y_col)
    mean_squared_error = find_residual_squared_error(df, x_col, y_col)
    return float(sum_of_squared_residuals / mean_squared_error)


def compute_p_value(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    f_statistic = compute_f_statistic(df, x_col, y_col)
    # return float(1 - f.cdf(f_statistic, 1, df.shape[0] - 2))
    return float(f.sf(f_statistic, 1, df.shape[0] - 2))


def find_r_squared(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    residuals = find_residuals(df, x_col, y_col)
    y_mean = df[y_col].mean()
    total_sum_of_squares = np.sum((df[y_col] - y_mean) ** 2)
    return float(1 - residuals / total_sum_of_squares)


def find_adj_r_squared(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    p: int = 1,
) -> float:
    r_squared = find_r_squared(df, x_col, y_col)
    n = df.shape[0]
    return float(1 - (1 - r_squared) * (n - 1) / (n - p - 1))


def find_standard_errors(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> dict[str, float]:
    variance_estimate = find_residual_squared_error(df, x_col, y_col)
    x_mean = df[x_col].mean()
    x_dev = df[x_col] - x_mean
    standard_error_slope = np.sqrt(variance_estimate / np.sum(x_dev**2))
    standard_error_intercept = np.sqrt(variance_estimate) * np.sqrt(
        1 / df.shape[0] + x_mean**2 / np.sum(x_dev**2)
    )
    return {
        "standard_error_slope": standard_error_slope,
        "standard_error_intercept": standard_error_intercept,
    }


def get_theoretical_quantiles(
    n: int,
    mu: float = 0,
    scale: float = 1,
) -> list[float]:
    arr = np.linspace(0, 1, n + 1)
    quantiles = []
    for i in range(len(arr)):
        if i == len(arr) - 1:
            continue
        midpoint = np.round((arr[i] + arr[i + 1]) / 2, 4)
        quantiles.append(norm.ppf(midpoint, loc=mu, scale=scale))
    return quantiles


def compute_likelihood_reg(
    df: pd.DataFrame,
    predictors: list[str],
    response: str,
) -> float:
    n = df.shape[0]
    p = len(predictors)
    X = df.loc[:, predictors]
    X = sm.add_constant(X)
    y = df[response]
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    residuals = y - y_pred
    rss = (residuals**2).sum()
    error_var = rss / (n - p - 1)

    # fmt: off
    log_likelihood = (
        n * math.log(1 / math.sqrt(2 * math.pi * error_var))
        - rss / (2 * error_var)
    )
    # fmt: on

    # get the model parameters
    params = model.params
    print(f"Intercept: {params['const']}")
    for predictor in predictors:
        print(f"{predictor}: {params[predictor]}")

    return float(log_likelihood)


# compute chatterjee coefficient using dataframe functions
def compute_chatterjee_corr_df(df: pd.DataFrame, x: str, y: str) -> float:
    n = len(df)
    df["y_rank"] = df[y].rank()
    df["x_rank"] = df[x].rank()
    df.sort_values(by="x_rank", inplace=True)
    a = df["y_rank"].diff().abs().sum()
    b = n**2 - 1
    chatt_corr = 1 - ((3 * a) / b)
    return float(chatt_corr)


def compute_chatterjee_corr_np(x: list[int], y: list[int]) -> float:
    n = len(x)
    x_sorted_indices = np.argsort(x)
    y_sorted = np.array(y[x_sorted_indices])
    y_rank = rankdata(y_sorted)
    a = np.abs(np.diff(y_rank)).sum() * n
    l_counts = []
    for i in y_sorted:
        less_than_i = y_sorted[y_sorted < i]
        l_counts.append(len(less_than_i) * (n - len(less_than_i)))
    b = np.sum(l_counts) * 2
    chatt_corr = 1 - (a / b)
    return float(chatt_corr)
