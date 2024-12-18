import numpy as np
import pandas as pd
from scipy.stats import f, norm


def find_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    col1_mean = df[col1].mean()
    col2_mean = df[col2].mean()
    col1_dev = df[col1] - col1_mean
    col2_dev = df[col2] - col2_mean
    numerator = np.sum(col1_dev * col2_dev)
    denominator = np.sqrt(np.sum(col1_dev**2) * np.sum(col2_dev**2))
    return float(numerator / denominator)


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
