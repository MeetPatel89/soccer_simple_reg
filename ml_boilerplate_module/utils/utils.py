import numpy as np
import pandas as pd


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


def find_variance_estimate(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    estimates = find_least_square_estimates(df, x_col, y_col)
    slope_estimate = estimates["slope_estimate"]
    intercept_estimate = estimates["intercept_estimate"]
    residuals = np.sum(
        (df[y_col] - (slope_estimate * df[x_col] + intercept_estimate)) ** 2
    )
    return float(residuals / (df.shape[0] - 2))


def find_standard_errors(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> dict[str, float]:
    variance_estimate = find_variance_estimate(df, x_col, y_col)
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
