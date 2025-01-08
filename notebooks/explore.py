from typing import Optional

import pandas as pd
from scipy.stats import t

from ml_boilerplate_module import (  # noqa: F401
    compute_chatterjee_corr_df,
    compute_chatterjee_corr_np,
    compute_f_statistic,
    compute_likelihood_reg,
    compute_p_value,
    find_adj_r_squared,
    find_correlation,
    find_least_square_estimates,
    find_r_squared,
    find_residual_squared_error,
    find_standard_errors,
    get_theoretical_quantiles,
    split_dataframe_k_folds,
)


def print_metrics(
    regressors: list[str],
    label: str,
    df: pd.DataFrame = None,
    filepath: Optional[str] = None,
) -> None:

    if df is None and filepath is None:
        raise ValueError("Either df or filepath should be provided")
    if filepath is not None:
        df = pd.read_csv(filepath)

    print("")
    print("Correlation between regressors and label: ")
    for regressor in regressors:
        print("--------")
        print(f"Regressor: {regressor}")
        print(f"Correlation: {find_correlation(df, regressor, label)}")
        print("--------")
        residual_squared_errror = find_residual_squared_error(
            df,
            regressor,
            label,
        )
        print(f"Residual squared error: {residual_squared_errror}")
        print(f"Residual_standard_error: {residual_squared_errror**0.5}")

        r_squared = find_r_squared(df, regressor, label)
        print(f"R-squared: {r_squared}")
        adj_r_squared = find_adj_r_squared(df, regressor, label)
        print(f"Adjusted R-squared: {adj_r_squared}")

        estimates = find_least_square_estimates(df, regressor, label)
        slope = estimates["slope_estimate"]
        intercept = estimates["intercept_estimate"]
        print(f"Slope estimate: {slope}")
        print(f"Intercept estimate: {intercept}")
        standard_errors = find_standard_errors(df, regressor, label)
        slope_std_error = standard_errors["standard_error_slope"]
        intercept_std_error = standard_errors["standard_error_intercept"]
        print(f"Slope standard error: {slope_std_error}")
        print(f"Intercept standard error: {intercept_std_error}")

        # t value for 95% confidence interval
        t_95 = t.ppf(0.975, df.shape[0] - 2)
        margin_of_error_slope = t_95 * slope_std_error
        margin_of_error_intercept = t_95 * intercept_std_error

        print("")
        print(f"T-value for 95% confidence interval: {t_95}")
        print(f"Slope margin of error: {margin_of_error_slope}")

        # fmt: off
        print(
            f"Interval of slope for 95 % confidence interval: "
            f"{slope - margin_of_error_slope} to "
            f"{slope + margin_of_error_slope}"
        )
        # fmt: on

        # t -statistic for hypothesis testing
        print(f"t-statistic for slope: {slope / slope_std_error}")
        print(
            f"p-value for slope: "
            f"{2*(1 - t.cdf(slope / slope_std_error, df.shape[0] - 2))}"
        )
        print("")
        print(f"Intercept margin of error: {margin_of_error_intercept}")
        print(
            f"Interval of intercept for 95 % confidence interval: "
            f"{intercept - margin_of_error_intercept} to "
            f"{intercept + margin_of_error_intercept}"
        )
        print(f"t-statistic for intercept: {intercept / intercept_std_error}")

        # fmt: off
        p_value_intercept = 2*(1 - t.cdf(
            intercept / intercept_std_error, df.shape[0] - 2
        ))
        # fmt: on

        print(f"p-value for intercept: {p_value_intercept}")

        print("")
        print("--------")
        print("")
        print(f"F-statistic: {compute_f_statistic(df, regressor, label)}")
        p_value_f_statistic = compute_p_value(df, regressor, label)
        print(f"P-value for F-statistic: {p_value_f_statistic}")
        print("")
        print("--------")
