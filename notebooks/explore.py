import pandas as pd
from scipy.stats import t

from ml_boilerplate_module import (
    find_correlation,
    find_least_square_estimates,
    find_standard_errors,
)


def print_metrics(filepath: str, regressors: list[str], label: str) -> None:
    df = pd.read_csv(filepath)

    print("")
    print("Correlation between regressors and label: ")
    for regressor in regressors:
        print("--------")
        print(f"Regressor: {regressor}")
        print(find_correlation(df, regressor, label))

    print("")
    print("Least square estimates: ")
    for regressor in regressors:
        print("--------")
        print(f"Regressor: {regressor}")
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
            f"Interval: {slope - margin_of_error_slope} to "
            f"{slope + margin_of_error_slope}"
        )
        # fmt: on

        # t -statistic for hypothesis testing
        print(f"t-statistic for slope: {slope / slope_std_error}")
        print(
            f"p-value for slope: "
            f"{1 - t.cdf(slope / slope_std_error, df.shape[0] - 2)}"
        )
        print("")
        print(f"Intercept margin of error: {margin_of_error_intercept}")
        print(
            f"Interval: {intercept - margin_of_error_intercept} to "
            f"{intercept + margin_of_error_intercept}"
        )
        print(f"t-statistic for intercept: {intercept / intercept_std_error}")

        # fmt: off
        p_value_intercept = 1 - t.cdf(
            intercept / intercept_std_error, df.shape[0] - 2
        )
        # fmt: on

        print(f"p-value for intercept: {p_value_intercept}")
        print(len(df))
