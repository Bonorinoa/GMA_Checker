"""
Linearity assumption tests for OLS regression models.

This module provides functions to test the linearity assumption in OLS regression models,
which states that the relationship between the dependent and independent variables
must be linear in parameters.

"""

import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from typing import Tuple


def ramsey_reset(model: sm.OLS, power: int = 2) -> Tuple[float, float, float, float]:
    """
    Performs Ramsey's RESET test for linearity.

    The RESET test checks if non-linear combinations of the fitted values
    help explain the response variable.  A significant p-value suggests
    a violation of the linearity assumption.

    Args:
        model: A fitted OLS regression model (statsmodels.regression.linear_model.OLS).
        power: The maximum power of the fitted values to include in the test.
               Defaults to 3.

    Returns:
        A tuple containing:
          - The RESET test statistic (F-value).
          - The p-value associated with the F-statistic.
          - The degrees of freedom for the numerator.
          - The degrees of freedom for the denominator.

    Hypotheses:
        - Null Hypothesis (H0): The model is correctly specified (linear relationship).
        - Alternative Hypothesis (H1): The model is misspecified (non-linear relationship).

    Interpretation:
        - A p-value < 0.05 suggests rejecting the null hypothesis, indicating
          evidence of non-linearity.  This implies the current linear
          specification may be inadequate.
        - A p-value >= 0.05 suggests that there is no significant evidence
          against the linear specification.

    Example:
        >>> import statsmodels.formula.api as smf
        >>> data = sm.datasets.get_rdataset(" હાઉસprices", "AER", cache=True).data
        >>> model = smf.ols("price ~ lotsize + bedrooms", data=data).fit()
        >>> reset_result = ramsey_reset(model)
        >>> print(reset_result)

    References:
        - Ramsey, J. B. (1969). Tests for Specification Errors in Classical Linear
          Least-Squares Regression Analysis. Journal of the Royal Statistical
          Society. Series B (Methodological), 31(2), 350–371.
        - https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_reset.html
    """
    reset_test = smd.linear_reset(res=model, power=power, test_type="fitted", use_f=True)
    return reset_test.fvalue, reset_test.pvalue, reset_test.df_num, reset_test.df_denom



if __name__ == "__main__":
    # Example usage with a simple dataset
    import statsmodels.formula.api as smf

    # Load example dataset (House Prices)
    houseprices = sm.datasets.get_rdataset(dataname="HousePrices", package="AER", cache=True).data

    # Fit a simple OLS model
    mlr = smf.ols(formula="price ~ lotsize + bedrooms", data=houseprices).fit()

    # Perform the RESET test
    reset_result = ramsey_reset(mlr)

    # Print the results with interpretation
    print(f"Ramsey RESET Test Results:")
    print(f"  F-statistic: {reset_result[0]:.4f}")
    print(f"  P-value: {reset_result[1]:.4f}")
    print(f"  Numerator df: {reset_result[2]:.0f}")
    print(f"  Denominator df: {reset_result[3]:.0f}")

    if reset_result[1] < 0.05:
        print("\nInterpretation: The p-value is less than 0.05, suggesting a rejection of the null hypothesis.")
        print("This indicates evidence of non-linearity in the model.")
    else:
        print("\nInterpretation: The p-value is greater than or equal to 0.05, suggesting that we fail to reject the null hypothesis.")
        print("This indicates no significant evidence of non-linearity in the model.")