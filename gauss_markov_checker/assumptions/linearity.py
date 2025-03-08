"""
Linearity assumption tests for OLS regression models.

This module provides functions to test the linearity assumption in OLS regression models,
which states that the relationship between the dependent and independent variables
must be linear in parameters.

"""

import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from typing import Dict, Union, Tuple


def reset_test(model: sm.OLS, power: int = 2, test_type: str = 'fitted') -> Dict[str, Union[float, str]]:
    """
    Performs Ramsey's RESET test for linearity.

    The RESET test checks if non-linear combinations of the fitted values
    help explain the response variable. A significant p-value suggests
    a violation of the linearity assumption.

    Args:
        model: A fitted OLS regression model (statsmodels.regression.linear_model.OLS).
        power: The maximum power of fitted values to include. Must be > 1.
        test_type: Type of test to perform ('fitted' or 'princomp').

    Returns:
        Dict containing:
          - 'f_stat': The RESET test F-statistic
          - 'p_value': The p-value for the test
          - 'df': Tuple of degrees of freedom (numerator, denominator)
          - 'conclusion': String interpretation of the test result

    Raises:
        ValueError: If power <= 1 or test_type is invalid
        TypeError: If model is not a fitted OLS model

    Example:
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.get_rdataset("Guerry", "HistData").data
        >>> model = sm.OLS(data['Lottery'], sm.add_constant(data['Pop1831'])).fit()
        >>> reset_test(model)
    """
    if not isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        raise TypeError("Input must be a fitted statsmodels OLS model")
    
    if power <= 1:
        raise ValueError("power must be greater than 1")
        
    if test_type not in ['fitted', 'princomp']:
        raise ValueError("test_type must be either 'fitted' or 'princomp'")

    reset_result = smd.linear_reset(model, power=power, test_type=test_type, use_f=True)
    
    conclusion = (
        "Reject null hypothesis: Non-linear relationship detected"
        if reset_result.pvalue < 0.05
        else "Fail to reject null hypothesis: No significant evidence of non-linearity"
    )
    
    return {
        'f_stat': float(reset_result.fvalue),
        'p_value': float(reset_result.pvalue),
        'df': (int(reset_result.df_num), int(reset_result.df_denom)),
        'conclusion': conclusion
    }


def harvey_collier_test(model: sm.OLS) -> Dict[str, Union[float, str]]:
    """
    Performs Harvey-Collier test for linearity.

    This test is based on recursive residuals and tests whether these
    residuals are significantly different from zero.

    Args:
        model: A fitted OLS regression model.

    Returns:
        Dict containing:
          - 't_stat': The test t-statistic
          - 'p_value': The p-value for the test
          - 'df': Degrees of freedom
          - 'conclusion': String interpretation of the test result

    Raises:
        TypeError: If model is not a fitted OLS model
    """
    if not isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        raise TypeError("Input must be a fitted statsmodels OLS model")

    hc_result = smd.linear_harvey_collier(model)
    
    conclusion = (
        "Reject null hypothesis: Non-linear relationship detected"
        if hc_result.pvalue < 0.05
        else "Fail to reject null hypothesis: No significant evidence of non-linearity"
    )
    
    return {
        't_stat': float(hc_result.statistic),
        'p_value': float(hc_result.pvalue),
        'df': int(hc_result.df),
        'conclusion': conclusion
    }


def rainbow_test(model: sm.OLS, frac: float = 0.5) -> Dict[str, Union[float, str]]:
    """
    Performs Rainbow test for linearity.

    This test orders the data by the fitted values and tests whether the
    parameters are stable across the ordered sample.

    Args:
        model: A fitted OLS regression model.
        frac: Fraction of the sample to use for parameter comparison (0 < frac < 1).

    Returns:
        Dict containing:
          - 'f_stat': The test F-statistic
          - 'p_value': The p-value for the test
          - 'conclusion': String interpretation of the test result

    Raises:
        TypeError: If model is not a fitted OLS model
        ValueError: If frac is not between 0 and 1
    """
    if not isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        raise TypeError("Input must be a fitted statsmodels OLS model")
        
    if not 0 < frac < 1:
        raise ValueError("frac must be between 0 and 1")

    rainbow_result = smd.linear_rainbow(model, frac=frac)
    
    f_stat, p_value = rainbow_result
    
    conclusion = (
        "Reject null hypothesis: Non-linear relationship detected"
        if p_value < 0.05
        else "Fail to reject null hypothesis: No significant evidence of non-linearity"
    )
    
    return {
        'f_stat': float(f_stat),
        'p_value': float(p_value),
        'conclusion': conclusion
    }


def run_all_linearity_tests(model: sm.OLS) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Run all available linearity tests on a fitted model.

    Args:
        model: A fitted OLS regression model.

    Returns:
        Dict containing results from all linearity tests:
          - 'reset': Results from RESET test
          - 'harvey_collier': Results from Harvey-Collier test
          - 'rainbow': Results from Rainbow test
    """
    results = {}
    
    # Run RESET test
    results['reset'] = reset_test(model)
    
    # Run Harvey-Collier test
    results['harvey_collier'] = harvey_collier_test(model)
    
    # Run Rainbow test
    results['rainbow'] = rainbow_test(model)
    
    return results


if __name__ == "__main__":
    # Example usage
    import statsmodels.formula.api as smf
    
    # Load example dataset
    data = sm.datasets.get_rdataset("Guerry", "HistData").data
    
    # Fit a simple model
    model = smf.ols("Lottery ~ Pop1831", data=data).fit()
    
    # Run all linearity tests
    results = run_all_linearity_tests(model)
    
    # Print results
    for test_name, result in results.items():
        print(f"\n{test_name.upper()} Test Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")