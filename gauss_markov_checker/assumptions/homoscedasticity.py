"""Tests for homoscedasticity in OLS regression models.

This module implements various tests for the homoscedasticity assumption in OLS regression,
which states that the variance of the error terms should be constant across all observations.

The module provides several statistical tests:
1. Breusch-Pagan test: Tests for linear forms of heteroscedasticity - statsmodels
2. White test: A more general test that includes squared terms and cross products - statsmodels
3. Goldfeld-Quandt test: Particularly useful for time-series data - statsmodels
4. Barlett's test: the null hypothesis that all input samples are from populations with equal variances (significant deviations from normality) - scipy
5. Levene's test: the null hypothesis that all input samples are from populations with equal variances (significantly non-normal populations) - scipy
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, List
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import bartlett, levene


def breusch_pagan_test(
    model_results, 
    exog: Optional[np.ndarray] = None
) -> Dict[str, Union[float, str]]:
    """
    Perform the Breusch-Pagan test for heteroscedasticity.
    
    The Breusch-Pagan test examines whether the variance of the errors from a regression 
    is dependent on the values of the independent variables. The null hypothesis is 
    homoscedasticity (constant variance).
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    exog : np.ndarray, optional
        If provided, uses these variables for the test. If None, uses the model's exog.
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Lagrange Multiplier statistic
        - 'p_value': p-value
        - 'f_statistic': F-statistic version of the test
        - 'f_p_value': p-value for F-statistic
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: Homoscedasticity is present (constant variance)
    Alternative hypothesis: Heteroscedasticity is present (non-constant variance)
    
    References
    ----------
    Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and
    random coefficient variation. Econometrica, 47(5), 1287-1294.
    """
    # Get residuals from the model results
    resid = model_results.resid
    
    # If exog is not provided, use the model's exog
    if exog is None:
        exog = model_results.model.exog
    
    # Ensure exog has a constant term
    if exog.ndim > 1 and exog.shape[1] > 0:
        if not np.all(exog[:, 0] == 1):
            exog = add_constant(exog)
    else:
        exog = add_constant(exog)
    
    # Perform the Breusch-Pagan test
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(resid, exog)
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if lm_pvalue < 0.05:
        conclusion = "Reject null hypothesis: Heteroscedasticity is present"
    else:
        conclusion = "Fail to reject null hypothesis: No significant evidence of heteroscedasticity"
    
    return {
        'statistic': lm,
        'p_value': lm_pvalue,
        'f_statistic': fvalue,
        'f_p_value': f_pvalue,
        'conclusion': conclusion
    }


def white_test(
    model_results
) -> Dict[str, Union[float, str]]:
    """
    Perform White's test for heteroscedasticity.
    
    White's test is a statistical test that establishes whether the variance of the errors 
    in a regression model is constant (homoscedasticity). It's more general than the 
    Breusch-Pagan test as it includes squared terms and cross products.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': White's test statistic
        - 'p_value': p-value
        - 'f_statistic': F-statistic version of the test
        - 'f_p_value': p-value for F-statistic
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: Homoscedasticity is present (constant variance)
    Alternative hypothesis: Heteroscedasticity is present (non-constant variance)
    
    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a
    direct test for heteroskedasticity. Econometrica, 48(4), 817-838.
    """
    # Get residuals and exogenous variables from the model
    resid = model_results.resid
    exog = model_results.model.exog
    
    # Perform White's test
    lm, lm_pvalue, fvalue, f_pvalue = het_white(resid, exog)
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if lm_pvalue < 0.05:
        conclusion = "Reject null hypothesis: Heteroscedasticity is present"
    else:
        conclusion = "Fail to reject null hypothesis: No significant evidence of heteroscedasticity"
    
    return {
        'statistic': lm,
        'p_value': lm_pvalue,
        'f_statistic': fvalue,
        'f_p_value': f_pvalue,
        'conclusion': conclusion
    }


def goldfeld_quandt_test(
    model_results, 
    alternative: str = 'two-sided',
    split_pct: float = 0.5
) -> Dict[str, Union[float, str]]:
    """
    Perform the Goldfeld-Quandt test for heteroscedasticity.
    
    The Goldfeld-Quandt test divides the data into two groups and compares the residual 
    variances. It's particularly useful for detecting heteroscedasticity that varies 
    systematically with one of the regressors.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    alternative : str, optional
        Alternative hypothesis, one of 'increasing', 'decreasing', or 'two-sided'.
        Default is 'two-sided'.
    split_pct : float, optional
        Percentage of observations to include in each subsample. Default is 0.5.
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Goldfeld-Quandt test statistic
        - 'p_value': p-value
        - 'alternative': Alternative hypothesis used
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: Homoscedasticity is present (constant variance)
    Alternative hypothesis: Heteroscedasticity is present (non-constant variance)
    
    References
    ----------
    Goldfeld, S. M., & Quandt, R. E. (1965). Some tests for homoscedasticity.
    Journal of the American Statistical Association, 60(310), 539-547.
    """
    # Get dependent and independent variables from the model
    y = model_results.model.endog
    X = model_results.model.exog
    
    # Perform the Goldfeld-Quandt test
    statistic, p_value, alternative = het_goldfeldquandt(
        y, X, alternative=alternative, split=split_pct
    )
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if p_value < 0.05:
        conclusion = "Reject null hypothesis: Heteroscedasticity is present"
    else:
        conclusion = "Fail to reject null hypothesis: No significant evidence of heteroscedasticity"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'alternative': alternative,
        'conclusion': conclusion
    }


def bartlett_test(
    model_results, 
    groups: List[np.ndarray]
) -> Dict[str, Union[float, str]]:
    """
    Perform Bartlett's test for equal variances across groups of residuals.
    
    Bartlett's test checks if multiple samples have equal variances. This implementation
    applies it to residuals grouped by some criterion (e.g., quantiles of a predictor).
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    groups : List[np.ndarray]
        List of arrays containing the indices for each group of residuals to compare
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Bartlett's test statistic
        - 'p_value': p-value
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: All groups have equal variances
    Alternative hypothesis: At least one group has a different variance
    
    This test is sensitive to departures from normality.
    
    References
    ----------
    Bartlett, M. S. (1937). Properties of sufficiency and statistical tests.
    Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences,
    160(901), 268-282.
    """
    # Get residuals from the model
    resid = model_results.resid
    
    # Create groups of residuals
    grouped_resids = [resid[group] for group in groups]
    
    # Perform Bartlett's test
    statistic, p_value = bartlett(*grouped_resids)
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if p_value < 0.05:
        conclusion = "Reject null hypothesis: Heteroscedasticity is present"
    else:
        conclusion = "Fail to reject null hypothesis: No significant evidence of heteroscedasticity"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'conclusion': conclusion
    }


def levene_test(
    model_results, 
    groups: List[np.ndarray],
    center: str = 'median'
) -> Dict[str, Union[float, str]]:
    """
    Perform Levene's test for equal variances across groups of residuals.
    
    Levene's test checks if multiple samples have equal variances. This implementation
    applies it to residuals grouped by some criterion (e.g., quantiles of a predictor).
    It's more robust to departures from normality than Bartlett's test.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    groups : List[np.ndarray]
        List of arrays containing the indices for each group of residuals to compare
    center : str, optional
        Which function of the data to use in the test. The default is 'median'.
        Other options are 'mean', 'trimmed'.
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Levene's test statistic
        - 'p_value': p-value
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: All groups have equal variances
    Alternative hypothesis: At least one group has a different variance
    
    This test is less sensitive to departures from normality than Bartlett's test.
    
    References
    ----------
    Levene, H. (1960). Robust tests for equality of variances. In Contributions
    to Probability and Statistics: Essays in Honor of Harold Hotelling, ed.
    I. Olkin et al., 278-292. Stanford, CA: Stanford University Press.
    """
    # Get residuals from the model
    resid = model_results.resid
    
    # Create groups of residuals
    grouped_resids = [resid[group] for group in groups]
    
    # Perform Levene's test
    statistic, p_value = levene(*grouped_resids, center=center)
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if p_value < 0.05:
        conclusion = "Reject null hypothesis: Heteroscedasticity is present"
    else:
        conclusion = "Fail to reject null hypothesis: No significant evidence of heteroscedasticity"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'conclusion': conclusion
    }


def create_groups_by_fitted_values(
    model_results, 
    n_groups: int = 4
) -> List[np.ndarray]:
    """
    Create groups of observations based on quantiles of fitted values.
    
    This is a helper function for tests that require grouping observations.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    n_groups : int, optional
        Number of groups to create. Default is 4.
        
    Returns
    -------
    List[np.ndarray]
        List of arrays containing the indices for each group
    """
    # Get fitted values
    fitted = model_results.fittedvalues
    
    # Create quantile-based groups
    quantiles = np.linspace(0, 1, n_groups + 1)
    thresholds = [np.quantile(fitted, q) for q in quantiles]
    
    groups = []
    for i in range(n_groups):
        if i == n_groups - 1:
            # Include the upper bound in the last group
            group_indices = np.where((fitted >= thresholds[i]) & 
                                    (fitted <= thresholds[i+1]))[0]
        else:
            group_indices = np.where((fitted >= thresholds[i]) & 
                                    (fitted < thresholds[i+1]))[0]
        groups.append(group_indices)
    
    return groups


def run_all_homoscedasticity_tests(
    model_results,
    n_groups: int = 4
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Run all available homoscedasticity tests on a fitted model.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    n_groups : int, optional
        Number of groups to create for group-based tests. Default is 4.
        
    Returns
    -------
    Dict[str, Dict[str, Union[float, str]]]
        Dictionary with test names as keys and test results as values
    """
    results = {}
    
    # Run Breusch-Pagan test
    results['breusch_pagan'] = breusch_pagan_test(model_results)
    
    # Run White test
    results['white'] = white_test(model_results)
    
    # Run Goldfeld-Quandt test
    results['goldfeld_quandt'] = goldfeld_quandt_test(model_results)
    
    # Create groups for Bartlett and Levene tests
    groups = create_groups_by_fitted_values(model_results, n_groups)
    
    # Run Bartlett test
    results['bartlett'] = bartlett_test(model_results, groups)
    
    # Run Levene test
    results['levene'] = levene_test(model_results, groups)
    
    return results


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from statsmodels.regression.linear_model import OLS
    
    # Generate sample data with heteroscedasticity
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    # Increasing variance with x
    e = np.random.normal(0, 0.5 + 0.5 * x, n)
    y = 2 + 3 * x + e
    
    # Create and fit OLS model
    X = np.column_stack((np.ones(n), x))
    model_fit = OLS(y, X).fit()
    
    # Run all homoscedasticity tests
    results = run_all_homoscedasticity_tests(model_fit)
    
    # Print results
    for test_name, test_results in results.items():
        print(f"\n{test_name.upper()} TEST:")
        for key, value in test_results.items():
            print(f"  {key}: {value}")
    
    # Generate homoscedastic data for comparison
    print("\n\nCOMPARISON WITH HOMOSCEDASTIC DATA:")
    e_homo = np.random.normal(0, 1, n)
    y_homo = 2 + 3 * x + e_homo
    model_homo = OLS(y_homo, X).fit()
    
    # Run Breusch-Pagan test on homoscedastic data
    bp_result = breusch_pagan_test(model_homo)
    print("\nBREUSCH-PAGAN TEST (HOMOSCEDASTIC DATA):")
    for key, value in bp_result.items():
        print(f"  {key}: {value}")
    
    # Add an example with pandas DataFrame
    print("\n\nEXAMPLE WITH PANDAS DATAFRAME:")
    import pandas as pd
    
    # Create a DataFrame
    df = pd.DataFrame({
        'x1': x,
        'x2': x**2,
        'y': y
    })
    
    # Fit model using formula
    import statsmodels.formula.api as smf
    formula_model = smf.ols('y ~ x1 + x2', data=df).fit()
    
    # Run Breusch-Pagan test
    bp_formula_result = breusch_pagan_test(formula_model)
    print("\nBREUSCH-PAGAN TEST (FORMULA MODEL):")
    for key, value in bp_formula_result.items():
        print(f"  {key}: {value}")
