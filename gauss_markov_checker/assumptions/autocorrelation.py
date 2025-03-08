"""Tests for autocorrelation in OLS regression models.

This module implements various tests for the autocorrelation assumption in OLS regression,
which states that the error terms should not be correlated with each other across observations.

The module provides several statistical tests:
1. Durbin-Watson test: Tests for first-order autocorrelation - statsmodels
2. Breusch-Godfrey test: Tests for higher-order autocorrelation - statsmodels
3. Ljung-Box test: Tests for autocorrelation at multiple lags - statsmodels
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, List
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox


def durbin_watson_test(
    model_results
) -> Dict[str, Union[float, str]]:
    """
    Perform the Durbin-Watson test for first-order autocorrelation.
    
    The Durbin-Watson test examines whether there is first-order autocorrelation
    in the residuals of a regression model.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Durbin-Watson test statistic
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    The Durbin-Watson statistic ranges from 0 to 4:
    - Values close to 2 indicate no autocorrelation
    - Values toward 0 indicate positive autocorrelation
    - Values toward 4 indicate negative autocorrelation
    
    As a rule of thumb, values between 1.5 and 2.5 generally indicate no concerning autocorrelation.
    
    References
    ----------
    Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least squares regression. II.
    Biometrika, 38(1/2), 159-177.
    """
    # Get residuals from the model
    resid = model_results.resid
    
    # Calculate the Durbin-Watson statistic
    dw_stat = durbin_watson(resid)
    
    # Determine conclusion based on the statistic
    if dw_stat < 1.5:
        conclusion = "Evidence of positive autocorrelation"
    elif dw_stat > 2.5:
        conclusion = "Evidence of negative autocorrelation"
    else:
        conclusion = "No significant evidence of autocorrelation"
    
    return {
        'statistic': dw_stat,
        'conclusion': conclusion
    }


def breusch_godfrey_test(
    model_results, 
    nlags: int = 4
) -> Dict[str, Union[float, str]]:
    """
    Perform the Breusch-Godfrey test for higher-order autocorrelation.
    
    The Breusch-Godfrey test examines whether there is autocorrelation in the residuals
    up to a specified lag order. It is more general than the Durbin-Watson test and
    can detect higher-order autocorrelation.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    nlags : int, optional
        Number of lags to include in the test. Default is 4.
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Lagrange Multiplier test statistic
        - 'p_value': p-value for the LM test
        - 'f_statistic': F-statistic version of the test
        - 'f_p_value': p-value for the F-statistic
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: No autocorrelation up to lag order p
    Alternative hypothesis: Autocorrelation present at some lags
    
    References
    ----------
    Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models.
    Australian Economic Papers, 17(31), 334-355.
    
    Godfrey, L. G. (1978). Testing against general autoregressive and moving average
    error models when the regressors include lagged dependent variables.
    Econometrica, 46(6), 1293-1301.
    """
    # Perform the Breusch-Godfrey test
    lm_stat, lm_pvalue, fvalue, fpvalue = acorr_breusch_godfrey(model_results, nlags=nlags)
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if lm_pvalue < 0.05:
        conclusion = "Reject null hypothesis: Autocorrelation is present"
    else:
        conclusion = "Fail to reject null hypothesis: No significant evidence of autocorrelation"
    
    return {
        'statistic': float(lm_stat),
        'p_value': float(lm_pvalue),
        'f_statistic': float(fvalue),
        'f_p_value': float(fpvalue),
        'conclusion': conclusion
    }


def ljung_box_test(
    model_results, 
    lags: Optional[Union[int, List[int]]] = None,
    model_df: int = 0
) -> Dict[str, Union[float, str]]:
    """
    Perform the Ljung-Box test for autocorrelation at multiple lags.
    
    The Ljung-Box test examines whether there is autocorrelation in the residuals
    at multiple lag orders simultaneously. It is particularly useful for time series data.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    lags : int or list of ints, optional
        Lags to include in the test. If an integer, tests all lags up to that value.
        If a list, tests only those specific lags. Default is None (automatically determined).
    model_df : int, optional
        Degrees of freedom used by the model. For ARMA(p,q) models, this would be p+q.
        Default is 0.
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Test statistic for the first lag
        - 'p_value': p-value for the test
        - 'conclusion': String interpretation of the test result
        - 'full_results': DataFrame with test statistics and p-values for each lag
        
    Notes
    -----
    Null hypothesis: No autocorrelation up to lag k
    Alternative hypothesis: Some autocorrelation present within lags 1 to k
    
    References
    ----------
    Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models.
    Biometrika, 65(2), 297-303.
    """
    # Get residuals from the model
    resid = model_results.resid
    
    # Perform the Ljung-Box test
    results_df = acorr_ljungbox(resid, lags=lags, model_df=model_df, return_df=True)
    
    # Get the results for the first lag (or specified lag)
    first_lag = results_df.index[0]
    lb_stat = float(results_df.loc[first_lag, 'lb_stat'])
    lb_pvalue = float(results_df.loc[first_lag, 'lb_pvalue'])
    
    # Determine conclusion based on p-values (using 0.05 significance level)
    if lb_pvalue < 0.05:
        conclusion = "Reject null hypothesis: Autocorrelation is present"
    else:
        conclusion = "Fail to reject null hypothesis: No significant evidence of autocorrelation"
    
    return {
        'statistic': lb_stat,
        'p_value': lb_pvalue,
        'conclusion': conclusion,
        'full_results': results_df
    }


def run_all_autocorrelation_tests(
    model_results,
    bg_lags: int = 4,
    lb_lags: Optional[Union[int, List[int]]] = None
) -> Dict[str, Dict[str, Union[float, pd.DataFrame, str]]]:
    """
    Run all available autocorrelation tests on a fitted model.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    bg_lags : int, optional
        Number of lags for the Breusch-Godfrey test. Default is 4.
    lb_lags : int or list of ints, optional
        Lags for the Ljung-Box test. Default is None (automatically determined).
        
    Returns
    -------
    Dict[str, Dict[str, Union[float, pd.DataFrame, str]]]
        Dictionary with test names as keys and test results as values
    """
    results = {}
    
    # Run Durbin-Watson test
    results['durbin_watson'] = durbin_watson_test(model_results)
    
    # Run Breusch-Godfrey test
    results['breusch_godfrey'] = breusch_godfrey_test(model_results, nlags=bg_lags)
    
    # Run Ljung-Box test
    results['ljung_box'] = ljung_box_test(model_results, lags=lb_lags)
    
    return results


if __name__ == "__main__":
    # Example using statsmodels dataset
    import statsmodels.api as sm
    from statsmodels.datasets import longley
    
    # Load example dataset
    data = longley.load_pandas()
    
    # Fit OLS model
    model = sm.OLS(data.endog, sm.add_constant(data.exog))
    results = model.fit()
    
    # Print summary of the model
    print("OLS Model Summary:")
    print(results.summary())
    print("\n" + "="*80 + "\n")
    
    # Test for autocorrelation using individual tests
    print("Durbin-Watson Test:")
    dw_results = durbin_watson_test(results)
    print(f"Statistic: {dw_results['statistic']}")
    print(f"Conclusion: {dw_results['conclusion']}")
    print("\n" + "-"*80 + "\n")
    
    print("Breusch-Godfrey Test:")
    bg_results = breusch_godfrey_test(results, nlags=2)
    print(f"Statistic: {bg_results['statistic']}")
    print(f"p-value: {bg_results['p_value']}")
    print(f"F Statistic: {bg_results['f_statistic']}")
    print(f"F p-value: {bg_results['f_p_value']}")
    print(f"Conclusion: {bg_results['conclusion']}")
    print("\n" + "-"*80 + "\n")
    
    print("Ljung-Box Test:")
    lb_results = ljung_box_test(results, lags=[1, 2, 4, 8])
    print("Results by lag:")
    print(lb_results['full_results'])
    print(f"Conclusion: {lb_results['conclusion']}")
    print("\n" + "-"*80 + "\n")
    
    # Run all tests at once
    print("All Autocorrelation Tests:")
    all_results = run_all_autocorrelation_tests(results, bg_lags=2, lb_lags=[1, 2, 4, 8])
    for test_name, test_results in all_results.items():
        print(f"{test_name}: {test_results.get('conclusion', '')}")
