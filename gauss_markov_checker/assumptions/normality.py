"""Tests for normality of residuals in OLS regression models.

This module implements various tests for the normality assumption in OLS regression,
which states that the error terms should be normally distributed. While this assumption
is not required for OLS estimators to be unbiased, it is necessary for valid hypothesis
testing and confidence intervals.

The module provides several statistical tests:
1. Jarque-Bera test: Tests normality based on skewness and kurtosis - statsmodels
2. Shapiro-Wilk test: Powerful test for normality, especially for small samples - scipy
3. D'Agostino-Pearson test: Omnibus test combining skewness and kurtosis - scipy
4. Anderson-Darling test: Sensitive to deviations in the tails of the distribution - scipy
5. Omnibus test: Chi-squared test for normality - statsmodels
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, List
import statsmodels.stats.api as sms
from statsmodels.regression.linear_model import RegressionResults
from scipy.stats import shapiro, normaltest, anderson


def jarque_bera_test(
    model_results: RegressionResults
) -> Dict[str, Union[float, str]]:
    """
    Perform the Jarque-Bera test for normality of residuals.
    
    The Jarque-Bera test examines whether the residuals have skewness and kurtosis 
    matching a normal distribution. The null hypothesis is that the data is normally distributed.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Jarque-Bera test statistic
        - 'p_value': p-value
        - 'skewness': Skewness of residuals
        - 'kurtosis': Kurtosis of residuals
        - 'conclusion': String interpretation of the test result
        
    Hypotheses
    ----------
    - H0: Residuals are normally distributed
    - H1: Residuals are not normally distributed
    
    Interpretation
    -------------
    - p-value < 0.05: Reject the null hypothesis, suggesting non-normality
    - p-value >= 0.05: Fail to reject the null hypothesis, suggesting normality
    
    Notes
    -----
    - Kurtosis reported is the sample kurtosis, not excess kurtosis
    - For large samples (n>30), non-normality may not be problematic due to the Central Limit Theorem
    """
    jb_stat, jb_pval, skew, kurtosis = sms.jarque_bera(model_results.resid)
    
    if jb_pval < 0.05:
        conclusion = "Reject normality: The residuals do not appear to be normally distributed."
    else:
        conclusion = "Fail to reject normality: The residuals appear to be normally distributed."
    
    return {
        'statistic': jb_stat,
        'p_value': jb_pval,
        'skewness': skew,
        'kurtosis': kurtosis,
        'conclusion': conclusion
    }


def shapiro_wilk_test(
    model_results: RegressionResults
) -> Dict[str, Union[float, str]]:
    """
    Perform the Shapiro-Wilk test for normality of residuals.
    
    The Shapiro-Wilk test is one of the most powerful tests for normality, 
    especially for small to medium-sized samples. The null hypothesis is that 
    the data is normally distributed.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Shapiro-Wilk test statistic (W)
        - 'p_value': p-value
        - 'conclusion': String interpretation of the test result
        
    Hypotheses
    ----------
    - H0: Residuals are normally distributed
    - H1: Residuals are not normally distributed
    
    Interpretation
    -------------
    - p-value < 0.05: Reject the null hypothesis, suggesting non-normality
    - p-value >= 0.05: Fail to reject the null hypothesis, suggesting normality
    
    Notes
    -----
    - The test works best for sample sizes between 3 and 5000
    - For large samples, the test may reject normality due to small deviations
    """
    sw_stat, sw_pval = shapiro(model_results.resid)
    
    if sw_pval < 0.05:
        conclusion = "Reject normality: The residuals do not appear to be normally distributed."
    else:
        conclusion = "Fail to reject normality: The residuals appear to be normally distributed."
    
    return {
        'statistic': sw_stat,
        'p_value': sw_pval,
        'conclusion': conclusion
    }


def dagostino_pearson_test(
    model_results: RegressionResults
) -> Dict[str, Union[float, str]]:
    """
    Perform the D'Agostino-Pearson omnibus test for normality of residuals.
    
    This test combines skew and kurtosis to produce an omnibus test of normality.
    The null hypothesis is that the data is normally distributed.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': D'Agostino-Pearson test statistic
        - 'p_value': p-value
        - 'conclusion': String interpretation of the test result
        
    Hypotheses
    ----------
    - H0: Residuals are normally distributed
    - H1: Residuals are not normally distributed
    
    Interpretation
    -------------
    - p-value < 0.05: Reject the null hypothesis, suggesting non-normality
    - p-value >= 0.05: Fail to reject the null hypothesis, suggesting normality
    
    Notes
    -----
    - The test requires at least 8 observations
    - The test statistic follows a chi-squared distribution with 2 degrees of freedom
    """
    dp_stat, dp_pval = normaltest(model_results.resid)
    
    if dp_pval < 0.05:
        conclusion = "Reject normality: The residuals do not appear to be normally distributed."
    else:
        conclusion = "Fail to reject normality: The residuals appear to be normally distributed."
    
    return {
        'statistic': dp_stat,
        'p_value': dp_pval,
        'conclusion': conclusion
    }


def anderson_darling_test(
    model_results: RegressionResults
) -> Dict[str, Union[float, np.ndarray, List[float], str]]:
    """
    Perform the Anderson-Darling test for normality of residuals.
    
    The Anderson-Darling test is a modification of the Kolmogorov-Smirnov test
    that gives more weight to the tails of the distribution. The null hypothesis 
    is that the data is normally distributed.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Union[float, np.ndarray, List[float], str]]
        Dictionary containing:
        - 'statistic': Anderson-Darling test statistic
        - 'critical_values': Critical values for different significance levels
        - 'significance_level': Significance levels (percentages)
        - 'conclusion': String interpretation of the test result
        
    Hypotheses
    ----------
    - H0: Residuals are normally distributed
    - H1: Residuals are not normally distributed
    
    Interpretation
    -------------
    - If the test statistic exceeds the critical value at a given significance level,
      reject the null hypothesis at that significance level
    
    Notes
    -----
    - The test is more sensitive to deviations in the tails of the distribution
    - Critical values are provided for significance levels: 15%, 10%, 5%, 2.5%, 1%
    """
    ad_result = anderson(model_results.resid, dist='norm')
    
    # Check if the test statistic exceeds the critical value at 5% significance
    if ad_result.statistic > ad_result.critical_values[2]:  # Index 2 corresponds to 5% significance
        conclusion = "Reject normality: The residuals do not appear to be normally distributed (at 5% significance)."
    else:
        conclusion = "Fail to reject normality: The residuals appear to be normally distributed (at 5% significance)."
    
    return {
        'statistic': ad_result.statistic,
        'critical_values': ad_result.critical_values,
        'significance_level': ad_result.significance_level,
        'conclusion': conclusion
    }


def omnibus_test(
    model_results: RegressionResults
) -> Dict[str, Union[float, str]]:
    """
    Perform the Omnibus test for normality of residuals.
    
    The Omnibus test is a chi-squared test of the skewness and kurtosis of residuals.
    The null hypothesis is that the data is normally distributed.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Omnibus test statistic
        - 'p_value': p-value
        - 'conclusion': String interpretation of the test result
        
    Hypotheses
    ----------
    - H0: Residuals are normally distributed
    - H1: Residuals are not normally distributed
    
    Interpretation
    -------------
    - p-value < 0.05: Reject the null hypothesis, suggesting non-normality
    - p-value >= 0.05: Fail to reject the null hypothesis, suggesting normality
    """
    omni_stat, omni_pval = sms.omni_normtest(model_results.resid)
    
    if omni_pval < 0.05:
        conclusion = "Reject normality: The residuals do not appear to be normally distributed."
    else:
        conclusion = "Fail to reject normality: The residuals appear to be normally distributed."
    
    return {
        'statistic': omni_stat,
        'p_value': omni_pval,
        'conclusion': conclusion
    }


def run_all_normality_tests(
    model_results: RegressionResults
) -> Dict[str, Dict[str, Union[float, str, np.ndarray, List[float]]]]:
    """
    Run all available normality tests on the residuals of a regression model.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
        
    Returns
    -------
    Dict[str, Dict[str, Union[float, str, np.ndarray, List[float]]]]
        Dictionary containing results from all normality tests
    """
    results = {
        'jarque_bera': jarque_bera_test(model_results),
        'shapiro_wilk': shapiro_wilk_test(model_results),
        'dagostino_pearson': dagostino_pearson_test(model_results),
        'anderson_darling': anderson_darling_test(model_results),
        'omnibus': omnibus_test(model_results)
    }
    
    return results


if __name__ == "__main__":
    # Example usage with a simple dataset
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    # Load example dataset
    data = sm.datasets.get_rdataset("Guerry", "HistData").data
    
    # Fit a simple OLS model
    model = smf.ols(formula="Lottery ~ Literacy + np.log(Pop1831)", data=data).fit()
    
    # Run all normality tests
    normality_results = run_all_normality_tests(model)
    
    # Print the results
    print("Normality Tests for OLS Residuals\n")
    print("=" * 80)
    
    for test_name, test_results in normality_results.items():
        print(f"\n{test_name.replace('_', ' ').title()} Test:")
        for key, value in test_results.items():
            if key != 'conclusion':
                if isinstance(value, (np.ndarray, list)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value:.4f}")
        print(f"  {test_results['conclusion']}")
    
    print("\n" + "=" * 80)
    
    # Summary of the overall normality assessment
    print("\nOverall Assessment:")
    reject_count = sum(1 for test in normality_results.values() 
                      if "Reject normality" in test['conclusion'])
    
    if reject_count > len(normality_results) / 2:
        print("The majority of tests suggest that the residuals are NOT normally distributed.")
        print("Consider transforming variables or using robust regression methods.")
    else:
        print("The majority of tests suggest that the residuals are normally distributed.")
        print("The normality assumption appears to be satisfied.")
