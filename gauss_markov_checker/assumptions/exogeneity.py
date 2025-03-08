"""Tests for exogeneity in regression models.

This module implements various tests for the exogeneity assumption in regression models,
which states that the independent variables should not be correlated with the error term.

The module provides several statistical tests:
1. Wu-Hausman test: Tests whether endogenous regressors are actually exogenous
2. Wooldridge's regression test: Alternative form of the Wu-Hausman test
3. Sargan test: Tests the validity of instrumental variables (overidentification)
4. Durbin-Wu-Hausman test: Tests for endogeneity in a regression estimated via instrumental variables

These tests are particularly important for instrumental variable estimation and
ensuring the consistency of OLS estimators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, List
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv.model import IV2SLS
from statsmodels.tools.tools import add_constant


def wu_hausman_test(
    iv_model_results
) -> Dict[str, Union[float, str]]:
    """
    Perform the Wu-Hausman test for exogeneity.
    
    The Wu-Hausman test examines whether variables that are treated as endogenous
    in an instrumental variables regression could actually be treated as exogenous.
    
    Parameters
    ----------
    iv_model_results : linearmodels.iv.model.IVResults
        Fitted IV2SLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Wu-Hausman test statistic
        - 'p_value': p-value
        - 'df': Degrees of freedom
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: All endogenous variables are exogenous
    Alternative hypothesis: At least one endogenous variable is not exogenous
    
    References
    ----------
    Hausman, J. A. (1978). Specification tests in econometrics. Econometrica, 46(6), 1251-1271.
    Wu, D. M. (1973). Alternative tests of independence between stochastic regressors and
    disturbances. Econometrica, 41(4), 733-750.
    """
    # Get Wu-Hausman test results
    wu_hausman_results = iv_model_results.wu_hausman()
    
    # Extract statistic, p-value, and distribution
    statistic = wu_hausman_results.stat
    p_value = wu_hausman_results.pval
    df_text = wu_hausman_results.dist_name
    
    # Parse degrees of freedom from distribution text (e.g., "F(1,542)")
    df = df_text.replace("F(", "").replace(")", "")
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if p_value < 0.05:
        conclusion = "Reject null hypothesis: At least one endogenous variable is not exogenous"
    else:
        conclusion = "Fail to reject null hypothesis: All endogenous variables can be treated as exogenous"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'df': df,
        'conclusion': conclusion
    }


def wooldridge_regression_test(
    iv_model_results
) -> Dict[str, Union[float, str]]:
    """
    Perform Wooldridge's regression test for exogeneity.
    
    This test is an alternative form of the Wu-Hausman test that uses a regression-based
    approach to test for endogeneity.
    
    Parameters
    ----------
    iv_model_results : linearmodels.iv.model.IVResults
        Fitted IV2SLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Wooldridge's regression test statistic
        - 'p_value': p-value
        - 'df': Degrees of freedom
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: Endogenous variables are exogenous
    Alternative hypothesis: Endogenous variables are not exogenous
    
    References
    ----------
    Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data.
    MIT press.
    """
    # Get Wooldridge regression test results
    wooldridge_results = iv_model_results.wooldridge_regression
    
    # Extract statistic, p-value, and distribution
    statistic = wooldridge_results.stat
    p_value = wooldridge_results.pval
    df_text = wooldridge_results.dist_name
    
    # Parse degrees of freedom from distribution text (e.g., "chi2(1)")
    df = df_text.replace("chi2(", "").replace(")", "")
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if p_value < 0.05:
        conclusion = "Reject null hypothesis: Endogenous variables are not exogenous"
    else:
        conclusion = "Fail to reject null hypothesis: Endogenous variables can be treated as exogenous"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'df': df,
        'conclusion': conclusion
    }


def sargan_test(
    iv_model_results
) -> Dict[str, Union[float, str]]:
    """
    Perform Sargan's test for overidentification.
    
    The Sargan test examines the validity of instrumental variables by testing whether
    the instruments are uncorrelated with the error term. This test is only applicable
    when there are more instruments than endogenous variables (overidentified model).
    
    Parameters
    ----------
    iv_model_results : linearmodels.iv.model.IVResults
        Fitted IV2SLS model results
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'statistic': Sargan test statistic
        - 'p_value': p-value
        - 'df': Degrees of freedom
        - 'conclusion': String interpretation of the test result
        
    Notes
    -----
    Null hypothesis: The model is not overidentified (instruments are valid)
    Alternative hypothesis: The model is overidentified (at least one instrument is invalid)
    
    References
    ----------
    Sargan, J. D. (1958). The estimation of economic relationships using instrumental
    variables. Econometrica, 26(3), 393-415.
    """
    # Get Sargan test results
    sargan_results = iv_model_results.sargan
    
    # Extract statistic, p-value, and distribution
    statistic = sargan_results.stat
    p_value = sargan_results.pval
    df_text = sargan_results.dist_name
    
    # Parse degrees of freedom from distribution text (e.g., "chi2(1)")
    df = df_text.replace("chi2(", "").replace(")", "")
    
    # Determine conclusion based on p-value (using 0.05 significance level)
    if p_value < 0.05:
        conclusion = "Reject null hypothesis: At least one instrument is invalid"
    else:
        conclusion = "Fail to reject null hypothesis: All instruments appear valid"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'df': df,
        'conclusion': conclusion
    }


def run_all_exogeneity_tests(
    iv_model_results
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Run all available exogeneity tests on a fitted IV model.
    
    Parameters
    ----------
    iv_model_results : linearmodels.iv.model.IVResults
        Fitted IV2SLS model results
        
    Returns
    -------
    Dict[str, Dict[str, Union[float, str]]]
        Dictionary with test names as keys and test results as values
    """
    results = {}
    
    # Run Wu-Hausman test
    results['wu_hausman'] = wu_hausman_test(iv_model_results)
    
    # Run Wooldridge regression test
    results['wooldridge_regression'] = wooldridge_regression_test(iv_model_results)
    
    # Run Sargan test (only if model is overidentified)
    try:
        results['sargan'] = sargan_test(iv_model_results)
    except (AttributeError, ValueError) as e:
        # Model might be exactly identified or underidentified
        results['sargan'] = {
            'error': str(e),
            'conclusion': "Sargan test not applicable - model may be exactly identified or underidentified"
        }
    
    return results


def compare_ols_iv(
    data: pd.DataFrame,
    formula: str,
    endog_vars: List[str],
    instrument_vars: List[str]
) -> Dict[str, Union[object, Dict]]:
    """
    Compare OLS and IV estimations and run exogeneity tests.
    
    This function fits both OLS and IV models, then runs exogeneity tests to determine
    whether the IV approach is necessary.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all variables
    formula : str
        Formula for the OLS model (e.g., "y ~ x1 + x2")
    endog_vars : List[str]
        List of endogenous variable names
    instrument_vars : List[str]
        List of instrument variable names
        
    Returns
    -------
    Dict[str, Union[object, Dict]]
        Dictionary containing:
        - 'ols_results': OLS model results
        - 'iv_results': IV model results
        - 'exogeneity_tests': Results of exogeneity tests
        
    Notes
    -----
    This function provides a convenient way to compare OLS and IV estimations
    and determine whether the IV approach is necessary based on exogeneity tests.
    """
    # Fit OLS model
    ols_results = smf.ols(formula=formula, data=data).fit()
    
    # Extract dependent variable name from formula
    dependent_var = formula.split('~')[0].strip()
    
    # Prepare data for IV model
    # Get exogenous variables (excluding the endogenous ones)
    exog_vars = []
    for var in ols_results.model.exog_names:
        if var == 'Intercept':
            continue
        if var not in endog_vars:
            exog_vars.append(var)
    
    # Add constant if needed
    data_iv = data.copy()
    if 'const' not in data_iv.columns:
        data_iv = add_constant(data_iv, prepend=False)
        exog_vars.append('const')
    
    # Fit IV model
    iv_model = IV2SLS(
        dependent=data_iv[dependent_var],
        exog=data_iv[exog_vars] if exog_vars else None,
        endog=data_iv[endog_vars],
        instruments=data_iv[instrument_vars]
    )
    iv_results = iv_model.fit(cov_type="homoskedastic", debiased=True)
    
    # Run exogeneity tests
    exogeneity_tests = run_all_exogeneity_tests(iv_results)
    
    return {
        'ols_results': ols_results,
        'iv_results': iv_results,
        'exogeneity_tests': exogeneity_tests
    }


if __name__ == "__main__":
    # Example usage with HousePrices dataset
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from linearmodels.iv.model import IV2SLS
    
    # Load HousePrices dataset
    try:
        houseprices = sm.datasets.get_rdataset(dataname="HousePrices", package="AER", cache=True).data
        print("Dataset loaded successfully.")
        print(houseprices.iloc[:, [0, 1, 2, 5, 10]].head())  # Display first few rows
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using simulated data instead.")
        # Create simulated data if dataset loading fails
        np.random.seed(42)
        n = 100
        
        # Create variables
        lotsize = np.random.normal(5000, 1000, n)
        bedrooms = np.random.randint(1, 5, n)
        driveway = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
        garage = np.random.randint(0, 3, n)
        
        # Create price with endogeneity (lotsize correlated with error)
        error = np.random.normal(0, 10000, n)
        lotsize = lotsize + 0.5 * error  # Introduce endogeneity
        price = 50000 + 10 * lotsize + 5000 * bedrooms + error
        
        # Create DataFrame
        houseprices = pd.DataFrame({
            'price': price,
            'lotsize': lotsize,
            'bedrooms': bedrooms,
            'driveway': driveway,
            'garage': garage
        })
    
    # Method 1: Using separate OLS and IV2SLS models
    print("\nMethod 1: Using separate OLS and IV2SLS models")
    
    # Fit OLS model
    mlr1 = smf.ols(formula="price ~ lotsize + bedrooms", data=houseprices).fit()
    print("\nOLS Model Summary:")
    print(mlr1.summary().tables[1])  # Print coefficient table
    
    # Prepare data for IV model
    mdatac = add_constant(houseprices, prepend=False)
    
    # Fit IV model
    try:
        mlr2 = IV2SLS(
            dependent=mdatac["price"],
            exog=mdatac[["const", "bedrooms"]],
            endog=mdatac["lotsize"],
            instruments=mdatac[["driveway", "garage"]]
        ).fit(cov_type="homoskedastic", debiased=True)
        
        print("\nIV Model Summary:")
        print(mlr2.summary.tables[1])  # Print coefficient table
        
        # Run exogeneity tests
        exogeneity_results = run_all_exogeneity_tests(mlr2)
        
        # Print test results
        for test_name, test_results in exogeneity_results.items():
            print(f"\n{test_name.upper()} TEST:")
            for key, value in test_results.items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error in IV estimation: {e}")
    
    # Method 2: Using the convenience function
    print("\n\nMethod 2: Using the convenience function")
    
    try:
        comparison_results = compare_ols_iv(
            data=houseprices,
            formula="price ~ lotsize + bedrooms",
            endog_vars=["lotsize"],
            instrument_vars=["driveway", "garage"]
        )
        
        print("\nOLS vs. IV Coefficient Comparison:")
        ols_params = comparison_results['ols_results'].params
        
        # Handle different parameter naming conventions between OLS and IV
        iv_params = pd.Series(index=ols_params.index)
        iv_model_params = comparison_results['iv_results'].params
        
        # Map IV parameter names to OLS parameter names
        param_mapping = {
            'const': 'Intercept',
            'lotsize': 'lotsize',
            'bedrooms': 'bedrooms'
        }
        
        for iv_name, ols_name in param_mapping.items():
            if iv_name in iv_model_params:
                iv_params[ols_name] = iv_model_params[iv_name]
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'OLS': ols_params,
            'IV': iv_params,
            'Difference': iv_params - ols_params
        })
        
        # Calculate percent difference where possible (avoiding division by zero)
        percent_diff = []
        for i in range(len(ols_params)):
            if ols_params[i] != 0:
                percent_diff.append(((iv_params[i] - ols_params[i]) / ols_params[i] * 100).round(2))
            else:
                percent_diff.append(np.nan)
        
        comparison_df['Percent Diff'] = percent_diff
        print(comparison_df)
        
        # Print exogeneity test results
        print("\nExogeneity Test Results:")
        for test_name, test_results in comparison_results['exogeneity_tests'].items():
            print(f"\n{test_name.upper()}:")
            for key, value in test_results.items():
                if key != 'conclusion':
                    print(f"  {key}: {value}")
            print(f"  CONCLUSION: {test_results.get('conclusion', 'N/A')}")
    
    except Exception as e:
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()
