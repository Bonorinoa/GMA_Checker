"""
Tests for exogeneity assumption functions.

This module contains tests for the exogeneity testing functions
in the gauss_markov_checker.assumptions.exogeneity module.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv.model import IV2SLS
from gauss_markov_checker.assumptions.exogeneity import (
    wu_hausman_test,
    wooldridge_regression_test,
    sargan_test,
    run_all_exogeneity_tests
)

@pytest.fixture
def iv_model_results():
    """Create a model with exogenous predictors and valid instruments."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate exogenous predictor
    x1 = np.random.normal(0, 1, n_samples)
    
    # Generate instruments
    z1 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)
    z2 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)
    
    # Generate endogenous predictor
    x2 = 0.5 * z1 + 0.3 * z2 + np.random.normal(0, 0.5, n_samples)
    
    # Generate response
    eps = np.random.normal(0, 1, n_samples)
    y = 1 + 0.5 * x1 + 0.8 * x2 + eps
    
    # Create DataFrame
    data = pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'z1': z1,
        'z2': z2
    })
    
    # Fit IV model
    exog = sm.add_constant(data[['x1']])
    endog = data[['x2']]
    instruments = data[['z1', 'z2']]
    
    model = IV2SLS(data['y'], exog, endog, instruments)
    return model.fit()

def test_wu_hausman():
    """Test Wu-Hausman test."""
    results = iv_model_results()
    test_result = wu_hausman_test(results)
    
    assert isinstance(test_result, dict), "Result should be dictionary"
    assert 'statistic' in test_result, "Should include test statistic"
    assert 'p_value' in test_result, "Should include p-value"
    assert 'df' in test_result, "Should include degrees of freedom"
    assert 'conclusion' in test_result, "Should include conclusion"
    assert test_result['p_value'] >= 0 and test_result['p_value'] <= 1, "P-value should be between 0 and 1"

def test_wooldridge_regression():
    """Test Wooldridge regression test."""
    results = iv_model_results()
    test_result = wooldridge_regression_test(results)
    
    assert isinstance(test_result, dict), "Result should be dictionary"
    assert 'statistic' in test_result, "Should include test statistic"
    assert 'p_value' in test_result, "Should include p-value"
    assert 'df' in test_result, "Should include degrees of freedom"
    assert 'conclusion' in test_result, "Should include conclusion"
    assert test_result['p_value'] >= 0 and test_result['p_value'] <= 1, "P-value should be between 0 and 1"

def test_sargan():
    """Test Sargan test."""
    results = iv_model_results()
    test_result = sargan_test(results)
    
    assert isinstance(test_result, dict), "Result should be dictionary"
    assert 'statistic' in test_result, "Should include test statistic"
    assert 'p_value' in test_result, "Should include p-value"
    assert 'df' in test_result, "Should include degrees of freedom"
    assert 'conclusion' in test_result, "Should include conclusion"
    assert test_result['p_value'] >= 0 and test_result['p_value'] <= 1, "P-value should be between 0 and 1"

def test_run_all_exogeneity_tests():
    """Test running all exogeneity tests."""
    results = iv_model_results()
    all_tests = run_all_exogeneity_tests(results)
    
    assert isinstance(all_tests, dict), "Result should be dictionary"
    assert 'wu_hausman' in all_tests, "Should include Wu-Hausman test"
    assert 'wooldridge_regression' in all_tests, "Should include Wooldridge regression test"
    assert 'sargan' in all_tests, "Should include Sargan test"
    
    for test_name, result in all_tests.items():
        if 'error' not in result:  # Skip if test wasn't applicable
            assert 'statistic' in result, f"{test_name} should include test statistic"
            assert 'p_value' in result, f"{test_name} should include p-value"
            assert 'conclusion' in result, f"{test_name} should include conclusion"

def test_invalid_inputs():
    """Test functions with invalid inputs."""
    with pytest.raises((TypeError, ValueError)):
        wu_hausman_test(None)
    with pytest.raises((TypeError, ValueError)):
        wooldridge_regression_test(None)
    with pytest.raises((TypeError, ValueError)):
        sargan_test(None)
    with pytest.raises((TypeError, ValueError)):
        run_all_exogeneity_tests(None)

def test_real_data():
    """Test with real data from statsmodels."""
    # Load data
    data = sm.datasets.statecrime.load_pandas().data
    
    # Prepare variables
    y = data['violent']
    X = sm.add_constant(data[['poverty']])
    endog = data[['hs_grad']]
    instruments = data[['urban']]
    
    # Fit IV model
    model = IV2SLS(y, X, endog, instruments)
    results = model.fit()
    
    # Run all tests
    all_tests = run_all_exogeneity_tests(results)
    
    # Check results
    assert isinstance(all_tests, dict), "Result should be dictionary"
    for test_name, result in all_tests.items():
        if 'error' not in result:  # Skip if test wasn't applicable
            assert isinstance(result, dict), f"{test_name} result should be dictionary"
            assert 'conclusion' in result, f"{test_name} should include conclusion"
