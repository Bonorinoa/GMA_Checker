"""
Tests for linearity assumption functions.

This module contains tests for the linearity testing functions
in the gauss_markov_checker.assumptions.linearity module.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from gauss_markov_checker.assumptions.linearity import (
    reset_test,
    harvey_collier_test,
    rainbow_test,
    run_all_linearity_tests
)

@pytest.fixture
def linear_model():
    """Create a model with linear relationships."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate predictors
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Generate linear response with normal errors
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + np.random.normal(0, 1, n_samples)
    
    # Fit OLS model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

@pytest.fixture
def nonlinear_model():
    """Create a model with non-linear relationships."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate predictors
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Generate strongly non-linear response (quadratic and cubic terms)
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + 2.0 * X[:, 0]**2 + 1.0 * X[:, 0]**3 + np.random.normal(0, 0.5, n_samples)
    
    # Fit OLS model (incorrectly assuming linearity)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def test_reset_linear(linear_model):
    """Test RESET test with linear data."""
    result = reset_test(linear_model)
    
    # For linear data, expect high p-value (>0.05)
    assert result['p_value'] > 0.05, "RESET test incorrectly rejects linearity"
    assert isinstance(result['f_stat'], float), "F-statistic should be float"
    assert isinstance(result['p_value'], float), "p-value should be float"
    assert isinstance(result['df'], tuple), "df should be tuple"
    assert 'conclusion' in result, "Result should include conclusion"

def test_reset_nonlinear(nonlinear_model):
    """Test RESET test with non-linear data."""
    result = reset_test(nonlinear_model)
    
    # For non-linear data, expect low p-value (<0.05)
    assert result['p_value'] < 0.05, "RESET test fails to detect non-linearity"
    assert isinstance(result['f_stat'], float), "F-statistic should be float"

def test_harvey_collier_linear(linear_model):
    """Test Harvey-Collier test with linear data."""
    result = harvey_collier_test(linear_model)
    
    # For linear data, expect high p-value (>0.05)
    assert result['p_value'] > 0.05, "Harvey-Collier test incorrectly rejects linearity"
    assert isinstance(result['t_stat'], float), "t-statistic should be float"
    assert isinstance(result['p_value'], float), "p-value should be float"
    assert isinstance(result['df'], int), "df should be integer"
    assert 'conclusion' in result, "Result should include conclusion"

def test_harvey_collier_nonlinear(nonlinear_model):
    """Test Harvey-Collier test with non-linear data."""
    result = harvey_collier_test(nonlinear_model)
    
    # For non-linear data, expect low p-value (<0.05)
    assert result['p_value'] < 0.05, "Harvey-Collier test fails to detect non-linearity"
    assert isinstance(result['t_stat'], float), "t-statistic should be float"

def test_rainbow_linear(linear_model):
    """Test Rainbow test with linear data."""
    result = rainbow_test(linear_model)
    
    # For linear data, expect high p-value (>0.05)
    assert result['p_value'] > 0.05, "Rainbow test incorrectly rejects linearity"
    assert isinstance(result['f_stat'], float), "F-statistic should be float"
    assert isinstance(result['p_value'], float), "p-value should be float"
    assert 'conclusion' in result, "Result should include conclusion"

def test_rainbow_nonlinear(nonlinear_model):
    """Test Rainbow test with non-linear data."""
    result = rainbow_test(nonlinear_model)
    
    # For non-linear data, expect low p-value (<0.05)
    assert result['p_value'] < 0.05, "Rainbow test fails to detect non-linearity"
    assert isinstance(result['f_stat'], float), "F-statistic should be float"

def test_run_all_linearity_tests(linear_model):
    """Test running all linearity tests."""
    results = run_all_linearity_tests(linear_model)
    
    # Check that all tests are included
    assert 'reset' in results, "RESET test results missing"
    assert 'harvey_collier' in results, "Harvey-Collier test results missing"
    assert 'rainbow' in results, "Rainbow test results missing"
    
    # Check structure of each test result
    for test_name, result in results.items():
        assert isinstance(result, dict), f"{test_name} result should be dictionary"
        assert 'conclusion' in result, f"{test_name} result should include conclusion"
        assert isinstance(result['p_value'], float), f"{test_name} p-value should be float"

def test_real_data():
    """Test all functions with real statsmodels dataset."""
    # Load Boston housing dataset
    data = sm.datasets.get_rdataset("Boston", "MASS").data
    
    # Fit simple model
    X = sm.add_constant(data[['lstat', 'rm']])
    y = data['medv']
    model = sm.OLS(y, X).fit()
    
    # Test RESET
    reset_result = reset_test(model)
    assert isinstance(reset_result['f_stat'], float), "RESET F-statistic should be float"
    assert 0 <= reset_result['p_value'] <= 1, "RESET p-value should be between 0 and 1"
    
    # Test Harvey-Collier
    hc_result = harvey_collier_test(model)
    assert isinstance(hc_result['t_stat'], float), "HC t-statistic should be float"
    assert 0 <= hc_result['p_value'] <= 1, "HC p-value should be between 0 and 1"
    
    # Test Rainbow
    rainbow_result = rainbow_test(model)
    assert isinstance(rainbow_result['f_stat'], float), "Rainbow F-statistic should be float"
    assert 0 <= rainbow_result['p_value'] <= 1, "Rainbow p-value should be between 0 and 1"

def test_invalid_inputs():
    """Test functions with invalid inputs."""
    # Test with None
    with pytest.raises((TypeError, ValueError)):
        reset_test(None)
    with pytest.raises((TypeError, ValueError)):
        harvey_collier_test(None)
    with pytest.raises((TypeError, ValueError)):
        rainbow_test(None)
    
    # Test with non-model input
    with pytest.raises((TypeError, ValueError)):
        reset_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        harvey_collier_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        rainbow_test("not a model")

def test_parameter_validation(linear_model):
    """Test parameter validation in test functions."""
    # Test RESET test with invalid power
    with pytest.raises(ValueError):
        reset_test(linear_model, power=1)  # power should be > 1
    
    # Test RESET test with invalid test_type
    with pytest.raises(ValueError):
        reset_test(linear_model, test_type='invalid')
    
    # Test Rainbow test with invalid fraction
    with pytest.raises(ValueError):
        rainbow_test(linear_model, frac=1.5)  # frac should be between 0 and 1
    with pytest.raises(ValueError):
        rainbow_test(linear_model, frac=-0.5)  # frac should be between 0 and 1
