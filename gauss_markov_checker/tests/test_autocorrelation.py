"""
Tests for autocorrelation assumption functions.

This module contains tests for the autocorrelation testing functions
in the gauss_markov_checker.assumptions.autocorrelation module.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from gauss_markov_checker.assumptions.autocorrelation import (
    durbin_watson_test,
    breusch_godfrey_test,
    ljung_box_test
)

@pytest.fixture
def uncorrelated_model():
    """Create a model with no autocorrelation."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate time series predictor
    t = np.linspace(0, 10, n_samples)
    X = np.column_stack([np.sin(t), np.cos(t)])
    
    # Generate independent errors
    eps = np.random.normal(0, 1, n_samples)
    
    # Generate response
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + eps
    
    # Fit OLS model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

@pytest.fixture
def autocorrelated_model():
    """Create a model with autocorrelated errors."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate time series predictor
    t = np.linspace(0, 10, n_samples)
    X = np.column_stack([np.sin(t), np.cos(t)])
    
    # Generate AR(1) errors
    eps = np.zeros(n_samples)
    eps[0] = np.random.normal(0, 1)
    for i in range(1, n_samples):
        eps[i] = 0.8 * eps[i-1] + np.random.normal(0, 0.5)
    
    # Generate response
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + eps
    
    # Fit OLS model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def test_durbin_watson_uncorrelated(uncorrelated_model):
    """Test Durbin-Watson test with uncorrelated data."""
    dw_result = durbin_watson_test(uncorrelated_model)
    
    # For no autocorrelation, DW statistic should be close to 2
    assert isinstance(dw_result['statistic'], float), "DW statistic should be float"
    assert 1.5 < dw_result['statistic'] < 2.5, "DW statistic should be close to 2 for uncorrelated data"
    assert dw_result['statistic'] > 0, "DW statistic should be positive"
    assert dw_result['statistic'] < 4, "DW statistic should be less than 4"
    assert 'conclusion' in dw_result, "Result should include conclusion"

def test_durbin_watson_autocorrelated(autocorrelated_model):
    """Test Durbin-Watson test with autocorrelated data."""
    dw_result = durbin_watson_test(autocorrelated_model)
    
    # For positive autocorrelation, DW statistic should be < 1.5
    assert dw_result['statistic'] < 1.5, "DW test fails to detect positive autocorrelation"
    assert dw_result['statistic'] > 0, "DW statistic should be positive"
    assert 'conclusion' in dw_result, "Result should include conclusion"

def test_breusch_godfrey_uncorrelated(uncorrelated_model):
    """Test Breusch-Godfrey test with uncorrelated data."""
    result = breusch_godfrey_test(uncorrelated_model)
    
    # For uncorrelated data, expect high p-value (>0.05)
    assert result['p_value'] > 0.05, "BG test incorrectly rejects no autocorrelation"
    assert result['statistic'] >= 0, "Test statistic should be non-negative"
    assert isinstance(result['statistic'], float), "Test statistic should be float"
    assert isinstance(result['p_value'], float), "p-value should be float"
    assert 0 <= result['p_value'] <= 1, "p-value should be between 0 and 1"
    assert 'conclusion' in result, "Result should include conclusion"

def test_breusch_godfrey_autocorrelated(autocorrelated_model):
    """Test Breusch-Godfrey test with autocorrelated data."""
    result = breusch_godfrey_test(autocorrelated_model)
    
    # For autocorrelated data, expect low p-value (<0.05)
    assert result['p_value'] < 0.05, "BG test fails to detect autocorrelation"
    assert result['statistic'] >= 0, "Test statistic should be non-negative"
    assert 'conclusion' in result, "Result should include conclusion"

def test_ljung_box_uncorrelated(uncorrelated_model):
    """Test Ljung-Box test with uncorrelated data."""
    result = ljung_box_test(uncorrelated_model)
    
    # For uncorrelated data, expect high p-value (>0.05)
    assert result['p_value'] > 0.05, "LB test incorrectly rejects no autocorrelation"
    assert result['statistic'] >= 0, "Test statistic should be non-negative"
    assert isinstance(result['statistic'], float), "Test statistic should be float"
    assert isinstance(result['p_value'], float), "p-value should be float"
    assert 0 <= result['p_value'] <= 1, "p-value should be between 0 and 1"
    assert 'conclusion' in result, "Result should include conclusion"

def test_ljung_box_autocorrelated(autocorrelated_model):
    """Test Ljung-Box test with autocorrelated data."""
    result = ljung_box_test(autocorrelated_model)
    
    # For autocorrelated data, expect low p-value (<0.05)
    assert result['p_value'] < 0.05, "LB test fails to detect autocorrelation"
    assert result['statistic'] >= 0, "Test statistic should be non-negative"
    assert 'conclusion' in result, "Result should include conclusion"

def test_breusch_godfrey_lags(autocorrelated_model):
    """Test Breusch-Godfrey test with different lag orders."""
    # Test with different lag orders
    for nlags in [1, 4, 12]:
        result = breusch_godfrey_test(autocorrelated_model, nlags=nlags)
        assert isinstance(result['statistic'], float), "Test statistic should be float"
        assert 0 <= result['p_value'] <= 1, "p-value should be between 0 and 1"
        assert 'conclusion' in result, "Result should include conclusion"

def test_ljung_box_lags(autocorrelated_model):
    """Test Ljung-Box test with different lag orders."""
    # Test with different lag orders
    for nlags in [1, 4, 12]:
        result = ljung_box_test(autocorrelated_model, nlags=nlags)
        assert isinstance(result['statistic'], float), "Test statistic should be float"
        assert 0 <= result['p_value'] <= 1, "p-value should be between 0 and 1"
        assert 'conclusion' in result, "Result should include conclusion"

def test_real_data():
    """Test all functions with real statsmodels dataset."""
    # Load Longley dataset (macroeconomic time series)
    data = sm.datasets.longley.load_pandas().data
    
    # Fit simple model
    X = sm.add_constant(data[['GNP', 'UNEMP']])
    y = data['TOTEMP']
    model = sm.OLS(y, X).fit()
    
    # Test Durbin-Watson
    dw_result = durbin_watson_test(model)
    assert isinstance(dw_result['statistic'], float), "DW statistic should be float"
    assert 0 < dw_result['statistic'] < 4, "DW statistic should be between 0 and 4"
    assert 'conclusion' in dw_result, "Result should include conclusion"
    
    # Test Breusch-Godfrey
    bg_result = breusch_godfrey_test(model)
    assert isinstance(bg_result['statistic'], float), "BG statistic should be float"
    assert 0 <= bg_result['p_value'] <= 1, "BG p-value should be between 0 and 1"
    assert 'conclusion' in bg_result, "Result should include conclusion"
    
    # Test Ljung-Box
    lb_result = ljung_box_test(model)
    assert isinstance(lb_result['statistic'], float), "LB statistic should be float"
    assert 0 <= lb_result['p_value'] <= 1, "LB p-value should be between 0 and 1"
    assert 'conclusion' in lb_result, "Result should include conclusion"

def test_invalid_inputs():
    """Test functions with invalid inputs."""
    # Test with None
    with pytest.raises((TypeError, ValueError)):
        durbin_watson_test(None)
    with pytest.raises((TypeError, ValueError)):
        breusch_godfrey_test(None)
    with pytest.raises((TypeError, ValueError)):
        ljung_box_test(None)
    
    # Test with non-model input
    with pytest.raises((TypeError, ValueError)):
        durbin_watson_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        breusch_godfrey_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        ljung_box_test("not a model")
