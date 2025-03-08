"""
Tests for homoscedasticity assumption functions.

This module contains tests for the homoscedasticity testing functions
in the gauss_markov_checker.assumptions.homoscedasticity module.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from gauss_markov_checker.assumptions.homoscedasticity import (
    breusch_pagan_test,
    white_test,
    goldfeld_quandt_test
)

@pytest.fixture
def homoscedastic_model():
    """Create a model with homoscedastic errors."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate predictors
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Generate homoscedastic errors
    eps = np.random.normal(0, 1, n_samples)
    
    # Generate response
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + eps
    
    # Fit OLS model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

@pytest.fixture
def heteroscedastic_model():
    """Create a model with heteroscedastic errors."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate predictors
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Generate heteroscedastic errors (variance increases with X[:, 0])
    eps = np.random.normal(0, np.exp(X[:, 0]), n_samples)
    
    # Generate response
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + eps
    
    # Fit OLS model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def test_breusch_pagan_homoscedastic():
    """Test Breusch-Pagan test with homoscedastic data."""
    model = homoscedastic_model()
    stat, p_value = breusch_pagan_test(model)
    
    # For homoscedastic data, expect high p-value (>0.05)
    assert p_value > 0.05, "BP test incorrectly rejects homoscedasticity"
    assert stat >= 0, "Test statistic should be non-negative"
    assert isinstance(stat, float), "Test statistic should be float"
    assert isinstance(p_value, float), "p-value should be float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"

def test_breusch_pagan_heteroscedastic():
    """Test Breusch-Pagan test with heteroscedastic data."""
    model = heteroscedastic_model()
    stat, p_value = breusch_pagan_test(model)
    
    # For heteroscedastic data, expect low p-value (<0.05)
    assert p_value < 0.05, "BP test fails to detect heteroscedasticity"
    assert stat >= 0, "Test statistic should be non-negative"

def test_white_homoscedastic():
    """Test White test with homoscedastic data."""
    model = homoscedastic_model()
    stat, p_value = white_test(model)
    
    # For homoscedastic data, expect high p-value (>0.05)
    assert p_value > 0.05, "White test incorrectly rejects homoscedasticity"
    assert stat >= 0, "Test statistic should be non-negative"
    assert isinstance(stat, float), "Test statistic should be float"
    assert isinstance(p_value, float), "p-value should be float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"

def test_white_heteroscedastic():
    """Test White test with heteroscedastic data."""
    model = heteroscedastic_model()
    stat, p_value = white_test(model)
    
    # For heteroscedastic data, expect low p-value (<0.05)
    assert p_value < 0.05, "White test fails to detect heteroscedasticity"
    assert stat >= 0, "Test statistic should be non-negative"

def test_goldfeld_quandt_homoscedastic():
    """Test Goldfeld-Quandt test with homoscedastic data."""
    model = homoscedastic_model()
    stat, p_value = goldfeld_quandt_test(model)
    
    # For homoscedastic data, expect high p-value (>0.05)
    assert p_value > 0.05, "GQ test incorrectly rejects homoscedasticity"
    assert stat >= 0, "Test statistic should be non-negative"
    assert isinstance(stat, float), "Test statistic should be float"
    assert isinstance(p_value, float), "p-value should be float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"

def test_goldfeld_quandt_heteroscedastic():
    """Test Goldfeld-Quandt test with heteroscedastic data."""
    model = heteroscedastic_model()
    stat, p_value = goldfeld_quandt_test(model)
    
    # For heteroscedastic data, expect low p-value (<0.05)
    assert p_value < 0.05, "GQ test fails to detect heteroscedasticity"
    assert stat >= 0, "Test statistic should be non-negative"

def test_real_data():
    """Test all functions with real statsmodels dataset."""
    # Load Boston housing dataset
    data = sm.datasets.get_rdataset("Boston", "MASS").data
    
    # Fit simple model
    X = sm.add_constant(data[['lstat', 'rm']])
    y = data['medv']
    model = sm.OLS(y, X).fit()
    
    # Test Breusch-Pagan
    bp_stat, bp_p = breusch_pagan_test(model)
    assert isinstance(bp_stat, float), "BP statistic should be float"
    assert 0 <= bp_p <= 1, "BP p-value should be between 0 and 1"
    
    # Test White
    w_stat, w_p = white_test(model)
    assert isinstance(w_stat, float), "White statistic should be float"
    assert 0 <= w_p <= 1, "White p-value should be between 0 and 1"
    
    # Test Goldfeld-Quandt
    gq_stat, gq_p = goldfeld_quandt_test(model)
    assert isinstance(gq_stat, float), "GQ statistic should be float"
    assert 0 <= gq_p <= 1, "GQ p-value should be between 0 and 1"

def test_invalid_inputs():
    """Test functions with invalid inputs."""
    # Test with None
    with pytest.raises((TypeError, ValueError)):
        breusch_pagan_test(None)
    with pytest.raises((TypeError, ValueError)):
        white_test(None)
    with pytest.raises((TypeError, ValueError)):
        goldfeld_quandt_test(None)
    
    # Test with non-model input
    with pytest.raises((TypeError, ValueError)):
        breusch_pagan_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        white_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        goldfeld_quandt_test("not a model")
