"""
Tests for normality assumption functions.

This module contains tests for the normality testing functions
in the gauss_markov_checker.assumptions.normality module.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from gauss_markov_checker.assumptions.normality import (
    jarque_bera_test,
    shapiro_wilk_test,
    anderson_darling_test,
    dagostino_pearson_test
)

@pytest.fixture
def normal_model():
    """Create a model with normally distributed errors."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate predictors
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Generate normal errors
    eps = np.random.normal(0, 1, n_samples)
    
    # Generate response
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + eps
    
    # Fit OLS model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

@pytest.fixture
def nonnormal_model():
    """Create a model with non-normal errors (chi-squared distribution)."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate predictors
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Generate chi-squared errors (right-skewed)
    eps = np.random.chisquare(df=3, size=n_samples)
    
    # Generate response
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + eps
    
    # Fit OLS model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def test_jarque_bera_normal():
    """Test Jarque-Bera test with normal data."""
    model = normal_model()
    stat, p_value = jarque_bera_test(model)
    
    # For normal data, expect high p-value (>0.05)
    assert p_value > 0.05, "JB test incorrectly rejects normality"
    assert stat >= 0, "Test statistic should be non-negative"
    assert isinstance(stat, float), "Test statistic should be float"
    assert isinstance(p_value, float), "p-value should be float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"

def test_jarque_bera_nonnormal():
    """Test Jarque-Bera test with non-normal data."""
    model = nonnormal_model()
    stat, p_value = jarque_bera_test(model)
    
    # For non-normal data, expect low p-value (<0.05)
    assert p_value < 0.05, "JB test fails to detect non-normality"
    assert stat >= 0, "Test statistic should be non-negative"

def test_shapiro_wilk_normal():
    """Test Shapiro-Wilk test with normal data."""
    model = normal_model()
    stat, p_value = shapiro_wilk_test(model)
    
    # For normal data, expect high p-value (>0.05)
    assert p_value > 0.05, "SW test incorrectly rejects normality"
    assert 0 <= stat <= 1, "Test statistic should be between 0 and 1"
    assert isinstance(stat, float), "Test statistic should be float"
    assert isinstance(p_value, float), "p-value should be float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"

def test_shapiro_wilk_nonnormal():
    """Test Shapiro-Wilk test with non-normal data."""
    model = nonnormal_model()
    stat, p_value = shapiro_wilk_test(model)
    
    # For non-normal data, expect low p-value (<0.05)
    assert p_value < 0.05, "SW test fails to detect non-normality"
    assert 0 <= stat <= 1, "Test statistic should be between 0 and 1"

def test_anderson_darling_normal():
    """Test Anderson-Darling test with normal data."""
    model = normal_model()
    stat, critical_values, significance_levels = anderson_darling_test(model)
    
    # For normal data, statistic should be less than critical values
    assert isinstance(stat, float), "Test statistic should be float"
    assert stat >= 0, "Test statistic should be non-negative"
    assert len(critical_values) == len(significance_levels), "Critical values and significance levels should match"
    assert all(cv > 0 for cv in critical_values), "Critical values should be positive"

def test_anderson_darling_nonnormal():
    """Test Anderson-Darling test with non-normal data."""
    model = nonnormal_model()
    stat, critical_values, significance_levels = anderson_darling_test(model)
    
    # For non-normal data, statistic should be larger than critical values
    assert stat > critical_values[2], "AD test fails to detect non-normality at 5% level"
    assert stat >= 0, "Test statistic should be non-negative"

def test_dagostino_pearson_normal():
    """Test D'Agostino-Pearson test with normal data."""
    model = normal_model()
    stat, p_value = dagostino_pearson_test(model)
    
    # For normal data, expect high p-value (>0.05)
    assert p_value > 0.05, "DP test incorrectly rejects normality"
    assert stat >= 0, "Test statistic should be non-negative"
    assert isinstance(stat, float), "Test statistic should be float"
    assert isinstance(p_value, float), "p-value should be float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"

def test_dagostino_pearson_nonnormal():
    """Test D'Agostino-Pearson test with non-normal data."""
    model = nonnormal_model()
    stat, p_value = dagostino_pearson_test(model)
    
    # For non-normal data, expect low p-value (<0.05)
    assert p_value < 0.05, "DP test fails to detect non-normality"
    assert stat >= 0, "Test statistic should be non-negative"

def test_real_data():
    """Test all functions with real statsmodels dataset."""
    # Load Boston housing dataset
    data = sm.datasets.get_rdataset("Boston", "MASS").data
    
    # Fit simple model
    X = sm.add_constant(data[['lstat', 'rm']])
    y = data['medv']
    model = sm.OLS(y, X).fit()
    
    # Test Jarque-Bera
    jb_stat, jb_p = jarque_bera_test(model)
    assert isinstance(jb_stat, float), "JB statistic should be float"
    assert 0 <= jb_p <= 1, "JB p-value should be between 0 and 1"
    
    # Test Shapiro-Wilk
    sw_stat, sw_p = shapiro_wilk_test(model)
    assert isinstance(sw_stat, float), "SW statistic should be float"
    assert 0 <= sw_p <= 1, "SW p-value should be between 0 and 1"
    
    # Test Anderson-Darling
    ad_stat, ad_crit, ad_sig = anderson_darling_test(model)
    assert isinstance(ad_stat, float), "AD statistic should be float"
    assert len(ad_crit) == len(ad_sig), "AD critical values and significance levels should match"
    
    # Test D'Agostino-Pearson
    dp_stat, dp_p = dagostino_pearson_test(model)
    assert isinstance(dp_stat, float), "DP statistic should be float"
    assert 0 <= dp_p <= 1, "DP p-value should be between 0 and 1"

def test_invalid_inputs():
    """Test functions with invalid inputs."""
    # Test with None
    with pytest.raises((TypeError, ValueError)):
        jarque_bera_test(None)
    with pytest.raises((TypeError, ValueError)):
        shapiro_wilk_test(None)
    with pytest.raises((TypeError, ValueError)):
        anderson_darling_test(None)
    with pytest.raises((TypeError, ValueError)):
        dagostino_pearson_test(None)
    
    # Test with non-model input
    with pytest.raises((TypeError, ValueError)):
        jarque_bera_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        shapiro_wilk_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        anderson_darling_test("not a model")
    with pytest.raises((TypeError, ValueError)):
        dagostino_pearson_test("not a model")

def test_small_sample():
    """Test behavior with small sample size."""
    # Generate small sample
    np.random.seed(42)
    n_samples = 10
    X = np.random.normal(0, 1, (n_samples, 2))
    y = 2 + 0.5 * X[:, 0] - 1.0 * X[:, 1] + np.random.normal(0, 1, n_samples)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # All tests should still run without error for small samples
    jarque_bera_test(model)
    shapiro_wilk_test(model)
    anderson_darling_test(model)
    dagostino_pearson_test(model)
