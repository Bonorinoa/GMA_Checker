"""
Tests for multicollinearity assumption functions.

This module contains tests for the multicollinearity testing functions
in the gauss_markov_checker.assumptions.multicollinearity module.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from gauss_markov_checker.assumptions.multicollinearity import (
    variance_inflation_factor,
    condition_number,
    correlation_analysis
)

@pytest.fixture
def uncorrelated_data():
    """Create a dataset with uncorrelated predictors."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate independent random variables
    X = pd.DataFrame({
        'x1': np.random.normal(0, 1, n_samples),
        'x2': np.random.normal(0, 1, n_samples),
        'x3': np.random.normal(0, 1, n_samples)
    })
    return X

@pytest.fixture
def correlated_data():
    """Create a dataset with highly correlated predictors."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate base variables
    x1 = np.random.normal(0, 1, n_samples)
    
    # Create correlated variables
    x2 = 0.9 * x1 + 0.1 * np.random.normal(0, 1, n_samples)  # Strong correlation with x1
    x3 = 0.8 * x1 - 0.1 * x2 + 0.1 * np.random.normal(0, 1, n_samples)  # Multicollinear
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3
    })
    return X

def test_vif_uncorrelated():
    """Test VIF calculation with uncorrelated predictors."""
    X = uncorrelated_data()
    vifs = variance_inflation_factor(X)
    
    # For uncorrelated predictors, VIFs should be close to 1
    assert all(vif < 2 for vif in vifs[1:]), "VIFs should be low for uncorrelated predictors"
    assert isinstance(vifs, pd.Series), "Result should be a pandas Series"
    assert all(isinstance(vif, float) for vif in vifs), "VIFs should be floats"
    assert len(vifs) == len(X.columns) + 1, "Should have VIF for each predictor plus constant"

def test_vif_correlated():
    """Test VIF calculation with correlated predictors."""
    X = correlated_data()
    vifs = variance_inflation_factor(X)
    
    # For correlated predictors, at least some VIFs should be high
    assert any(vif > 5 for vif in vifs[1:]), "VIFs should detect multicollinearity"
    assert isinstance(vifs, pd.Series), "Result should be a pandas Series"
    assert all(vif > 1 for vif in vifs[1:]), "VIFs should be greater than 1"

def test_condition_number_uncorrelated():
    """Test condition number calculation with uncorrelated predictors."""
    X = uncorrelated_data()
    cond_num = condition_number(X)
    
    # For uncorrelated predictors, condition number should be low
    assert cond_num < 10, "Condition number should be low for uncorrelated predictors"
    assert isinstance(cond_num, float), "Condition number should be float"
    assert cond_num > 1, "Condition number should be greater than 1"

def test_condition_number_correlated():
    """Test condition number calculation with correlated predictors."""
    X = correlated_data()
    cond_num = condition_number(X)
    
    # For correlated predictors, condition number should be high
    assert cond_num > 10, "Condition number should detect multicollinearity"
    assert isinstance(cond_num, float), "Condition number should be float"

def test_correlation_analysis_uncorrelated():
    """Test correlation analysis with uncorrelated predictors."""
    X = uncorrelated_data()
    corr_matrix = correlation_analysis(X)
    
    # Check correlation matrix properties
    assert isinstance(corr_matrix, pd.DataFrame), "Result should be a DataFrame"
    assert corr_matrix.shape == (3, 3), "Should be square matrix"
    assert np.allclose(corr_matrix.values, corr_matrix.values.T), "Should be symmetric"
    assert np.allclose(np.diag(corr_matrix), 1), "Diagonal should be 1"
    
    # For uncorrelated predictors, off-diagonal elements should be small
    off_diag = corr_matrix.values[~np.eye(3, dtype=bool)]
    assert np.all(np.abs(off_diag) < 0.3), "Correlations should be weak"

def test_correlation_analysis_correlated():
    """Test correlation analysis with correlated predictors."""
    X = correlated_data()
    corr_matrix = correlation_analysis(X)
    
    # Check correlation matrix properties
    assert isinstance(corr_matrix, pd.DataFrame), "Result should be a DataFrame"
    assert corr_matrix.shape == (3, 3), "Should be square matrix"
    assert np.allclose(corr_matrix.values, corr_matrix.values.T), "Should be symmetric"
    assert np.allclose(np.diag(corr_matrix), 1), "Diagonal should be 1"
    
    # For correlated predictors, some off-diagonal elements should be large
    off_diag = corr_matrix.values[~np.eye(3, dtype=bool)]
    assert np.any(np.abs(off_diag) > 0.8), "Should detect strong correlations"

def test_real_data():
    """Test all functions with real statsmodels dataset."""
    # Load Boston housing dataset
    data = sm.datasets.get_rdataset("Boston", "MASS").data
    X = data[['lstat', 'rm', 'age', 'tax']]
    
    # Test VIF
    vifs = variance_inflation_factor(X)
    assert isinstance(vifs, pd.Series), "VIF result should be pandas Series"
    assert len(vifs) == len(X.columns) + 1, "Should have VIF for each predictor plus constant"
    
    # Test condition number
    cond_num = condition_number(X)
    assert isinstance(cond_num, float), "Condition number should be float"
    assert cond_num > 0, "Condition number should be positive"
    
    # Test correlation matrix
    corr_matrix = correlation_analysis(X)
    assert isinstance(corr_matrix, pd.DataFrame), "Correlation result should be DataFrame"
    assert corr_matrix.shape == (4, 4), "Should be square matrix"
    assert np.allclose(np.diag(corr_matrix), 1), "Diagonal should be 1"

def test_invalid_inputs():
    """Test functions with invalid inputs."""
    # Test with None
    with pytest.raises((TypeError, ValueError)):
        variance_inflation_factor(None)
    with pytest.raises((TypeError, ValueError)):
        condition_number(None)
    with pytest.raises((TypeError, ValueError)):
        correlation_analysis(None)
    
    # Test with invalid DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        variance_inflation_factor(empty_df)
    with pytest.raises(ValueError):
        condition_number(empty_df)
    with pytest.raises(ValueError):
        correlation_analysis(empty_df)
    
    # Test with single column
    single_col = pd.DataFrame({'x': [1, 2, 3]})
    with pytest.raises(ValueError):
        variance_inflation_factor(single_col)

def test_edge_cases():
    """Test edge cases and numerical stability."""
    np.random.seed(42)
    n_samples = 100
    
    # Test with perfectly collinear variables
    x = np.random.normal(0, 1, n_samples)
    X = pd.DataFrame({
        'x1': x,
        'x2': 2 * x,  # Perfect collinearity
        'x3': np.random.normal(0, 1, n_samples)
    })
    
    # VIF should be very high for collinear variables
    vifs = variance_inflation_factor(X)
    assert any(vif > 100 for vif in vifs), "Should detect perfect collinearity"
    
    # Condition number should be very high
    cond_num = condition_number(X)
    assert cond_num > 100, "Should detect perfect collinearity"
    
    # Correlation should be exactly 1 between collinear variables
    corr_matrix = correlation_analysis(X)
    assert np.isclose(corr_matrix.loc['x1', 'x2'], 1), "Should detect perfect correlation"
