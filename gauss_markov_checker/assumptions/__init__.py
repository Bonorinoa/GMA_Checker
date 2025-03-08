"""
Gauss-Markov Assumptions Testing Modules

This package contains modules for testing each of the Gauss-Markov assumptions
required for OLS estimators to be BLUE (Best Linear Unbiased Estimators).
"""

from gauss_markov_checker.assumptions import (
    linearity,
    multicollinearity,
    exogeneity,
    homoscedasticity,
    autocorrelation,
    normality
)

__all__ = [
    'linearity',
    'multicollinearity',
    'exogeneity',
    'homoscedasticity',
    'autocorrelation',
    'normality'
]
