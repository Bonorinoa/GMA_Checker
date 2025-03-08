"""
Gauss-Markov Assumptions Checker Package

A comprehensive tool for testing OLS regression assumptions.
"""

# Version of the gauss_markov_checker package
__version__ = "0.1.0"

from gauss_markov_checker.assumptions import (
    run_RESET_test,
    run_all_autocorrelation_tests,
    run_all_exogeneity_tests,
    run_all_homoscedasticity_tests,
    run_all_normality_tests
)

__all__ = [
    'run_RESET_test',
    'run_all_autocorrelation_tests',
    'run_all_exogeneity_tests',
    'run_all_homoscedasticity_tests',
    'run_all_normality_tests'
]
