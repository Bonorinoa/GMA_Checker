# Gauss-Markov Assumptions Checker (GMA_Checker)

A comprehensive Python tool for testing whether OLS regression models satisfy the Gauss-Markov assumptions necessary for the estimator to be BLUE (Best Linear Unbiased Estimator).

## Overview

This tool provides econometricians and PhD economics students with both formal hypothesis tests and visual diagnostics to assess the validity of OLS regression models. By combining statistical rigor with intuitive visualizations, GMA_Checker helps users identify potential violations of the Gauss-Markov assumptions and make informed decisions about model improvements.

## Features

- **Comprehensive Testing Suite**: Formal hypothesis tests for all Gauss-Markov assumptions
- **Interactive Visualizations**: Diagnostic plots to visually assess assumption violations
- **User-Friendly Interface**: Streamlit app for easy data upload and model specification
- **Educational Content**: Detailed explanations and interpretation guidelines
- **Modular Design**: Clean separation between statistical tests and visualization code

## Directory Structure

```
gauss_markov_checker/
├── __init__.py
├── assumptions/              # Statistical tests for each assumption
│   ├── __init__.py
│   ├── linearity.py          # RESET test
│   ├── multicollinearity.py  # VIF, condition number, correlations
│   ├── homoscedasticity.py   # Breusch-Pagan, White, Goldfeld-Quandt tests
│   ├── autocorrelation.py    # Durbin-Watson, Breusch-Godfrey, Ljung-Box tests
│   ├── normality.py          # Jarque-Bera, Shapiro-Wilk, D'Agostino-Pearson tests
│   ├── exogeneity.py         # Exogeneity tests (for IV/2SLS models)
│   └── interpretation_rules.json  # Test interpretations knowledge base
├── visualization/            # Visualization functions
│   ├── __init__.py
│   ├── diagnostic_plots.py   # Diagnostic plots for each assumption
│   └── report_generator.py   # Report generation functionality (pending)
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── model_wrappers.py     # Model wrapper classes (pending)
│   └── statistics.py         # Statistical utility functions (pending)
├── app/                      # Streamlit application
│   ├── __init__.py
│   └── streamlit_app.py      # Main Streamlit app
└── tests/                    # Unit tests (pending)
    ├── __init__.py
    ├── test_linearity.py
    ├── test_multicollinearity.py
    ├── test_homoscedasticity.py
    ├── test_autocorrelation.py
    ├── test_normality.py
    ├── test_exogeneity.py
    └── test_app.py
```

## Implemented Tests by Assumption

### Linearity
- **RESET Test (Ramsey's Regression Equation Specification Error Test)**
  - Tests whether non-linear combinations of fitted values help explain the response variable
  - Null hypothesis: The model is correctly specified (linear)

### Multicollinearity
- **Variance Inflation Factor (VIF)**
  - Measures how much the variance of a regression coefficient is inflated due to multicollinearity
  - Values > 10 indicate problematic multicollinearity
- **Condition Number**
  - Measures the sensitivity of a regression to small changes in the independent variables
  - Values > 30 indicate moderate to severe multicollinearity
- **Correlation Matrix**
  - Pairwise correlations between predictor variables
  - High absolute correlations (> 0.8) suggest multicollinearity

### Homoscedasticity
- **Breusch-Pagan Test**
  - Tests whether the variance of the errors depends on the values of the independent variables
  - Null hypothesis: Homoscedasticity (constant variance)
- **White Test**
  - A special case of the Breusch-Pagan test that includes cross-products of the predictors
  - Null hypothesis: Homoscedasticity (constant variance)
- **Goldfeld-Quandt Test**
  - Compares the variance of residuals in two subsets of the data
  - Null hypothesis: Homoscedasticity (constant variance)

### Autocorrelation
- **Durbin-Watson Test**
  - Tests for first-order autocorrelation in the residuals
  - Values close to 2 indicate no autocorrelation
- **Breusch-Godfrey Test**
  - Tests for higher-order autocorrelation in the residuals
  - Null hypothesis: No autocorrelation up to order p
- **Ljung-Box Test**
  - Tests whether autocorrelations of residuals are different from zero
  - Null hypothesis: No autocorrelation up to lag k

### Normality
- **Jarque-Bera Test**
  - Tests whether the residuals have skewness and kurtosis matching a normal distribution
  - Null hypothesis: Residuals are normally distributed
- **Shapiro-Wilk Test**
  - Tests whether a sample came from a normally distributed population
  - Null hypothesis: Residuals are normally distributed
- **D'Agostino-Pearson Test**
  - Combines skewness and kurtosis tests into an omnibus test
  - Null hypothesis: Residuals are normally distributed

## Visualization Suite

### Comprehensive Diagnostics
- **Regression Diagnostics Plot**: Four-panel plot showing residuals vs. fitted, Q-Q plot, scale-location, and residuals vs. leverage

### Normality Diagnostics
- **Residual Histogram**: Histogram of residuals with normal curve overlay
- **Q-Q Plot**: Quantile-Quantile plot of residuals against theoretical normal distribution

### Homoscedasticity Diagnostics
- **Residuals vs. Fitted Values**: Scatter plot of residuals against fitted values
- **Scale-Location Plot**: Square root of standardized residuals against fitted values

### Autocorrelation Diagnostics
- **ACF Plot**: Autocorrelation function of residuals
- **PACF Plot**: Partial autocorrelation function of residuals

### Multicollinearity Diagnostics
- **Correlation Heatmap**: Heatmap of correlations between predictor variables
- **Component-Plus-Residual Plots**: Partial residual plots for each predictor

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GMA_Checker.git
cd GMA_Checker

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run gauss_markov_checker/app/streamlit_app.py
```

### Using the API

```python
import pandas as pd
import statsmodels.formula.api as smf
from gauss_markov_checker.assumptions import (
    run_RESET_test,
    run_all_autocorrelation_tests,
    run_all_homoscedasticity_tests,
    run_all_normality_tests
)

# Load your data
data = pd.read_csv("your_data.csv")

# Fit a model
model = smf.ols(formula="y ~ x1 + x2", data=data).fit()

# Run tests
linearity_results = run_RESET_test(model)
autocorrelation_results = run_all_autocorrelation_tests(model)
homoscedasticity_results = run_all_homoscedasticity_tests(model)
normality_results = run_all_normality_tests(model)

# Print results
print(linearity_results)
```

## Requirements

- Python 3.8+
- statsmodels
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- patsy
- scipy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The statsmodels team for providing the foundation for many of the statistical tests
- The Streamlit team for their excellent framework for building data apps
