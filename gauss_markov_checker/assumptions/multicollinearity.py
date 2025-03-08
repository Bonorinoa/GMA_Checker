"""
Multicollinearity tests for OLS regression models.

This module provides functions to test for multicollinearity among independent variables
in OLS regression models. Multicollinearity occurs when independent variables are highly
correlated with each other, which can lead to unstable and unreliable coefficient estimates.

The module implements three main approaches to detecting multicollinearity:
1. Variance Inflation Factors (VIF)
2. Condition Numbers and Condition Indices
3. Correlation Analysis
"""

import pandas as pd
import numpy as np
import statsmodels.stats.outliers_influence as smoi
import statsmodels.api as sm
from typing import Tuple, Union


def variance_inflation_factor(X: pd.DataFrame) -> pd.Series:
    """
    Calculates the Variance Inflation Factor (VIF) for each independent variable.

    The VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity.

    Args:
        X: A Pandas DataFrame containing the independent variables.  Should *not*
           include an intercept term; statsmodels adds it automatically.

    Returns:
        A Pandas Series containing the VIF for each variable.

    Interpretation:
        - VIF = 1: No correlation between a predictor and other variables.
        - 1 < VIF < 5: Moderate correlation, generally acceptable.
        - 5 <= VIF <= 10: High correlation, potentially problematic.
        - VIF > 10: Severe multicollinearity, requires corrective action.

    Example:
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> data = sm.datasets.get_rdataset("HousePrices", "AER", cache=True).data
        >>> X = data[["lotsize", "bedrooms", "bathrms"]]  # No intercept needed
        >>> vifs = variance_inflation_factor(X)
        >>> print(vifs)
    """
    # statsmodels VIF function requires an intercept.  Add it.
    X = sm.add_constant(X)
    vif_data = pd.Series(
        [smoi.variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
    )
    return vif_data


def condition_number(X: pd.DataFrame) -> float:
    """
    Calculates the condition number of the design matrix.

    The condition number is a global measure of multicollinearity.  It's the
    ratio of the largest to the smallest singular value of the design matrix.

    Args:
        X: A Pandas DataFrame containing the independent variables.

    Returns:
        The condition number.

    Interpretation:
        - Condition Number < 10: Weak multicollinearity.
        - 10 <= Condition Number <= 30: Moderate multicollinearity.
        - 30 < Condition Number <= 100: Strong multicollinearity.
        - Condition Number > 100: Severe multicollinearity.

    Example:
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> data = sm.datasets.get_rdataset("HousePrices", "AER", cache=True).data
        >>> X = data[["lotsize", "bedrooms", "bathrms"]]
        >>> cond_num = condition_number(X)
        >>> print(cond_num)

    """
    # Use numpy for singular value decomposition, more efficient than eigenvalues here.
    singular_values = np.linalg.svd(X, compute_uv=False)
    cond_num = singular_values[0] / singular_values[-1]
    return cond_num

def correlation_analysis(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation matrix of the independent variables.

    This provides a pairwise view of the linear relationships between predictors.

    Args:
        X: A Pandas DataFrame containing the independent variables.

    Returns:
        A Pandas DataFrame representing the correlation matrix.

    Interpretation:
        - |r| < 0.3: Weak correlation.
        - 0.3 <= |r| < 0.6: Moderate correlation.
        - 0.6 <= |r| < 0.8: Strong correlation.
        - |r| >= 0.8: Very strong correlation.

    Example:
        >>> import pandas as pd
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.get_rdataset("HousePrices", "AER", cache=True).data
        >>> X = data[["lotsize", "bedrooms", "bathrms"]]
        >>> correlations = correlation_analysis(X)
        >>> print(correlations)
    """
    return X.corr()


if __name__ == "__main__":
    import statsmodels.formula.api as smf

    # Load example dataset (House Prices)
    houseprices = sm.datasets.get_rdataset(dataname="HousePrices", package="AER", cache=True).data

    # Select independent variables (no need to add constant here for the example)
    X = houseprices[["lotsize", "bedrooms", "bathrooms"]]

    # Calculate VIFs
    vifs = variance_inflation_factor(X)
    print("Variance Inflation Factors (VIFs):\n", vifs)
    print("\nVIF Interpretation:")
    for var, vif in vifs.items():
        if vif > 10:
            print(f"  {var}: Severe multicollinearity (VIF = {vif:.2f})")
        elif 5 <= vif <= 10:
            print(f"  {var}: High multicollinearity (VIF = {vif:.2f})")
        elif 1 < vif < 5:
            print(f"  {var}: Moderate multicollinearity (VIF = {vif:.2f})")
        else:
            print(f"  {var}: No multicollinearity (VIF = {vif:.2f})")

    # Calculate condition number
    cond_num = condition_number(X)
    print(f"\nCondition Number: {cond_num:.2f}")
    if cond_num > 100:
        print("Interpretation: Severe multicollinearity")
    elif 30 < cond_num <= 100:
        print("Interpretation: Strong multicollinearity")
    elif 10 <= cond_num <= 30:
        print("Interpretation: Moderate multicollinearity")
    else:
        print("Interpretation: Weak multicollinearity")

    # Calculate correlation matrix
    correlations = correlation_analysis(X)
    print("\nCorrelation Matrix:\n", correlations)
    print("\nCorrelation Interpretation:")
    # Iterate through the upper triangle of the correlation matrix
    for i in range(len(correlations)):
        for j in range(i + 1, len(correlations)):
            var1 = correlations.columns[i]
            var2 = correlations.columns[j]
            corr = correlations.iloc[i, j]
            if abs(corr) >= 0.8:
                print(f"  {var1} and {var2}: Very strong correlation (r = {corr:.2f})")
            elif 0.6 <= abs(corr) < 0.8:
                print(f"  {var1} and {var2}: Strong correlation (r = {corr:.2f})")
            elif 0.3 <= abs(corr) < 0.6:
                print(f"  {var1} and {var2}: Moderate correlation (r = {corr:.2f})")
            else:
                print(f"  {var1} and {var2}: Weak correlation (r = {corr:.2f})")

