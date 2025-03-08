"""
Diagnostic plots for OLS regression assumptions.

This module provides visualization functions for assessing the various
Gauss-Markov assumptions in OLS regression models. The plots complement
the formal statistical tests in the assumptions package.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from typing import Tuple, Optional, List, Union, Dict
import pandas as pd


def plot_residual_histogram(
    model_results: RegressionResults,
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Histogram of Residuals with Normal Curve',
    color: str = 'lightblue',
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a histogram of residuals with a normal curve overlay.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    bins : int, default=20
        Number of bins for the histogram
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, default='Histogram of Residuals with Normal Curve'
        Plot title
    color : str, default='lightblue'
        Color for the histogram bars
    show : bool, default=True
        Whether to display the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    ax.hist(model_results.resid, bins=bins, density=True, alpha=0.6, 
            color=color, edgecolor='black', label='Residuals')
    
    # Normal curve overlay
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - np.mean(model_results.resid)) / np.std(model_results.resid))**2) / \
        (np.std(model_results.resid) * np.sqrt(2 * np.pi))
    ax.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
    
    ax.set_title(title)
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_qq(
    model_results: RegressionResults,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Q-Q Plot of Residuals',
    line: str = '45',
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a Q-Q plot of residuals to assess normality.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, default='Q-Q Plot of Residuals'
        Plot title
    line : {'45', 's', 'r', 'q'}, default='45'
        Reference line to be plotted:
        - '45': 45-degree line
        - 's': standardized line
        - 'r': robust line
        - 'q': no line
    show : bool, default=True
        Whether to display the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create Q-Q plot
    qq_plot = sm.qqplot(model_results.resid, line=line, ax=ax, fit=True)
    
    ax.set_title(title)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_residuals_vs_fitted(
    model_results: RegressionResults,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Residuals vs Fitted Values',
    color: str = 'blue',
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot residuals against fitted values to check for heteroscedasticity and linearity.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, default='Residuals vs Fitted Values'
        Plot title
    color : str, default='blue'
        Color for the scatter points
    show : bool, default=True
        Whether to display the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(model_results.fittedvalues, model_results.resid, 
               alpha=0.6, color=color, edgecolor='k')
    
    # Reference line at y=0
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add a lowess smoother
    lowess = sm.nonparametric.lowess(model_results.resid, model_results.fittedvalues, frac=2/3)
    ax.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1)
    
    ax.set_title(title)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_scale_location(
    model_results: RegressionResults,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Scale-Location Plot',
    color: str = 'blue',
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a scale-location plot (sqrt of standardized residuals vs fitted values).
    
    This plot is useful for checking the homoscedasticity assumption.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, default='Scale-Location Plot'
        Plot title
    color : str, default='blue'
        Color for the scatter points
    show : bool, default=True
        Whether to display the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate standardized residuals
    std_resid = model_results.get_influence().resid_studentized_internal
    
    # Take the square root of the absolute standardized residuals
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))
    
    # Scatter plot
    ax.scatter(model_results.fittedvalues, sqrt_abs_resid, 
               alpha=0.6, color=color, edgecolor='k')
    
    # Add a lowess smoother
    lowess = sm.nonparametric.lowess(sqrt_abs_resid, model_results.fittedvalues, frac=2/3)
    ax.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1)
    
    ax.set_title(title)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Standardized Residuals|')
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_leverage(
    model_results: RegressionResults,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Residuals vs Leverage',
    color: str = 'blue',
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a residuals vs leverage plot to identify influential observations.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, default='Residuals vs Leverage'
        Plot title
    color : str, default='blue'
        Color for the scatter points
    show : bool, default=True
        Whether to display the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get influence measures
    influence = model_results.get_influence()
    leverage = influence.hat_matrix_diag
    
    # Standardized residuals
    std_resid = influence.resid_studentized_internal
    
    # Scatter plot
    ax.scatter(leverage, std_resid, alpha=0.6, color=color, edgecolor='k')
    
    # Reference line at y=0
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add a lowess smoother
    lowess = sm.nonparametric.lowess(std_resid, leverage, frac=2/3)
    ax.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1)
    
    # Add Cook's distance contours
    p = len(model_results.params)
    n = len(model_results.resid)
    
    for cook_d in [0.5, 1.0]:
        cook_x = np.linspace(0, max(leverage) * 1.1, 100)
        cook_y = np.sqrt(cook_d * p * (1 - cook_x) / cook_x)
        ax.plot(cook_x, cook_y, 'r--', lw=0.5, label=f"Cook's distance = {cook_d}")
        ax.plot(cook_x, -cook_y, 'r--', lw=0.5)
    
    ax.set_title(title)
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_normality_diagnostics(
    model_results: RegressionResults,
    figsize: Tuple[int, int] = (12, 10),
    show: bool = True
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a comprehensive set of plots for assessing normality of residuals.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    figsize : tuple, default=(12, 10)
        Figure size (width, height) in inches
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    tuple
        Figure and list of axes containing the plots
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Histogram with normal curve
    plot_residual_histogram(model_results, show=False, ax=axes[0])
    
    # Q-Q plot
    plot_qq(model_results, show=False, ax=axes[1])
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes


def plot_regression_diagnostics(
    model_results: RegressionResults,
    figsize: Tuple[int, int] = (12, 10),
    show: bool = True
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a comprehensive set of diagnostic plots for OLS regression.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    figsize : tuple, default=(12, 10)
        Figure size (width, height) in inches
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    tuple
        Figure and list of axes containing the plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Residuals vs Fitted
    plot_residuals_vs_fitted(model_results, show=False, ax=axes[0])
    
    # Q-Q plot
    plot_qq(model_results, show=False, ax=axes[1])
    
    # Scale-Location
    plot_scale_location(model_results, show=False, ax=axes[2])
    
    # Residuals vs Leverage
    plot_leverage(model_results, show=False, ax=axes[3])
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes


def plot_component_plus_residual(
    model_results: RegressionResults,
    variable: str,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    color: str = 'blue',
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a component-plus-residual plot (partial residual plot) for a specific variable.
    
    This plot helps assess linearity of the relationship between a predictor and the response,
    controlling for other predictors.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    variable : str
        Name of the predictor variable to plot
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, a default title is used
    color : str, default='blue'
        Color for the scatter points
    show : bool, default=True
        Whether to display the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get the variable index
    var_idx = model_results.model.exog_names.index(variable)
    
    # Calculate component and residuals
    x = model_results.model.exog[:, var_idx]
    coef = model_results.params[var_idx]
    comp_plus_resid = coef * x + model_results.resid
    
    # Create scatter plot
    ax.scatter(x, comp_plus_resid, alpha=0.6, color=color, edgecolor='k')
    
    # Add regression line
    ax.plot(x, coef * x + model_results.params[0], 'r-', lw=1)
    
    # Add lowess smoother
    lowess = sm.nonparametric.lowess(comp_plus_resid, x, frac=2/3)
    ax.plot(lowess[:, 0], lowess[:, 1], 'g--', lw=1)
    
    if title is None:
        title = f'Component-Plus-Residual Plot for {variable}'
    ax.set_title(title)
    ax.set_xlabel(variable)
    ax.set_ylabel(f'Component+Residual')
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_correlation_heatmap(
    model_results: RegressionResults,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu',
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a correlation heatmap for predictor variables.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    cmap : str, default='RdBu'
        Color map for the heatmap
    show : bool, default=True
        Whether to display the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlation matrix
    exog_df = pd.DataFrame(model_results.model.exog, columns=model_results.model.exog_names)
    corr_matrix = exog_df.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0,
                vmin=-1, vmax=1, ax=ax, fmt='.2f')
    
    ax.set_title('Correlation Heatmap of Predictors')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_acf_pacf(
    model_results: RegressionResults,
    lags: int = 40,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create ACF and PACF plots of residuals to check for autocorrelation.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model results
    lags : int, default=40
        Number of lags to include in the plots
    figsize : tuple, default=(12, 5)
        Figure size (width, height) in inches
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    tuple
        Figure and list of axes containing the plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF plot
    sm.graphics.tsa.plot_acf(model_results.resid, lags=lags, ax=axes[0],
                            title='Autocorrelation Function')
    
    # PACF plot
    sm.graphics.tsa.plot_pacf(model_results.resid, lags=lags, ax=axes[1],
                             title='Partial Autocorrelation Function')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, axes


if __name__ == "__main__":
    # Example usage with a simple dataset
    import statsmodels.formula.api as smf
    
    # Load example dataset
    data = sm.datasets.get_rdataset("Guerry", "HistData").data
    
    # Fit a simple OLS model
    model = smf.ols(formula="Lottery ~ Literacy + np.log(Pop1831)", data=data).fit()
    
    # Create diagnostic plots
    print("Creating normality diagnostic plots...")
    plot_normality_diagnostics(model)
    
    print("\nCreating comprehensive regression diagnostic plots...")
    plot_regression_diagnostics(model)
    
    print("\nCreating component-plus-residual plot...")
    plot_component_plus_residual(model, "Literacy")
    
    print("\nCreating correlation heatmap...")
    plot_correlation_heatmap(model)
    
    print("\nCreating ACF and PACF plots...")
    plot_acf_pacf(model)
