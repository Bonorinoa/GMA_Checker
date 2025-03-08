"""Streamlit app for testing Gauss-Markov assumptions in OLS regression models.

This module provides a user-friendly interface for:
1. Data upload and preview
2. Model specification using statsmodels formula
3. Selection of assumptions to test
4. Display of test results with interpretations
5. Visualization of diagnostic plots for each assumption
"""

import os
import sys
from pathlib import Path
from threading import RLock

# Add the project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, List, Optional, Tuple

# Import assumption tests from the package
from gauss_markov_checker.assumptions import (
    run_RESET_test,
    run_all_autocorrelation_tests,
    run_all_exogeneity_tests,
    run_all_homoscedasticity_tests,
    run_all_normality_tests
)

# Import visualization functions
from gauss_markov_checker.visualization.diagnostic_plots import (
    plot_residual_histogram,
    plot_qq,
    plot_residuals_vs_fitted,
    plot_scale_location,
    plot_leverage,
    plot_normality_diagnostics,
    plot_regression_diagnostics,
    plot_component_plus_residual,
    plot_correlation_heatmap,
    plot_acf_pacf
)

# Set page config with all keyword arguments
st.set_page_config(
    layout="wide", 
    page_title="Gauss-Markov Assumptions Checker", 
    page_icon="📊"
)

# Create a lock for matplotlib operations
_lock = RLock()

def load_example_data() -> pd.DataFrame:
    """Load example dataset from statsmodels."""
    try:
        data = sm.datasets.get_rdataset(dataname="HousePrices", package="AER")
        return data.data
    except:
        # Fallback to simple synthetic data if dataset loading fails
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 2 + 3*x1 + 4*x2 + np.random.normal(0, 1, n)
        return pd.DataFrame({
            'y': y,
            'x1': x1,
            'x2': x2
        })

def run_assumption_tests(
    model_results,
    assumptions: List[str]
) -> Dict:
    """Run selected assumption tests on the fitted model.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model
    assumptions : List[str]
        List of assumptions to test
        
    Returns
    -------
    Dict
        Dictionary containing test results for each assumption
    """
    results = {}
    
    if 'linearity' in assumptions:
        results['linearity'] = {'reset': run_RESET_test(model_results)}
    
    if 'autocorrelation' in assumptions:
        results['autocorrelation'] = run_all_autocorrelation_tests(model_results)
    
    if 'homoscedasticity' in assumptions:
        results['homoscedasticity'] = run_all_homoscedasticity_tests(model_results)
    
    if 'normality' in assumptions:
        results['normality'] = run_all_normality_tests(model_results)
    
    if 'exogeneity' in assumptions:
        # For OLS models, exogeneity tests are not applicable
        # since they require instrumental variables
        results['exogeneity'] = {
            'note': {
                'conclusion': 'Exogeneity tests (Wu-Hausman, Sargan, etc.) require instrumental variables and are not applicable to standard OLS models.',
                'recommendation': 'If you suspect endogeneity, consider using instrumental variables regression (IV/2SLS) instead of OLS.'
            }
        }
    
    return results

def display_test_results(results: Dict):
    """Display test results in a structured format using tabs.
    
    Parameters
    ----------
    results : Dict
        Dictionary containing test results for each assumption
    """
    st.header("Test Results")
    
    # Create tabs for each assumption
    assumption_tabs = st.tabs([assumption.title() for assumption in results.keys()])
    
    for i, (assumption, tests) in enumerate(results.items()):
        with assumption_tabs[i]:
            if isinstance(tests, dict):
                # Create expanders for each test within the assumption
                for test_name, test_results in tests.items():
                    with st.expander(f"{test_name.replace('_', ' ').title()}", expanded=True):
                        # Create two columns for test statistics and conclusion
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            # Display test statistics
                            for key, value in test_results.items():
                                if key != 'conclusion' and key != 'recommendation':
                                    # Special handling for Ljung-Box test which returns a DataFrame
                                    if key == 'full_results' and isinstance(value, (pd.DataFrame, pd.Series)):
                                        st.subheader("Ljung-Box Test Results by Lag")
                                        # Format the DataFrame for better display
                                        formatted_df = value.copy()
                                        # Format p-values to be more readable
                                        if 'lb_pvalue' in formatted_df.columns:
                                            formatted_df['lb_pvalue'] = formatted_df['lb_pvalue'].apply(
                                                lambda x: f"{x:.4e}" if x < 0.0001 else f"{x:.4f}"
                                            )
                                        st.dataframe(formatted_df, use_container_width=True)
                                    elif isinstance(value, (float, np.float64)):
                                        st.write(f"**{key}:** {value:.4f}")
                                    else:
                                        st.write(f"**{key}:** {value}")
                        
                        with col2:
                            # Display conclusion with colored box
                            if 'conclusion' in test_results:
                                if "reject" in test_results['conclusion'].lower():
                                    st.error(test_results['conclusion'])
                                else:
                                    st.success(test_results['conclusion'])
                            
                            # Display recommendation if available
                            if 'recommendation' in test_results:
                                st.info(test_results['recommendation'])
                                
                            # For Ljung-Box test, add an explanation about the multiple lags
                            if test_name.lower() == 'ljung_box':
                                st.info("""
                                The Ljung-Box test is performed at multiple lags to detect 
                                autocorrelation at different time distances. A significant 
                                result (p < 0.05) at any lag suggests the presence of 
                                autocorrelation in the residuals.
                                """)

def display_visualizations(model_results):
    """Display diagnostic visualizations for the model.
    
    Parameters
    ----------
    model_results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS model
    """
    st.header("Diagnostic Visualizations")
    
    # Create tabs for different types of visualizations
    viz_tabs = st.tabs([
        "Comprehensive Diagnostics", 
        "Normality", 
        "Homoscedasticity", 
        "Autocorrelation",
        "Multicollinearity"
    ])
    
    # Comprehensive diagnostics tab
    with viz_tabs[0]:
        st.subheader("Regression Diagnostics")
        
        st.write("""
        These plots provide a comprehensive overview of the model diagnostics:
        
        - **Top Left**: Residuals vs Fitted Values - Checks linearity and homoscedasticity
        - **Top Right**: Q-Q Plot - Checks normality of residuals
        - **Bottom Left**: Scale-Location Plot - Checks homoscedasticity
        - **Bottom Right**: Residuals vs Leverage - Identifies influential observations
        """)
        
        with _lock:
            fig, axes = plot_regression_diagnostics(model_results, show=False)
            st.pyplot(fig)
    
    # Normality tab
    with viz_tabs[1]:
        st.subheader("Normality Diagnostics")
        
        st.write("""
        These plots help assess whether the residuals follow a normal distribution, which is 
        an assumption for valid hypothesis testing in OLS regression.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Histogram of Residuals**")
            st.write("The histogram should approximate a normal curve if residuals are normally distributed.")
            with _lock:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_residual_histogram(model_results, show=False, ax=ax)
                st.pyplot(fig)
        
        with col2:
            st.write("**Q-Q Plot**")
            st.write("Points should follow the 45-degree line if residuals are normally distributed.")
            with _lock:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_qq(model_results, show=False, ax=ax)
                st.pyplot(fig)
    
    # Homoscedasticity tab
    with viz_tabs[2]:
        st.subheader("Homoscedasticity Diagnostics")
        
        st.write("""
        These plots help assess whether the residuals have constant variance across all levels of the 
        fitted values (homoscedasticity), which is a key assumption of OLS regression.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Residuals vs Fitted Values**")
            st.write("Residuals should be randomly scattered around zero with no clear pattern.")
            with _lock:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_residuals_vs_fitted(model_results, show=False, ax=ax)
                st.pyplot(fig)
        
        with col2:
            st.write("**Scale-Location Plot**")
            st.write("The red line should be approximately horizontal if variance is constant.")
            with _lock:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_scale_location(model_results, show=False, ax=ax)
                st.pyplot(fig)
    
    # Autocorrelation tab
    with viz_tabs[3]:
        st.subheader("Autocorrelation Diagnostics")
        
        st.write("""
        The plots below show the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) 
        of the residuals. These plots help identify patterns of autocorrelation at different lags.
        
        - **ACF**: Shows correlation between residuals at different lags
        - **PACF**: Shows direct correlation between residuals at different lags, controlling for intermediate lags
        
        Significant spikes (extending beyond the blue confidence bands) indicate autocorrelation at those lags.
        """)
        
        with _lock:
            fig, axes = plot_acf_pacf(model_results, show=False)
            st.pyplot(fig)
            
        st.info("""
        **Interpretation Guide:**
        
        - For **no autocorrelation**: Both ACF and PACF should show no significant spikes
        - For **AR(p) process**: PACF cuts off after lag p, while ACF tails off
        - For **MA(q) process**: ACF cuts off after lag q, while PACF tails off
        - For **ARMA(p,q) process**: Both ACF and PACF tail off
        
        In the context of regression residuals, significant autocorrelation indicates violation of the 
        independence assumption of the Gauss-Markov theorem.
        """)
    
    # Multicollinearity tab
    with viz_tabs[4]:
        st.subheader("Multicollinearity Diagnostics")
        
        st.write("""
        Multicollinearity occurs when predictor variables are highly correlated with each other, 
        which can lead to unstable coefficient estimates and inflated standard errors.
        """)
        
        st.write("**Correlation Heatmap of Predictors**")
        st.write("High absolute correlation values (close to 1 or -1) indicate potential multicollinearity.")
        
        with _lock:
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_correlation_heatmap(model_results, show=False, ax=ax)
            st.pyplot(fig)
        
        # Component plus residual plots for each predictor
        st.subheader("Component-Plus-Residual Plots")
        
        st.write("""
        These plots help assess the linearity of the relationship between each predictor and the response, 
        controlling for other predictors. The red line represents the fitted relationship, while the green 
        dashed line represents a nonparametric smoother. Deviations between these lines suggest non-linearity.
        """)
        
        # Get predictor variables (excluding the constant)
        predictors = [var for var in model_results.model.exog_names if var != 'const']
        
        # Create columns based on the number of predictors
        cols = st.columns(min(3, len(predictors)))
        
        for i, var in enumerate(predictors):
            with cols[i % len(cols)]:
                with _lock:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    plot_component_plus_residual(model_results, var, show=False, ax=ax)
                    st.pyplot(fig)

def main():
    """Main function for the Streamlit app."""
    
    # Title and introduction
    st.title("📊 Gauss-Markov Assumptions Checker")
    st.write("""
    This app helps you test whether your OLS regression model satisfies the Gauss-Markov 
    assumptions necessary for the estimator to be BLUE (Best Linear Unbiased Estimator).
    
    The app combines formal hypothesis tests with visual diagnostics to provide both 
    statistical rigor and intuitive understanding.
    """)
    
    # Sidebar for data input and model specification
    with st.sidebar:
        st.header("Data Input")
        
        # Data upload
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV", "Use example data"]
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=["csv"]
            )
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return
            else:
                st.info("Please upload a CSV file")
                return
        else:
            data = load_example_data()
        
        # Model specification
        st.header("Model Specification")
        
        # Show available variables
        st.write("Available variables:")
        st.write(", ".join(data.columns))
        
        # Formula input with help text
        formula_help = """
        Statsmodels formula examples:
        - Simple: y ~ x1 + x2
        - With transformation: y ~ np.log(x1) + np.sqrt(x2)
        - With interaction: y ~ x1 + x2 + x1:x2
        
        Note: For mathematical functions (log, sqrt, etc.), 
        you must use the 'np.' prefix (e.g., np.log, np.sqrt)
        """
        
        # Add a note about numpy functions
        st.info("💡 Remember to use 'np.' prefix for mathematical functions (e.g., np.log, np.sqrt)")
        
        formula = st.text_input(
            "Enter model formula:",
            value="np.log(price) ~ lotsize + bedrooms" if data_source == "Use example data" else "",
            help=formula_help
        )
    
    # Main panel
    # Data preview
    st.header("Data Preview")
    st.dataframe(data.head(), use_container_width=True)
    
    # Assumption selection
    st.header("Select Assumptions to Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_all = st.checkbox("Test all assumptions", value=True)
    
    assumptions_to_test = []
    
    if test_all:
        assumptions_to_test = [
            'linearity',
            'autocorrelation',
            'homoscedasticity',
            'normality',
            'exogeneity'
        ]
    else:
        with col2:
            assumptions_to_test = st.multiselect(
                "Select assumptions to test:",
                [
                    'linearity',
                    'autocorrelation',
                    'homoscedasticity',
                    'normality',
                    'exogeneity'
                ]
            )
    
    # Run tests button
    if st.button("Run Tests", type="primary"):
        try:
            # Show a spinner while running tests
            with st.spinner("Running tests and generating visualizations..."):
                # Fit the model using statsmodels formula API
                model = smf.ols(formula=formula, data=data).fit()
                
                # Create tabs for different sections
                summary_tab, results_tab, viz_tab = st.tabs([
                    "Model Summary", 
                    "Test Results", 
                    "Visualizations"
                ])
                
                # Model summary tab
                with summary_tab:
                    st.header("Model Summary")
                    st.text(str(model.summary()))
                
                # Run selected tests
                results = run_assumption_tests(model, assumptions_to_test)
                
                # Test results tab
                with results_tab:
                    display_test_results(results)
                
                # Visualizations tab
                with viz_tab:
                    display_visualizations(model)
                
        except Exception as e:
            st.error(f"Error running tests: {str(e)}")
            st.error("Please check your model formula and data for errors.")

if __name__ == "__main__":
    main()
