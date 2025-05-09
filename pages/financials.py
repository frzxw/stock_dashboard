"""
Financial reports visualization page for the stock dashboard application.
"""
import logging
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import streamlit as st

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import data_loader
from utils import helpers

logger = logging.getLogger(__name__)

def render_financials_page(ticker: str, start_date: datetime, end_date: datetime):
    """
    Render the financial reports visualization page.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
    """
    st.header(f"💰 Financial Reports: {ticker}")
    
    # Get years from date range
    years = helpers.get_years_range(start_date, end_date)
    
    # Create tabs for quarterly and annual reports
    tab1, tab2 = st.tabs(["Quarterly Reports", "Annual Reports"])
    
    with tab1:
        _render_quarterly_reports(ticker, years)
        
    with tab2:
        _render_annual_reports(ticker, years)

def _render_quarterly_reports(ticker: str, years: List[int]):
    """
    Render quarterly financial reports.
    
    Args:
        ticker: Stock ticker symbol
        years: List of years to include
    """
    try:
        # Load quarterly financial data
        df_quarterly = data_loader.load_financial_reports(ticker, years, quarterly=True)
        
        if df_quarterly is None or df_quarterly.empty:
            helpers.display_no_data_message("quarterly financial", ticker)
            return
        
        # Format the data for display
        df_display = _format_financial_data(df_quarterly)
        
        # Display key metrics
        _display_key_metrics(df_display, "Quarterly")
        
        # Display the full table
        st.subheader("Quarterly Financial Data")
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True
        )
        
    except Exception as e:
        logger.error(f"Error rendering quarterly reports: {e}")
        helpers.display_error_message("Data Loading Error", str(e))

def _render_annual_reports(ticker: str, years: List[int]):
    """
    Render annual financial reports.
    
    Args:
        ticker: Stock ticker symbol
        years: List of years to include
    """
    try:
        # Load annual financial data
        df_annual = data_loader.load_financial_reports(ticker, years, quarterly=False)
        
        if df_annual is None or df_annual.empty:
            helpers.display_no_data_message("annual financial", ticker)
            return
        
        # Format the data for display
        df_display = _format_financial_data(df_annual)
        
        # Display key metrics
        _display_key_metrics(df_display, "Annual")
        
        # Display the full table
        st.subheader("Annual Financial Data")
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True
        )
        
        # Display growth rates
        _display_growth_rates(df_annual)
        
    except Exception as e:
        logger.error(f"Error rendering annual reports: {e}")
        helpers.display_error_message("Data Loading Error", str(e))

def _format_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format financial data for display.
    
    Args:
        df: DataFrame with financial data
        
    Returns:
        DataFrame: Formatted for display
    """
    # Create a copy to avoid modifying the original
    df_display = df.copy()
    
    # Select columns to display (customize based on your data)
    display_columns = [
        'period', 'revenue', 'operating_income', 'net_income',
        'total_assets', 'total_liabilities', 'total_equity',
        'operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow',
        'roa', 'roe', 'net_margin'
    ]
    
    # Keep only columns that exist in the dataframe
    display_columns = [col for col in display_columns if col in df_display.columns]
    
    df_display = df_display[display_columns]
    
    # Format monetary values
    monetary_columns = [
        'revenue', 'operating_income', 'net_income',
        'total_assets', 'total_liabilities', 'total_equity',
        'operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow'
    ]
    
    for col in monetary_columns:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(helpers.format_large_number)
    
    # Format ratio values
    ratio_columns = ['roa', 'roe', 'net_margin']
    
    for col in ratio_columns:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(helpers.format_percentage)
    
    # Rename columns for display
    column_names = {
        'period': 'Period',
        'revenue': 'Revenue',
        'operating_income': 'Operating Income',
        'net_income': 'Net Income',
        'total_assets': 'Total Assets',
        'total_liabilities': 'Total Liabilities',
        'total_equity': 'Total Equity',
        'operating_cash_flow': 'Operating Cash Flow',
        'investing_cash_flow': 'Investing Cash Flow',
        'financing_cash_flow': 'Financing Cash Flow',
        'roa': 'ROA',
        'roe': 'ROE',
        'net_margin': 'Net Margin'
    }
    
    df_display = df_display.rename(columns=column_names)
    
    return df_display

def _display_key_metrics(df_display: pd.DataFrame, report_type: str):
    """
    Display key financial metrics.
    
    Args:
        df_display: DataFrame with formatted financial data
        report_type: Type of report ("Quarterly" or "Annual")
    """
    st.subheader(f"{report_type} Key Metrics")
    
    # Check if we have enough columns to display metrics
    required_cols = ["Period", "Revenue", "Net Income", "ROA", "ROE", "Net Margin"]
    if not all(col in df_display.columns for col in required_cols):
        st.info("Insufficient data to display key metrics.")
        return
    
    latest_period = df_display.iloc[0]
    
    # Create metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latest Period", latest_period["Period"])
        st.metric("Revenue", latest_period["Revenue"])
        
    with col2:
        st.metric("Net Income", latest_period["Net Income"])
        st.metric("Net Margin", latest_period["Net Margin"])
        
    with col3:
        st.metric("ROE", latest_period["ROE"])
        st.metric("ROA", latest_period["ROA"])

def _display_growth_rates(df_annual: pd.DataFrame):
    """
    Calculate and display annual growth rates.
    
    Args:
        df_annual: DataFrame with annual financial data
    """
    # Check if we have enough data for growth rates
    if len(df_annual) < 2:
        return
    
    st.subheader("Annual Growth Rates")
    
    # Sort by year ascending for calculations
    df_sorted = df_annual.sort_values(by='year')
    
    # Calculate growth rates for key metrics
    growth_metrics = ['revenue', 'net_income', 'total_assets']
    
    # Create a dictionary for growth rates
    growth_data = {'period': []}
    
    for metric in growth_metrics:
        if metric in df_sorted.columns:
            growth_data[f'{metric}_growth'] = []
    
    # Calculate year-over-year growth rates
    for i in range(1, len(df_sorted)):
        prev_row = df_sorted.iloc[i-1]
        curr_row = df_sorted.iloc[i]
        
        growth_data['period'].append(f"{curr_row['year']}")
        
        for metric in growth_metrics:
            if metric in df_sorted.columns:
                prev_value = prev_row[metric]
                curr_value = curr_row[metric]
                
                if prev_value and prev_value != 0:
                    growth = (curr_value - prev_value) / abs(prev_value)
                    growth_data[f'{metric}_growth'].append(growth)
                else:
                    growth_data[f'{metric}_growth'].append(None)
    
    # Create DataFrame for display
    df_growth = pd.DataFrame(growth_data)
    
    # Format the growth rates
    for metric in growth_metrics:
        if f'{metric}_growth' in df_growth.columns:
            df_growth[f'{metric}_growth'] = df_growth[f'{metric}_growth'].apply(
                lambda x: helpers.format_percentage(x) if x is not None else 'N/A'
            )
    
    # Rename columns for display
    column_names = {
        'period': 'Year',
        'revenue_growth': 'Revenue Growth',
        'net_income_growth': 'Net Income Growth',
        'total_assets_growth': 'Total Assets Growth'
    }
    
    df_growth = df_growth.rename(columns=column_names)
    
    # Display the growth rates table
    st.dataframe(
        df_growth,
        hide_index=True,
        use_container_width=True
    )