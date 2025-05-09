"""
Stock price visualization page for the stock dashboard application.
"""
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import data_loader
from utils import helpers, plotting

logger = logging.getLogger(__name__)

def render_stock_price_page(ticker: str, start_date: datetime, end_date: datetime):
    """
    Render the stock price visualization page.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
    """
    st.header(f"📊 Stock Price Analysis: {ticker}")
    
    # Load stock price data
    try:
        df_prices = data_loader.load_stock_prices(ticker, start_date, end_date)
        
        if df_prices is None or df_prices.empty:
            helpers.display_no_data_message("stock price", ticker)
            return
            
        # Calculate technical indicators
        df_prices = helpers.calculate_technical_indicators(df_prices)
        
        # Display price chart with moving averages
        st.subheader("Price Chart")
        fig_price = plotting.plot_stock_price(df_prices, ticker, ma_periods=config.MA_WINDOWS)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        latest_price = df_prices['close'].iloc[-1]
        prev_price = df_prices['close'].iloc[-2]
        price_change = latest_price - prev_price
        price_change_pct = price_change / prev_price
        
        period_return = (df_prices['close'].iloc[-1] / df_prices['close'].iloc[0]) - 1
        
        # Calculate volatility (standard deviation of returns)
        volatility = df_prices['daily_return'].std() * (252 ** 0.5)  # Annualized
        
        with col1:
            st.metric(
                "Latest Close", 
                f"Rp {latest_price:,.2f}", 
                f"{price_change_pct:.2%}"
            )
        
        with col2:
            st.metric(
                "Period Return", 
                f"{period_return:.2%}"
            )
            
        with col3:
            st.metric(
                "Volatility (Ann.)", 
                f"{volatility:.2%}"
            )
            
        with col4:
            avg_volume = df_prices['volume'].mean()
            st.metric(
                "Avg Volume", 
                helpers.format_large_number(avg_volume)
            )
        
        # Create tabs for additional charts
        tab1, tab2, tab3 = st.tabs(["Volume", "Returns", "Technical Indicators"])
        
        with tab1:
            # Display volume chart
            fig_volume = plotting.plot_volume(df_prices)
            st.plotly_chart(fig_volume, use_container_width=True)
            
        with tab2:
            # Display returns chart
            fig_returns = plotting.plot_returns(df_prices)
            st.plotly_chart(fig_returns, use_container_width=True)
            
        with tab3:
            # Technical indicators selection
            indicators = st.multiselect(
                "Select Technical Indicators",
                ["rsi", "macd", "bollinger"],
                default=["rsi"]
            )
            
            if indicators:
                fig_indicators = plotting.plot_technical_indicators(df_prices, indicators)
                st.plotly_chart(fig_indicators, use_container_width=True)
            else:
                st.info("Select at least one technical indicator to display.")
                
    except Exception as e:
        logger.error(f"Error rendering stock price page: {e}")
        helpers.display_error_message("Data Loading Error", str(e))