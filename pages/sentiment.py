"""
News sentiment visualization page for the stock dashboard application.
"""
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import data_loader
from utils import helpers, plotting

logger = logging.getLogger(__name__)

def render_sentiment_page(ticker: str, start_date: datetime, end_date: datetime):
    """
    Render the news sentiment visualization page.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
    """
    st.header(f"📰 News Sentiment Analysis: {ticker}")
    
    # Load sentiment data
    try:
        df_sentiment = data_loader.load_sentiment_data(ticker, start_date, end_date)
        
        if df_sentiment is None or df_sentiment.empty:
            helpers.display_no_data_message("sentiment", ticker)
            return
            
        # Display sentiment metrics
        _display_sentiment_metrics(df_sentiment)
        
        # Display sentiment distribution chart
        st.subheader("Sentiment Distribution")
        fig_distribution = plotting.plot_sentiment_distribution(df_sentiment)
        st.plotly_chart(fig_distribution, use_container_width=True)
        
        # Display sentiment trend chart
        st.subheader("Sentiment Score Trend")
        fig_trend = plotting.plot_sentiment_trend(df_sentiment)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Display sentiment correlation with price if price data is available
        _display_sentiment_price_correlation(ticker, start_date, end_date, df_sentiment)
        
    except Exception as e:
        logger.error(f"Error rendering sentiment page: {e}")
        helpers.display_error_message("Data Loading Error", str(e))

def _display_sentiment_metrics(df_sentiment: pd.DataFrame):
    """
    Display key sentiment metrics.
    
    Args:
        df_sentiment: DataFrame with sentiment data
    """
    # Calculate metrics
    total_news = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    if 'positive' in df_sentiment.columns:
        positive_count = df_sentiment['positive'].sum()
        total_news += positive_count
    
    if 'negative' in df_sentiment.columns:
        negative_count = df_sentiment['negative'].sum()
        total_news += negative_count
    
    if 'neutral' in df_sentiment.columns:
        neutral_count = df_sentiment['neutral'].sum()
        total_news += neutral_count
    
    avg_sentiment = df_sentiment['avg_score'].mean() if 'avg_score' in df_sentiment.columns else None
    
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total News Articles", f"{int(total_news):,}")
        
    with col2:
        positive_pct = (positive_count / total_news) * 100 if total_news > 0 else 0
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        
    with col3:
        neutral_pct = (neutral_count / total_news) * 100 if total_news > 0 else 0
        st.metric("Neutral Sentiment", f"{neutral_pct:.1f}%")
        
    with col4:
        negative_pct = (negative_count / total_news) * 100 if total_news > 0 else 0
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")

def _display_sentiment_price_correlation(
    ticker: str, 
    start_date: datetime, 
    end_date: datetime, 
    df_sentiment: pd.DataFrame
):
    """
    Display correlation between sentiment and stock price.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        df_sentiment: DataFrame with sentiment data
    """
    # Try to load price data
    try:
        df_price = data_loader.load_stock_prices(ticker, start_date, end_date)
        
        if df_price is None or df_price.empty:
            return
        
        st.subheader("Sentiment vs. Price Correlation")
        
        # Merge sentiment and price data on date
        df_sentiment_reset = df_sentiment.reset_index()
        df_price_reset = df_price.reset_index()
        
        merged_df = pd.merge(
            df_sentiment_reset[['date', 'avg_score']], 
            df_price_reset[['date', 'close', 'daily_return']], 
            on='date', 
            how='inner'
        )
        
        if merged_df.empty:
            st.info("No overlapping data to calculate correlation.")
            return
        
        # Calculate correlations
        corr_with_price = merged_df['avg_score'].corr(merged_df['close'])
        corr_with_return = merged_df['avg_score'].corr(merged_df['daily_return'])
        
        # Display correlations
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Correlation with Price", 
                f"{corr_with_price:.2f}",
                help="Correlation between sentiment score and closing price"
            )
        
        with col2:
            st.metric(
                "Correlation with Returns", 
                f"{corr_with_return:.2f}",
                help="Correlation between sentiment score and daily returns"
            )
        
        # Create scatter plot of sentiment vs returns
        if len(merged_df) > 5:  # Only show if we have enough data points
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=merged_df['avg_score'],
                    y=merged_df['daily_return'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=merged_df['avg_score'],
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=1,
                        showscale=True,
                        colorbar=dict(title="Sentiment Score")
                    ),
                    name='Daily Return vs Sentiment'
                )
            )
            
            # Add trendline
            from scipy import stats
            if len(merged_df) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    merged_df['avg_score'], merged_df['daily_return']
                )
                
                x_range = np.linspace(merged_df['avg_score'].min(), merged_df['avg_score'].max(), 100)
                y_range = slope * x_range + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        line=dict(color='black', width=2, dash='dash'),
                        name=f'Trend (r={r_value:.2f})'
                    )
                )
            
            fig.update_layout(
                title="Sentiment Score vs. Daily Returns",
                xaxis_title="Sentiment Score",
                yaxis_title="Daily Return",
                height=400,
                template='plotly_white'
            )
            
            # Format y-axis as percentage
            fig.update_yaxes(tickformat=".2%")
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error calculating sentiment-price correlation: {e}")
        # Silently fail - this is not a critical feature