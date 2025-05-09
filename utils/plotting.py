"""
Plotting utility functions for the stock dashboard application.
"""
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import get_color_scale

def plot_stock_price(
    df: pd.DataFrame,
    ticker: str,
    ma_periods: List[int] = None
) -> go.Figure:
    """
    Create a plotly figure with stock price and moving averages.
    
    Args:
        df: DataFrame with stock price data
        ticker: Stock ticker symbol
        ma_periods: List of moving average periods to plot
        
    Returns:
        Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Reset index to make date a column
    df_plot = df.reset_index()
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_plot['date'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price',
            increasing_line_color=config.THEME_COLORS['positive'],
            decreasing_line_color=config.THEME_COLORS['negative']
        )
    )
    
    # Add moving averages
    if ma_periods:
        for period in ma_periods:
            col_name = f'ma_{period}'
            if col_name in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot['date'],
                        y=df_plot[col_name],
                        name=f'MA {period}',
                        line=dict(width=1.5)
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )
    
    return fig

def plot_volume(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart of trading volume.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        Figure: Plotly figure object
    """
    # Reset index to make date a column
    df_plot = df.reset_index()
    
    fig = go.Figure()
    
    # Add volume bars
    colors = [
        config.THEME_COLORS['positive'] if row['close'] >= row['open'] else config.THEME_COLORS['negative']
        for _, row in df_plot.iterrows()
    ]
    
    fig.add_trace(
        go.Bar(
            x=df_plot['date'],
            y=df_plot['volume'],
            marker_color=colors,
            name='Volume'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300,
        template='plotly_white'
    )
    
    # Update y-axis to use SI units (K, M, etc.)
    fig.update_yaxes(ticksuffix="")
    
    # Update x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )
    
    return fig

def plot_returns(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart of daily returns.
    
    Args:
        df: DataFrame with stock price data including daily returns
        
    Returns:
        Figure: Plotly figure object
    """
    # Reset index to make date a column
    df_plot = df.reset_index()
    
    fig = go.Figure()
    
    # Add returns line
    fig.add_trace(
        go.Scatter(
            x=df_plot['date'],
            y=df_plot['daily_return'],
            mode='lines',
            name='Daily Return',
            line=dict(color=config.THEME_COLORS['primary'], width=1)
        )
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=df_plot['date'].min(),
        y0=0,
        x1=df_plot['date'].max(),
        y1=0,
        line=dict(color=config.THEME_COLORS['neutral'], width=1, dash="dash")
    )
    
    # Update layout
    fig.update_layout(
        title="Daily Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=300,
        template='plotly_white'
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".2%")
    
    # Update x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )
    
    return fig

def plot_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> go.Figure:
    """
    Create plots of technical indicators.
    
    Args:
        df: DataFrame with stock price data and indicators
        indicators: List of indicators to plot
        
    Returns:
        Figure: Plotly figure object
    """
    # Reset index to make date a column
    df_plot = df.reset_index()
    
    # Create subplots depending on indicators required
    rows = len(indicators)
    fig = make_subplots(
        rows=rows, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[ind.upper() for ind in indicators]
    )
    
    # Add traces for each indicator
    row = 1
    for ind in indicators:
        if ind == 'rsi':
            # RSI line
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'], 
                    y=df_plot['rsi_14'], 
                    name="RSI (14)",
                    line=dict(color=config.THEME_COLORS['primary'])
                ), 
                row=row, 
                col=1
            )
            
            # Add overbought/oversold threshold lines
            fig.add_shape(
                type="line",
                x0=df_plot['date'].min(),
                y0=70,
                x1=df_plot['date'].max(),
                y1=70,
                line=dict(color=config.THEME_COLORS['negative'], width=1, dash="dot"),
                row=row,
                col=1
            )
            
            fig.add_shape(
                type="line",
                x0=df_plot['date'].min(),
                y0=30,
                x1=df_plot['date'].max(),
                y1=30,
                line=dict(color=config.THEME_COLORS['positive'], width=1, dash="dot"),
                row=row,
                col=1
            )
            
        elif ind == 'macd':
            # MACD line
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'],
                    y=df_plot['macd'],
                    name="MACD",
                    line=dict(color=config.THEME_COLORS['primary'])
                ), 
                row=row, 
                col=1
            )
            
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'],
                    y=df_plot['macd_signal'],
                    name="Signal",
                    line=dict(color=config.THEME_COLORS['secondary'])
                ), 
                row=row, 
                col=1
            )
            
            # MACD histogram
            colors = [
                config.THEME_COLORS['positive'] if val >= 0 else config.THEME_COLORS['negative']
                for val in df_plot['macd_hist']
            ]
            
            fig.add_trace(
                go.Bar(
                    x=df_plot['date'],
                    y=df_plot['macd_hist'],
                    name="MACD Hist",
                    marker_color=colors
                ), 
                row=row, 
                col=1
            )
            
            # Zero line
            fig.add_shape(
                type="line",
                x0=df_plot['date'].min(),
                y0=0,
                x1=df_plot['date'].max(),
                y1=0,
                line=dict(color=config.THEME_COLORS['neutral'], width=1, dash="dash"),
                row=row,
                col=1
            )
            
        elif ind == 'bollinger':
            # Price
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'], 
                    y=df_plot['close'], 
                    name="Price",
                    line=dict(color=config.THEME_COLORS['primary'])
                ), 
                row=row, 
                col=1
            )
            
            # Middle band
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'], 
                    y=df_plot['bb_middle'], 
                    name="Middle Band",
                    line=dict(color=config.THEME_COLORS['neutral'], width=1)
                ), 
                row=row, 
                col=1
            )
            
            # Upper and lower bands with fill
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'], 
                    y=df_plot['bb_upper'], 
                    name="Upper Band",
                    line=dict(color=config.THEME_COLORS['secondary'], width=1)
                ), 
                row=row, 
                col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'], 
                    y=df_plot['bb_lower'], 
                    name="Lower Band",
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    line=dict(color=config.THEME_COLORS['secondary'], width=1)
                ), 
                row=row, 
                col=1
            )
        
        row += 1
    
    # Update layout
    fig.update_layout(
        height=300 * rows,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis for all subplots
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )
    
    return fig

def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a stacked bar chart of sentiment distribution.
    
    Args:
        df: DataFrame with sentiment data
        
    Returns:
        Figure: Plotly figure object
    """
    # Reset index to make date a column
    df_plot = df.reset_index()
    
    # Prepare data for plotting
    plot_data = []
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in df_plot.columns:
            plot_data.append(
                go.Bar(
                    x=df_plot['date'],
                    y=df_plot[sentiment],
                    name=sentiment.capitalize(),
                    marker_color=get_color_scale(sentiment)
                )
            )
    
    # Create figure
    fig = go.Figure(data=plot_data)
    
    # Update layout
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Date",
        yaxis_title="Count",
        barmode='stack',
        height=400,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_sentiment_trend(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart of sentiment score trend.
    
    Args:
        df: DataFrame with sentiment data
        
    Returns:
        Figure: Plotly figure object
    """
    # Reset index to make date a column
    df_plot = df.reset_index()
    
    fig = go.Figure()
    
    # Add scatter plot for daily average sentiment
    fig.add_trace(
        go.Scatter(
            x=df_plot['date'],
            y=df_plot['avg_score'],
            mode='markers',
            name='Daily Score',
            marker=dict(
                color=df_plot['avg_score'],
                colorscale=[
                    config.THEME_COLORS['negative'],
                    config.THEME_COLORS['neutral'],
                    config.THEME_COLORS['positive']
                ],
                cmin=0,
                cmax=1,
                size=8
            )
        )
    )
    
    # Add 7-day moving average line
    if 'sentiment_ma7' in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot['date'],
                y=df_plot['sentiment_ma7'],
                mode='lines',
                name='7-Day MA',
                line=dict(color=config.THEME_COLORS['secondary'], width=2)
            )
        )
    
    # Add horizontal line at neutral sentiment (0.5)
    fig.add_shape(
        type="line",
        x0=df_plot['date'].min(),
        y0=0.5,
        x1=df_plot['date'].max(),
        y1=0.5,
        line=dict(color=config.THEME_COLORS['neutral'], width=1, dash="dash")
    )
    
    # Update layout
    fig.update_layout(
        title="Sentiment Score Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score (0-1)",
        height=400,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axis range
    fig.update_yaxes(range=[0, 1])
    
    return fig