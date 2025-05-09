"""
Helper utility functions for the stock dashboard application.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
import streamlit as st

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def flatten_json(nested_json: Dict, prefix: str = '') -> Dict:
    """
    Flatten a nested JSON into a single-level dictionary.
    
    Args:
        nested_json: Nested JSON/dictionary
        prefix: Prefix for flattened keys
        
    Returns:
        Dict: Flattened dictionary
    """
    flattened = {}
    
    for key, value in nested_json.items():
        new_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key))
        elif isinstance(value, list):
            # Handle lists by joining values if they're simple types
            if value and all(not isinstance(item, (dict, list)) for item in value):
                flattened[new_key] = ', '.join(str(item) for item in value)
            else:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flattened.update(flatten_json(item, f"{new_key}[{i}]"))
                    else:
                        flattened[f"{new_key}[{i}]"] = item
        else:
            flattened[new_key] = value
            
    return flattened

def validate_date_range(
    start_date: datetime, 
    end_date: datetime
) -> Tuple[bool, Optional[str]]:
    """
    Validate a date range to ensure it's valid.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Tuple[bool, str]: Tuple of (is_valid, error_message)
    """
    if start_date > end_date:
        return False, "Start date cannot be after end date"
    
    if end_date > datetime.now():
        return False, "End date cannot be in the future"
    
    if (end_date - start_date).days < 7:
        return False, "Date range must be at least 7 days"
    
    if (end_date - start_date).days > 3650:  # ~10 years
        return False, "Date range cannot exceed 10 years"
    
    return True, None

def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format a float as a percentage string.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"{value * 100:.{precision}f}%"

def format_large_number(value: float, precision: int = 2) -> str:
    """
    Format a large number with K, M, B suffixes.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    if abs(value) >= 1e9:
        return f"{value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"

def color_by_value(
    value: float, 
    neutral_threshold: float = 0.0
) -> str:
    """
    Return a color based on a value (positive, negative or neutral).
    
    Args:
        value: Value to evaluate
        neutral_threshold: Threshold for considering value as neutral
        
    Returns:
        str: CSS color string
    """
    if pd.isna(value) or value is None:
        return config.THEME_COLORS["neutral"]
    
    if value > neutral_threshold:
        return config.THEME_COLORS["positive"]
    elif value < -neutral_threshold:
        return config.THEME_COLORS["negative"]
    else:
        return config.THEME_COLORS["neutral"]

def display_error_message(error_type: str, message: str):
    """
    Display a standardized error message to the user.
    
    Args:
        error_type: Type of error
        message: Error message
    """
    st.error(f"**{error_type}**: {message}")

def display_no_data_message(data_type: str, ticker: str = None):
    """
    Display a standardized no-data message to the user.
    
    Args:
        data_type: Type of data that's missing
        ticker: Ticker symbol (optional)
    """
    ticker_text = f" for {ticker}" if ticker else ""
    st.warning(f"No {data_type} data available{ticker_text}. Try a different date range or ticker.")

def get_default_date_range() -> Tuple[datetime, datetime]:
    """
    Get a default date range of one year ending today.
    
    Returns:
        Tuple[datetime, datetime]: Start date and end date
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return start_date, end_date

def get_years_range(start_date: datetime, end_date: datetime) -> List[int]:
    """
    Get a list of years covered by a date range.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List[int]: List of years
    """
    return list(range(start_date.year, end_date.year + 1))

def get_color_scale(sentiment: str) -> str:
    """
    Get color for sentiment category.
    
    Args:
        sentiment: Sentiment category ('positive', 'negative', 'neutral')
        
    Returns:
        str: CSS color string
    """
    color_map = {
        'positive': config.THEME_COLORS['positive'],
        'negative': config.THEME_COLORS['negative'],
        'neutral': config.THEME_COLORS['neutral']
    }
    return color_map.get(sentiment, config.THEME_COLORS['neutral'])

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock price data.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        DataFrame: DataFrame with technical indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Relative Strength Index (RSI)
    # Using a simple implementation with 14-day window
    delta = result['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    result['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    window = 20
    result['bb_middle'] = result['close'].rolling(window=window).mean()
    result['bb_std'] = result['close'].rolling(window=window).std()
    result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
    result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
    
    # MACD
    result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
    result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
    result['macd'] = result['ema_12'] - result['ema_26']
    result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    return result