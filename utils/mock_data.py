"""
Mock data generator module for stock dashboard application.
Provides mock data when MongoDB connection is not available.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

def generate_mock_stock_prices(
    ticker: str, 
    start_date: datetime, 
    end_date: datetime
) -> pd.DataFrame:
    """
    Generate mock stock price data for a given ticker and date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame: Mock stock price data
    """
    # Generate date range (exclude weekends for realism)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
    
    # Set initial price based on ticker (just for variety)
    ticker_hash = sum(ord(c) for c in ticker)
    base_price = 1000 + (ticker_hash % 10000)
    
    # Generate random price movements with some trend
    n = len(date_range)
    # Add trend component
    trend = np.linspace(0, 0.3 * base_price * (np.random.random() - 0.5), n)
    # Add random component
    noise = np.random.normal(0, base_price * 0.015, n)
    # Add cyclical component
    cycle = base_price * 0.1 * np.sin(np.linspace(0, 5 * np.pi, n))
    
    # Combine components
    closes = base_price + trend + noise + cycle
    closes = np.maximum(closes, base_price * 0.5)  # Ensure prices don't go too low
    
    # Generate high, low, open values based on close
    daily_volatility = base_price * 0.02
    highs = closes + np.abs(np.random.normal(0, daily_volatility, n))
    lows = closes - np.abs(np.random.normal(0, daily_volatility, n))
    opens = lows + np.random.random(n) * (highs - lows)
    
    # Generate volume
    volume = np.random.normal(1000000, 300000, n).astype(int)
    volume = np.maximum(volume, 100000)  # Ensure minimum volume
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'ticker': ticker,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    })
    
    # Set date as index
    df = df.set_index('date')
    
    # Calculate returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate moving averages
    for window in config.MA_WINDOWS:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
    
    return df

def generate_mock_financials(
    ticker: str, 
    years: List[int], 
    quarterly: bool = True
) -> pd.DataFrame:
    """
    Generate mock financial report data for a given ticker and years.
    
    Args:
        ticker: Stock ticker symbol
        years: List of years to generate data for
        quarterly: Whether to generate quarterly or annual data
        
    Returns:
        DataFrame: Mock financial report data
    """
    ticker_hash = sum(ord(c) for c in ticker)
    base_revenue = 1000000000 + (ticker_hash % 5000000000)  # Base annual revenue ~1-6B
    
    data = []
    
    # Set growth rates
    annual_growth = 0.05 + 0.1 * np.random.random()  # 5-15% annual growth
    
    for year in sorted(years):
        # Apply annual growth to base revenue
        year_idx = year - min(years)
        year_revenue = base_revenue * ((1 + annual_growth) ** year_idx)
        
        if quarterly:
            # Generate quarterly data
            quarters = [1, 2, 3, 4]
            for quarter in quarters:
                # Add some seasonality
                season_factor = 0.9 + 0.2 * (quarter / 4)
                quarter_revenue = year_revenue * 0.25 * season_factor
                
                # Generate financial metrics
                operating_margin = 0.15 + 0.05 * np.random.random()
                net_margin = 0.08 + 0.04 * np.random.random()
                operating_income = quarter_revenue * operating_margin
                net_income = quarter_revenue * net_margin
                
                # Balance sheet items
                total_assets = quarter_revenue * 1.5
                total_equity = total_assets * 0.4
                total_liabilities = total_assets - total_equity
                
                # Cash flow items
                operating_cash_flow = net_income * 1.2
                investing_cash_flow = -quarter_revenue * 0.1
                financing_cash_flow = -operating_cash_flow - investing_cash_flow
                
                data.append({
                    'ticker': ticker,
                    'year': year,
                    'quarter': quarter,
                    'revenue': quarter_revenue,
                    'operating_income': operating_income,
                    'net_income': net_income,
                    'total_assets': total_assets,
                    'total_liabilities': total_liabilities,
                    'total_equity': total_equity,
                    'operating_cash_flow': operating_cash_flow,
                    'investing_cash_flow': investing_cash_flow,
                    'financing_cash_flow': financing_cash_flow
                })
        else:
            # Generate annual data
            operating_margin = 0.15 + 0.05 * np.random.random()
            net_margin = 0.08 + 0.04 * np.random.random()
            operating_income = year_revenue * operating_margin
            net_income = year_revenue * net_margin
            
            # Balance sheet items
            total_assets = year_revenue * 1.5
            total_equity = total_assets * 0.4
            total_liabilities = total_assets - total_equity
            
            # Cash flow items
            operating_cash_flow = net_income * 1.2
            investing_cash_flow = -year_revenue * 0.1
            financing_cash_flow = -operating_cash_flow - investing_cash_flow
            
            data.append({
                'ticker': ticker,
                'year': year,
                'revenue': year_revenue,
                'operating_income': operating_income,
                'net_income': net_income,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'total_equity': total_equity,
                'operating_cash_flow': operating_cash_flow,
                'investing_cash_flow': investing_cash_flow,
                'financing_cash_flow': financing_cash_flow
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate financial ratios
    df['roa'] = df['net_income'] / df['total_assets']
    df['roe'] = df['net_income'] / df['total_equity']
    df['net_margin'] = df['net_income'] / df['revenue']
    
    # Create period field for better display
    if quarterly:
        df['period'] = df.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
    else:
        df['period'] = df['year'].astype(str)
    
    # Sort by year and quarter (if applicable) in descending order
    if quarterly:
        df = df.sort_values(by=['year', 'quarter'], ascending=False)
    else:
        df = df.sort_values(by='year', ascending=False)
    
    return df

def generate_mock_sentiment(
    ticker: str, 
    start_date: datetime, 
    end_date: datetime
) -> pd.DataFrame:
    """
    Generate mock sentiment data for a given ticker and date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame: Mock sentiment data
    """
    # Generate date range (exclude weekends for realism)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate base sentiment scores with some trend based on ticker
    ticker_hash = sum(ord(c) for c in ticker)
    base_sentiment = 0.5 + (ticker_hash % 10) / 100
    
    # Sentiment varies between 0 and 1, with a baseline around 0.5
    n = len(date_range)
    
    # Add trend component
    trend = np.linspace(0, 0.2 * (np.random.random() - 0.5), n)
    
    # Add random component
    noise = np.random.normal(0, 0.05, n)
    
    # Combine components
    sentiment_scores = base_sentiment + trend + noise
    sentiment_scores = np.clip(sentiment_scores, 0.1, 0.9)  # Keep within 0.1-0.9 range
    
    # Generate daily sentiment distribution
    data = []
    for i, date in enumerate(date_range):
        avg_score = sentiment_scores[i]
        
        # Number of articles per day (random)
        num_articles = np.random.randint(3, 15)
        
        # Calculate sentiment distribution based on average score
        if avg_score > 0.6:
            # More positive sentiment
            positive = np.random.randint(max(1, int(num_articles * 0.5)), num_articles)
            negative = np.random.randint(0, num_articles - positive)
            neutral = num_articles - positive - negative
        elif avg_score < 0.4:
            # More negative sentiment
            negative = np.random.randint(max(1, int(num_articles * 0.5)), num_articles)
            positive = np.random.randint(0, num_articles - negative)
            neutral = num_articles - positive - negative
        else:
            # Balanced sentiment
            neutral = np.random.randint(max(1, int(num_articles * 0.3)), num_articles)
            remaining = num_articles - neutral
            positive = np.random.randint(0, remaining)
            negative = remaining - positive
        
        data.append({
            'date': date,
            'avg_score': avg_score,
            'positive': positive,
            'neutral': neutral,
            'negative': negative
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set date as index
    df = df.set_index('date')
    
    # Calculate 7-day moving average of sentiment score
    df['sentiment_ma7'] = df['avg_score'].rolling(window=7).mean()
    
    return df

def get_available_mock_tickers() -> List[str]:
    """
    Get list of available mock tickers.
    
    Returns:
        List[str]: List of ticker symbols
    """
    return config.STOCK_TICKERS

def get_mock_date_range() -> Tuple[datetime, datetime]:
    """
    Get a default date range for mock data.
    
    Returns:
        Tuple[datetime, datetime]: Start date and end date
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return start_date, end_date