"""
Data loader module to fetch data from MongoDB collections and convert to pandas DataFrames.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# Import local modules
from db import mongo_connector
from utils import mock_data
import config

logger = logging.getLogger(__name__)

# Global flag to indicate if we're using mock data
_USING_MOCK_DATA = False

def is_using_mock_data() -> bool:
    """
    Check if the application is using mock data.
    
    Returns:
        bool: True if using mock data, False if using real data
    """
    return _USING_MOCK_DATA

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_prices(
    ticker: str, 
    start_date: datetime, 
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Load stock price data for a specific ticker and date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame: Stock price data or None if no data available
    """
    global _USING_MOCK_DATA
    
    try:
        if _USING_MOCK_DATA:
            # Use mock data
            logger.info(f"Using mock stock price data for {ticker}")
            return mock_data.generate_mock_stock_prices(ticker, start_date, end_date)
        
        # Try to load from MongoDB
        query = {
            "ticker": ticker,
            "date": {
                "$gte": start_date.strftime(config.DATE_FORMAT),
                "$lte": end_date.strftime(config.DATE_FORMAT)
            }
        }
        sort = [("date", 1)]  # Sort by date ascending
        
        documents = mongo_connector.find_documents(
            config.COLLECTION_STOCK_PRICES,
            query=query,
            sort=sort
        )
        
        if not documents:
            logger.warning(f"No stock price data found for {ticker} in date range")
            # Fallback to mock data
            _USING_MOCK_DATA = True
            return mock_data.generate_mock_stock_prices(ticker, start_date, end_date)
        
        df = pd.DataFrame(documents)
        
        # Convert date strings to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df = df.set_index('date')
        
        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        
        # Calculate moving averages for different windows
        for window in config.MA_WINDOWS:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading stock price data: {e}")
        # Fallback to mock data
        _USING_MOCK_DATA = True
        return mock_data.generate_mock_stock_prices(ticker, start_date, end_date)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_financial_reports(
    ticker: str, 
    years: Optional[List[int]] = None, 
    quarterly: bool = True
) -> Optional[pd.DataFrame]:
    """
    Load financial report data for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        years: List of years to filter by (None for all)
        quarterly: If True, return quarterly reports, else annual reports
        
    Returns:
        DataFrame: Financial report data or None if no data available
    """
    global _USING_MOCK_DATA
    
    try:
        if _USING_MOCK_DATA:
            # Use mock data
            logger.info(f"Using mock financial report data for {ticker}")
            return mock_data.generate_mock_financials(ticker, years or [datetime.now().year - i for i in range(5)], quarterly)
            
        # Try to load from MongoDB
        query = {"ticker": ticker}
        
        if years:
            query["year"] = {"$in": years}
            
        # For quarterly reports, include quarter field
        # For annual reports, exclude quarterly reports
        if quarterly:
            query["quarter"] = {"$exists": True}
        else:
            query["quarter"] = {"$exists": False}
            
        sort = [("year", -1)]
        if quarterly:
            sort = [("year", -1), ("quarter", -1)]
            
        documents = mongo_connector.find_documents(
            config.COLLECTION_FINANCIAL_REPORTS,
            query=query,
            sort=sort
        )
        
        if not documents:
            logger.warning(f"No financial report data found for {ticker}")
            # Fallback to mock data
            _USING_MOCK_DATA = True
            return mock_data.generate_mock_financials(ticker, years or [datetime.now().year - i for i in range(5)], quarterly)
        
        df = pd.DataFrame(documents)
        
        # Create period field for better display
        if quarterly:
            df['period'] = df.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
        else:
            df['period'] = df['year'].astype(str)
            
        # Calculate financial ratios
        if all(col in df.columns for col in ['net_income', 'total_assets']):
            df['roa'] = df['net_income'] / df['total_assets']
            
        if all(col in df.columns for col in ['net_income', 'total_equity']):
            df['roe'] = df['net_income'] / df['total_equity']
            
        if all(col in df.columns for col in ['net_income', 'revenue']):
            df['net_margin'] = df['net_income'] / df['revenue']
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading financial report data: {e}")
        # Fallback to mock data
        _USING_MOCK_DATA = True
        return mock_data.generate_mock_financials(ticker, years or [datetime.now().year - i for i in range(5)], quarterly)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sentiment_data(
    ticker: str, 
    start_date: datetime, 
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Load sentiment data for a specific ticker and date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame: Sentiment data or None if no data available
    """
    global _USING_MOCK_DATA
    
    try:
        if _USING_MOCK_DATA:
            # Use mock data
            logger.info(f"Using mock sentiment data for {ticker}")
            return mock_data.generate_mock_sentiment(ticker, start_date, end_date)
            
        # Try to load from MongoDB
        query = {
            "ticker": ticker,
            "date": {
                "$gte": start_date.strftime(config.DATE_FORMAT),
                "$lte": end_date.strftime(config.DATE_FORMAT)
            }
        }
        sort = [("date", 1)]  # Sort by date ascending
        
        documents = mongo_connector.find_documents(
            config.COLLECTION_SENTIMENT_NEWS,
            query=query,
            sort=sort
        )
        
        if not documents:
            logger.warning(f"No sentiment data found for {ticker} in date range")
            # Fallback to mock data
            _USING_MOCK_DATA = True
            return mock_data.generate_mock_sentiment(ticker, start_date, end_date)
        
        df = pd.DataFrame(documents)
        
        # Convert date strings to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate daily average sentiment score
        daily_avg = df.groupby('date')['score'].mean().reset_index()
        daily_avg = daily_avg.rename(columns={'score': 'avg_score'})
        
        # Calculate sentiment distribution
        sentiment_counts = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        sentiment_pivot = sentiment_counts.pivot(index='date', columns='sentiment', values='count').fillna(0)
        sentiment_pivot = sentiment_pivot.reset_index()
        
        # Merge daily average with distribution
        result = pd.merge(daily_avg, sentiment_pivot, on='date', how='left')
        
        # Ensure all sentiment columns exist
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment not in result.columns:
                result[sentiment] = 0
                
        # Set date as index
        result = result.set_index('date')
        
        # Calculate 7-day moving average of sentiment score
        result['sentiment_ma7'] = result['avg_score'].rolling(window=7).mean()
        
        return result
    
    except Exception as e:
        logger.error(f"Error loading sentiment data: {e}")
        # Fallback to mock data
        _USING_MOCK_DATA = True
        return mock_data.generate_mock_sentiment(ticker, start_date, end_date)

@st.cache_data
def get_available_tickers() -> List[str]:
    """
    Get a list of available tickers in the database.
    
    Returns:
        List[str]: List of ticker symbols
    """
    global _USING_MOCK_DATA
    
    try:
        if _USING_MOCK_DATA:
            return mock_data.get_available_mock_tickers()
            
        # Try to load from MongoDB
        tickers = mongo_connector.get_distinct_values(config.COLLECTION_STOCK_PRICES, 'ticker')
        
        if not tickers:
            # Fallback to mock data
            _USING_MOCK_DATA = True
            return mock_data.get_available_mock_tickers()
            
        return tickers
    
    except Exception as e:
        logger.error(f"Error getting available tickers: {e}")
        # Fallback to mock data
        _USING_MOCK_DATA = True
        return mock_data.get_available_mock_tickers()

@st.cache_data
def get_available_date_range(ticker: str) -> Tuple[datetime, datetime]:
    """
    Get the available date range for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple[datetime, datetime]: Start date and end date
    """
    global _USING_MOCK_DATA
    
    try:
        if _USING_MOCK_DATA:
            return mock_data.get_mock_date_range()
            
        # Try to load from MongoDB
        # Get min date
        pipeline = [
            {"$match": {"ticker": ticker}},
            {"$sort": {"date": 1}},
            {"$limit": 1},
            {"$project": {"_id": 0, "date": 1}}
        ]
        
        min_date_result = mongo_connector.aggregate(config.COLLECTION_STOCK_PRICES, pipeline)
        
        if not min_date_result:
            # Fallback to mock date range if no data available
            _USING_MOCK_DATA = True
            return mock_data.get_mock_date_range()
            
        min_date_str = min_date_result[0].get('date')
        min_date = datetime.strptime(min_date_str, config.DATE_FORMAT)
        
        # Get max date
        pipeline = [
            {"$match": {"ticker": ticker}},
            {"$sort": {"date": -1}},
            {"$limit": 1},
            {"$project": {"_id": 0, "date": 1}}
        ]
        
        max_date_result = mongo_connector.aggregate(config.COLLECTION_STOCK_PRICES, pipeline)
        max_date_str = max_date_result[0].get('date')
        max_date = datetime.strptime(max_date_str, config.DATE_FORMAT)
        
        return min_date, max_date
    
    except Exception as e:
        logger.error(f"Error getting available date range: {e}")
        # Fallback to mock date range
        _USING_MOCK_DATA = True
        return mock_data.get_mock_date_range()
        
def set_mock_data_mode(use_mock: bool = True):
    """
    Set the application to use mock data.
    
    Args:
        use_mock: Whether to use mock data (True) or real data (False)
    """
    global _USING_MOCK_DATA
    _USING_MOCK_DATA = use_mock
    if use_mock:
        logger.info("Mock data mode enabled")
    else:
        logger.info("Mock data mode disabled")