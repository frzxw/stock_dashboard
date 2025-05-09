"""
Stock Dashboard Application - Main entry point.

This Streamlit application visualizes stock data from MongoDB collections.
The data includes stock prices, financial reports, and news sentiment analysis.
"""
import logging
import os
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from db import mongo_connector
import config
import data_loader
from utils import helpers
from pages import stock_price, financials, sentiment

# Set page config
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide"
)

def main():
    """Main function to render the Streamlit application."""
    # Display title
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    
    # Check MongoDB connection in sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Try to connect to MongoDB
        db_connected = False
        try:
            client = mongo_connector.get_mongo_client()
            db = client[config.MONGO_DB_NAME]
            collections = db.list_collection_names()
            db_connected = True
            
            # Check if required collections exist
            required_collections = [
                config.COLLECTION_STOCK_PRICES,
                config.COLLECTION_FINANCIAL_REPORTS,
                config.COLLECTION_SENTIMENT_NEWS
            ]
            
            missing_collections = [c for c in required_collections if c not in collections]
            
            if missing_collections:
                st.warning(f"⚠️ Missing collections: {', '.join(missing_collections)}")
                data_loader.set_mock_data_mode(True)
                st.info("Using mock data for missing collections.")
            else:
                data_loader.set_mock_data_mode(False)
                st.success("✅ Connected to MongoDB")
                st.info(f"📊 All required collections are available.")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            st.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            data_loader.set_mock_data_mode(True)
            st.info("Using mock data. MongoDB connection not available.")
        
        # Data source indicator
        if data_loader.is_using_mock_data():
            st.info("📊 Data Source: Mock Data (Generated on-the-fly)")
            data_source_msg = "Mock data is being used since MongoDB is not available or missing collections."
        else:
            st.info("📊 Data Source: MongoDB")
            data_source_msg = "Data is being retrieved from MongoDB collections."
            
        with st.expander("About Data Source"):
            st.write(data_source_msg)
            if data_loader.is_using_mock_data():
                st.write("Mock data includes:")
                st.write("- Simulated stock prices with trend and volatility")
                st.write("- Synthetic financial reports with realistic growth rates")
                st.write("- Generated sentiment analysis with randomized distributions")
        
        st.divider()
        
        # Ticker selection
        try:
            available_tickers = data_loader.get_available_tickers()
            if not available_tickers:
                available_tickers = config.STOCK_TICKERS
                
            ticker = st.selectbox(
                "Select Stock Ticker",
                options=available_tickers,
                index=0 if available_tickers else None
            )
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            ticker = st.selectbox("Select Stock Ticker", options=config.STOCK_TICKERS)
            st.error("Failed to load tickers. Using default list.")
            
        st.divider()
        
        # Date range selection
        st.subheader("Select Date Range")
        
        try:
            # Get available date range for selected ticker
            min_date, max_date = data_loader.get_available_date_range(ticker)
            
            default_end_date = max_date
            default_start_date = max_date - timedelta(days=365) if max_date else datetime.now() - timedelta(days=365)
            
            if default_start_date < min_date:
                default_start_date = min_date
            
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            default_start_date, default_end_date = helpers.get_default_date_range()
            st.error("Failed to load date range. Using default range.")
        
        # Date picker widgets
        start_date = st.date_input(
            "Start Date",
            value=default_start_date.date() if isinstance(default_start_date, datetime) else default_start_date,
            min_value=min_date.date() if 'min_date' in locals() and isinstance(min_date, datetime) else None,
            max_value=default_end_date.date() if isinstance(default_end_date, datetime) else default_end_date
        )
        
        end_date = st.date_input(
            "End Date",
            value=default_end_date.date() if isinstance(default_end_date, datetime) else default_end_date,
            min_value=start_date,
            max_value=datetime.now().date()
        )
        
        # Convert to datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        # Validate date range
        is_valid, error_msg = helpers.validate_date_range(start_datetime, end_datetime)
        if not is_valid:
            st.error(error_msg)
            return
    
    # Main content area
    # Create tabs for different pages
    tab1, tab2, tab3 = st.tabs(["Stock Price", "Financials", "News Sentiment"])
    
    with tab1:
        stock_price.render_stock_price_page(ticker, start_datetime, end_datetime)
        
    with tab2:
        financials.render_financials_page(ticker, start_datetime, end_datetime)
        
    with tab3:
        sentiment.render_sentiment_page(ticker, start_datetime, end_datetime)
    
    # Display footer
    st.divider()
    data_label = "Mock data (generated on-the-fly)" if data_loader.is_using_mock_data() else "MongoDB collections populated by ETL processes"
    st.caption(f"{config.APP_TITLE} - Data sourced from {data_label}")
    st.caption("© 2025 - Built with Streamlit")

if __name__ == "__main__":
    main()