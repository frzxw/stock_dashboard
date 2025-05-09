"""
Global configuration settings for the stock dashboard application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "stock_dashboard")

# Collections
COLLECTION_FINANCIAL_REPORTS = "financial_reports"
COLLECTION_STOCK_PRICES = "stock_prices"
COLLECTION_SENTIMENT_NEWS = "sentiment_news"

# Stock tickers
STOCK_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI",  # Banks
    "TLKM", "EXCL", "ISAT",          # Telecommunication
    "ASII", "SRIL", "INTP",          # Manufacturing
    "UNVR", "ICBP", "INDF"           # Consumer goods
]

# Date format
DATE_FORMAT = "%Y-%m-%d"

# App configuration
APP_TITLE = "Indonesian Stock Market Dashboard"
APP_ICON = "📈"

# Visualization configuration
THEME_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f"
}

# Moving average windows
MA_WINDOWS = [20, 50, 200]