"""
Scripts package for Nova Financial Analysis project.
"""

# Import Task 3 functions for easier access
try:
    from .task3_correlation_analysis import (
        load_news_data,
        load_stock_data,
        clean_news_data,
        clean_stock_data,
        normalize_dates,
        calculate_sentiment,
        calculate_stock_returns,
        aggregate_sentiment,
        merge_data,
        calculate_correlations,
        save_results,
        STOCK_TICKERS,
        MARKET_CLOSE_HOUR
    )
except ImportError:
    # If relative import fails, allow direct import
    pass

__all__ = [
    'load_news_data',
    'load_stock_data',
    'clean_news_data',
    'clean_stock_data',
    'normalize_dates',
    'calculate_sentiment',
    'calculate_stock_returns',
    'aggregate_sentiment',
    'merge_data',
    'calculate_correlations',
    'save_results',
    'STOCK_TICKERS',
    'MARKET_CLOSE_HOUR'
]

