"""
Task 3: Correlation Analysis Between News Sentiment and Stock Movements

This script implements the complete Task 3 workflow:
1. Data loading and profiling
2. Date alignment and normalization
3. Sentiment analysis (VADER and TextBlob)
4. Stock returns calculation
5. Sentiment aggregation
6. Correlation analysis
7. Visualization and insights

Author: Data Analyst - Nova Financial Solutions
Date: 2024
"""

# -*- coding: utf-8 -*-
"""
Configure encoding for Windows console compatibility
"""
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        # Try to set console to UTF-8
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # Fallback if stdout/stderr don't have buffer attribute
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

# Sentiment Analysis Libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Statistical Analysis
from scipy.stats import pearsonr

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize NLTK resources
def setup_nltk():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

# Initialize VADER
setup_nltk()
vader = SentimentIntensityAnalyzer()

# Constants
MARKET_CLOSE_HOUR = 16  # 4 PM EST/EDT
STOCK_TICKERS = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA']


def load_news_data(file_path, target_stocks=None):
    """
    Load news data from CSV file, filtering for target stocks.
    
    Parameters:
    -----------
    file_path : str
        Path to the news data CSV file
    target_stocks : list
        List of stock tickers to filter for
        
    Returns:
    --------
    pd.DataFrame
        Loaded and filtered news data
    """
    print("📰 Loading Financial News Dataset...")
    
    if target_stocks is None:
        target_stocks = STOCK_TICKERS
    
    chunk_size = 100000
    news_chunks = []
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            chunk_filtered = chunk[chunk['stock'].isin(target_stocks)]
            if not chunk_filtered.empty:
                news_chunks.append(chunk_filtered)
        
        if news_chunks:
            news_df = pd.concat(news_chunks, ignore_index=True)
            print(f"✅ News data loaded: {len(news_df):,} records")
            return news_df
        else:
            print("⚠️ No matching records found")
            return pd.DataFrame()
            
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading news data: {e}")
        return pd.DataFrame()


def load_stock_data(data_path, tickers=None):
    """
    Load stock price data from Excel files.
    
    Parameters:
    -----------
    data_path : str
        Path to directory containing stock price Excel files
    tickers : list
        List of stock tickers to load
        
    Returns:
    --------
    pd.DataFrame
        Combined stock price data
    """
    print("\n📈 Loading Stock Price Data...")
    
    if tickers is None:
        tickers = STOCK_TICKERS
    
    stock_data_dict = {}
    
    for ticker in tickers:
        file_path = os.path.join(data_path, f'{ticker}.xlsx')
        try:
            df = pd.read_excel(
                file_path,
                index_col='Date',
                parse_dates=True,
                engine='openpyxl',
                date_format='%m/%d/%Y'
            )
            df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            stock_data_dict[ticker] = df
            print(f"  ✅ {ticker}: {len(df):,} records")
        except FileNotFoundError:
            print(f"  ❌ {ticker}: File not found")
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")
    
    if stock_data_dict:
        stock_prices_df = pd.concat(stock_data_dict.values(), ignore_index=True)
        stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'])
        print(f"\n✅ Total stock price records: {len(stock_prices_df):,}")
        return stock_prices_df
    else:
        print("\n❌ No stock price data loaded")
        return pd.DataFrame()


def clean_news_data(news_df):
    """
    Clean and prepare news data for analysis.
    
    Parameters:
    -----------
    news_df : pd.DataFrame
        Raw news data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned news data
    """
    print("\n🧹 Cleaning News Data...")
    
    if news_df.empty:
        return news_df
    
    news_df_clean = news_df.copy()
    news_df_clean.columns = news_df_clean.columns.str.strip().str.lower()
    
    # Remove rows with missing critical fields
    initial_count = len(news_df_clean)
    news_df_clean = news_df_clean.dropna(subset=['headline', 'stock', 'date'])
    print(f"  Removed {initial_count - len(news_df_clean)} rows with missing data")
    
    # Filter target stocks
    news_df_clean = news_df_clean[news_df_clean['stock'].isin(STOCK_TICKERS)]
    
    # Clean headline text
    news_df_clean['headline'] = news_df_clean['headline'].str.strip()
    news_df_clean = news_df_clean[news_df_clean['headline'].str.len() > 0]
    
    print(f"✅ Cleaned news data: {len(news_df_clean):,} records")
    return news_df_clean


def clean_stock_data(stock_prices_df):
    """
    Clean and prepare stock price data.
    
    Parameters:
    -----------
    stock_prices_df : pd.DataFrame
        Raw stock price data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned stock price data
    """
    print("\n🧹 Cleaning Stock Price Data...")
    
    if stock_prices_df.empty:
        return stock_prices_df
    
    stock_prices_clean = stock_prices_df.copy()
    
    # Remove rows with missing prices or dates
    initial_count = len(stock_prices_clean)
    stock_prices_clean = stock_prices_clean.dropna(subset=['Close', 'Date'])
    print(f"  Removed {initial_count - len(stock_prices_clean)} rows with missing data")
    
    # Remove duplicates
    initial_count = len(stock_prices_clean)
    stock_prices_clean = stock_prices_clean.drop_duplicates(subset=['Ticker', 'Date'], keep='first')
    print(f"  Removed {initial_count - len(stock_prices_clean)} duplicate entries")
    
    # Sort
    stock_prices_clean = stock_prices_clean.sort_values(['Ticker', 'Date'])
    
    print(f"✅ Cleaned stock price data: {len(stock_prices_clean):,} records")
    return stock_prices_clean


def normalize_dates(news_df):
    """
    Normalize dates and align news with trading days.
    
    Parameters:
    -----------
    news_df : pd.DataFrame
        News data with date column
        
    Returns:
    --------
    pd.DataFrame
        News data with aligned dates
    """
    print("\n📅 Normalizing Dates for News Data...")
    
    if news_df.empty:
        return news_df
    
    # Convert date to datetime with timezone
    news_df['date_parsed'] = pd.to_datetime(news_df['date'], errors='coerce', utc=True)
    news_df['timestamp_nyc'] = news_df['date_parsed'].dt.tz_convert('America/New_York')
    
    # Determine aligned trading date
    def determine_aligned_date(timestamp):
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.hour >= MARKET_CLOSE_HOUR:
            return (timestamp.normalize() + pd.Timedelta(days=1)).normalize()
        else:
            return timestamp.normalize()
    
    news_df['aligned_date'] = news_df['timestamp_nyc'].apply(determine_aligned_date)
    
    # Remove invalid dates
    initial_count = len(news_df)
    news_df = news_df.dropna(subset=['aligned_date'])
    print(f"  Removed {initial_count - len(news_df)} rows with invalid dates")
    
    print(f"✅ Date normalization complete: {len(news_df):,} records")
    return news_df


def calculate_sentiment(news_df):
    """
    Calculate sentiment scores using VADER and TextBlob.
    
    Parameters:
    -----------
    news_df : pd.DataFrame
        News data with headlines
        
    Returns:
    --------
    pd.DataFrame
        News data with sentiment scores
    """
    print("\n💭 Performing Sentiment Analysis...")
    
    if news_df.empty:
        return news_df
    
    # VADER Sentiment
    def calculate_vader_sentiment(text):
        if pd.isna(text) or text == '':
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        scores = vader.polarity_scores(str(text))
        return scores
    
    print("  Calculating VADER scores...")
    sentiment_scores = news_df['headline'].apply(calculate_vader_sentiment)
    news_df['vader_compound'] = sentiment_scores.apply(lambda x: x['compound'])
    news_df['vader_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    news_df['vader_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    news_df['vader_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    
    # TextBlob Sentiment (sample for large datasets)
    print("  Calculating TextBlob scores...")
    if len(news_df) > 100000:
        print(f"    Large dataset ({len(news_df):,} records). Sampling 50,000 records...")
        sample_df = news_df.sample(n=50000, random_state=42)
        textblob_scores = sample_df['headline'].apply(
            lambda x: TextBlob(str(x)).sentiment if pd.notna(x) and x != '' else (0.0, 0.0)
        )
        news_df.loc[sample_df.index, 'textblob_polarity'] = textblob_scores.apply(lambda x: x[0])
        news_df.loc[sample_df.index, 'textblob_subjectivity'] = textblob_scores.apply(lambda x: x[1])
        news_df['textblob_polarity'] = news_df['textblob_polarity'].fillna(0.0)
        news_df['textblob_subjectivity'] = news_df['textblob_subjectivity'].fillna(0.0)
    else:
        textblob_scores = news_df['headline'].apply(
            lambda x: TextBlob(str(x)).sentiment if pd.notna(x) and x != '' else (0.0, 0.0)
        )
        news_df['textblob_polarity'] = textblob_scores.apply(lambda x: x[0])
        news_df['textblob_subjectivity'] = textblob_scores.apply(lambda x: x[1])
    
    print("✅ Sentiment analysis complete")
    return news_df


def calculate_stock_returns(stock_prices_df):
    """
    Calculate daily stock returns.
    
    Parameters:
    -----------
    stock_prices_df : pd.DataFrame
        Stock price data
        
    Returns:
    --------
    pd.DataFrame
        Stock price data with daily returns
    """
    print("\n📈 Calculating Daily Stock Returns...")
    
    if stock_prices_df.empty:
        return stock_prices_df
    
    stock_prices_df = stock_prices_df.sort_values(['Ticker', 'Date'])
    stock_prices_df['daily_return'] = stock_prices_df.groupby('Ticker')['Close'].pct_change()
    stock_prices_df = stock_prices_df.dropna(subset=['daily_return'])
    
    print(f"✅ Daily returns calculated: {len(stock_prices_df):,} records")
    return stock_prices_df


def aggregate_sentiment(news_df):
    """
    Aggregate daily sentiment scores by stock and date.
    
    Parameters:
    -----------
    news_df : pd.DataFrame
        News data with sentiment scores
        
    Returns:
    --------
    pd.DataFrame
        Aggregated sentiment data
    """
    print("\n📊 Aggregating Daily Sentiment Scores...")
    
    if news_df.empty:
        return pd.DataFrame()
    
    # Validate required columns exist
    required_cols = ['stock', 'aligned_date', 'vader_compound', 'vader_pos', 'vader_neg', 'textblob_polarity']
    missing_cols = [col for col in required_cols if col not in news_df.columns]
    
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(news_df.columns)}")
        raise KeyError(f"Missing required columns for aggregation: {missing_cols}")
    
    sentiment_aggregated = news_df.groupby(['stock', 'aligned_date']).agg({
        'vader_compound': ['mean', 'max', 'min', 'count'],
        'vader_pos': 'mean',
        'vader_neg': 'mean',
        'textblob_polarity': 'mean'
    }).reset_index()
    
    # Flatten column names
    sentiment_aggregated.columns = ['stock', 'aligned_date', 'avg_vader_compound', 'max_vader_compound',
                                     'min_vader_compound', 'news_count', 'avg_vader_pos', 'avg_vader_neg',
                                     'avg_textblob_polarity']
    
    # Remove timezone from aligned_date to ensure compatibility with stock price dates
    if isinstance(sentiment_aggregated['aligned_date'].dtype, pd.DatetimeTZDtype):
        sentiment_aggregated['aligned_date'] = sentiment_aggregated['aligned_date'].dt.tz_localize(None)
    # Normalize to date-only (no time component)
    sentiment_aggregated['aligned_date'] = pd.to_datetime(sentiment_aggregated['aligned_date']).dt.normalize()
    
    print(f"✅ Sentiment aggregation complete: {len(sentiment_aggregated):,} daily records")
    return sentiment_aggregated


def merge_data(stock_prices_df, sentiment_aggregated):
    """
    Merge stock returns with aggregated sentiment.
    
    Parameters:
    -----------
    stock_prices_df : pd.DataFrame
        Stock price data with returns
    sentiment_aggregated : pd.DataFrame
        Aggregated sentiment data
        
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("\n🔗 Merging Sentiment and Stock Returns Data...")
    
    if stock_prices_df.empty or sentiment_aggregated.empty:
        return pd.DataFrame()
    
    # Make copies to avoid modifying the original
    sentiment_merged = sentiment_aggregated.copy()
    stock_prices_merged = stock_prices_df.copy()
    
    # Remove timezone info from aligned_date to match stock_prices_df['Date'] (timezone-naive)
    # Use more robust check for timezone-aware datetime columns
    if isinstance(sentiment_merged['aligned_date'].dtype, pd.DatetimeTZDtype):
        sentiment_merged['aligned_date'] = sentiment_merged['aligned_date'].dt.tz_localize(None)
    elif hasattr(sentiment_merged['aligned_date'].dtype, 'tz') and sentiment_merged['aligned_date'].dtype.tz is not None:
        sentiment_merged['aligned_date'] = sentiment_merged['aligned_date'].dt.tz_localize(None)
    
    # Ensure both Date columns are timezone-naive and normalized (date only, no time)
    # First ensure stock Date is timezone-naive
    if isinstance(stock_prices_merged['Date'].dtype, pd.DatetimeTZDtype):
        stock_prices_merged['Date'] = stock_prices_merged['Date'].dt.tz_localize(None)
    
    # Normalize both to date-only (no time component)
    stock_prices_merged['Date'] = pd.to_datetime(stock_prices_merged['Date']).dt.tz_localize(None).dt.normalize()
    sentiment_merged['aligned_date'] = pd.to_datetime(sentiment_merged['aligned_date']).dt.tz_localize(None).dt.normalize()
    
    merged_df = pd.merge(
        stock_prices_merged[['Ticker', 'Date', 'Close', 'daily_return']],
        sentiment_merged,
        left_on=['Ticker', 'Date'],
        right_on=['stock', 'aligned_date'],
        how='left'
    )
    
    # Fill missing sentiment with 0 (neutral)
    sentiment_cols = ['avg_vader_compound', 'max_vader_compound', 'min_vader_compound',
                      'avg_vader_pos', 'avg_vader_neg', 'avg_textblob_polarity', 'news_count']
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
    merged_df = merged_df.dropna(subset=['daily_return'])
    
    print(f"✅ Data merge complete: {len(merged_df):,} records")
    return merged_df


def calculate_correlations(merged_df):
    """
    Calculate Pearson correlation coefficients.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged sentiment and returns data
        
    Returns:
    --------
    pd.DataFrame
        Correlation results
    """
    print("\n🔗 Calculating Pearson Correlation Coefficients...")
    print("=" * 80)
    
    if merged_df.empty:
        return pd.DataFrame()
    
    sentiment_metrics = ['avg_vader_compound', 'max_vader_compound', 'min_vader_compound',
                         'avg_vader_pos', 'avg_vader_neg', 'avg_textblob_polarity']
    
    correlation_results = []
    
    for ticker in STOCK_TICKERS:
        ticker_data = merged_df[merged_df['Ticker'] == ticker]
        
        if len(ticker_data) > 10:
            for metric in sentiment_metrics:
                corr, p_value = pearsonr(ticker_data[metric], ticker_data['daily_return'])
                correlation_results.append({
                    'Ticker': ticker,
                    'Sentiment_Metric': metric,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
    
    correlation_df = pd.DataFrame(correlation_results)
    
    print("\n📊 CORRELATION RESULTS BY TICKER")
    print("=" * 80)
    
    for ticker in STOCK_TICKERS:
        ticker_corr = correlation_df[correlation_df['Ticker'] == ticker]
        if not ticker_corr.empty:
            print(f"\n{ticker}:")
            print(f"  {'Metric':<30} {'Correlation':<15} {'P-Value':<15} {'Significant':<10}")
            print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")
            for _, row in ticker_corr.iterrows():
                sig_marker = '✓' if row['Significant'] else '✗'
                print(f"  {row['Sentiment_Metric']:<30} {row['Correlation']:>10.4f}     {row['P_Value']:>10.4f}     {sig_marker:>10}")
    
    return correlation_df


def save_results(merged_df, correlation_df, sentiment_aggregated, output_dir='../data/processed/'):
    """
    Save processed data and results to CSV files.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataset
    correlation_df : pd.DataFrame
        Correlation results
    sentiment_aggregated : pd.DataFrame
        Aggregated sentiment
    output_dir : str
        Output directory path
    """
    print("\n💾 Saving Results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not merged_df.empty:
        merged_df.to_csv(os.path.join(output_dir, 'task3_merged_sentiment_returns.csv'), index=False)
        print(f"  ✅ Saved merged dataset: {len(merged_df):,} records")
    
    if not correlation_df.empty:
        correlation_df.to_csv(os.path.join(output_dir, 'task3_correlation_results.csv'), index=False)
        print(f"  ✅ Saved correlation results")
    
    if not sentiment_aggregated.empty:
        sentiment_aggregated.to_csv(os.path.join(output_dir, 'task3_aggregated_sentiment.csv'), index=False)
        print(f"  ✅ Saved aggregated sentiment data")
    
    print(f"\n✅ All results saved to {output_dir}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("🚀 TASK 3: CORRELATION ANALYSIS - NOVA FINANCIAL SOLUTIONS")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Data paths (relative to project root)
    news_file_path = os.path.join(project_root, 'data', 'raw', 'raw_analyst_ratings.csv')
    stock_prices_path = os.path.join(project_root, 'data', 'raw', 'stock_prices')
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    # Load data
    news_df = load_news_data(news_file_path)
    stock_prices_df = load_stock_data(stock_prices_path)
    
    if news_df.empty or stock_prices_df.empty:
        print("\n❌ Error: Could not load required data files")
        return
    
    # Clean data
    news_df = clean_news_data(news_df)
    stock_prices_df = clean_stock_data(stock_prices_df)
    
    # Normalize dates
    news_df = normalize_dates(news_df)
    stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date']).dt.normalize()
    
    # Calculate sentiment
    news_df = calculate_sentiment(news_df)
    
    # Calculate returns
    stock_prices_df = calculate_stock_returns(stock_prices_df)
    
    # Aggregate sentiment
    sentiment_aggregated = aggregate_sentiment(news_df)
    
    # Merge data
    merged_df = merge_data(stock_prices_df, sentiment_aggregated)
    
    # Calculate correlations
    correlation_df = calculate_correlations(merged_df)
    
    # Save results
    save_results(merged_df, correlation_df, sentiment_aggregated, output_dir=output_dir)
    
    print("\n" + "=" * 80)
    print("✅ TASK 3 ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

