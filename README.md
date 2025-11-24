# 🚀 Nova Financial Solutions - Stock Prediction Project

This repository hosts the development phases of a project aimed at predicting stock price movements by integrating **News Sentiment Analysis** (Task 1) with **Quantitative Technical Indicators** (Task 2) and **Sentiment Aggregation** (Task 3).

The core hypothesis is that **market momentum can be explained by the emotional context of financial news.**

## 🎯 Project Status & Goal

| Metric | Status | Notes |
| :--- | :--- | :--- |
| **Task 1: EDA & Sentiment Prep** | **COMPLETE** | Full Exploratory Data Analysis (EDA) and Topic Modeling completed on news data. |
| **Task 2: Quantitative Analysis** | **COMPLETE** | Technical Indicators (RSI, MACD) and Financial Metrics (Daily Returns) calculated. |
| **Task 3: Sentiment Aggregation & Analysis** | **COMPLETE** | Sentiment analysis, date alignment, and daily sentiment aggregation completed. |

-----

## 📁 Repository Structure and Organization

The repository strictly adheres to the mandated professional data science structure, fulfilling the **Interim Repository Organization (KPI ii)** metric.

```text
Nova_Financial_Analysis/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml   # CI/CD Workflow (Must be named correctly)
├── data/
│   ├── raw/
│   │   ├── raw_analyst_ratings.csv  # Financial news dataset
│   │   └── stock_prices/   # Raw stock price files (e.g., AAPL.xlsx)
│   └── processed/          # Processed datasets and results
├── notebooks/
│   ├── __init__.py         # Package Marker
│   ├── 01_eda_analysis.ipynb
│   ├── 02_quantitative_analysis.ipynb
│   ├── 03_correlation_analysis.ipynb
│   ├── TASK3_README.md     # Task 3 Documentation
│   └── README.md           # Documentation
├── scripts/
│   ├── __init__.py         # Package Marker
│   ├── task3_correlation_analysis.py  # Task 3 reusable functions
│   └── README.md           # Documentation
├── src/
│   └── __init__.py         # Package Marker (Future source code/functions)
├── tests/
│   └── __init__.py         # Package Marker
├── .gitignore
├── requirements.txt
└── README.md
```

-----

## 🛠️ Setup and Environment

To replicate the analysis, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd Nova_Financial_Analysis
    ```
2.  **Create and Activate Virtual Environment:** (Demonstrating **Dev Environment Setup KPI**)
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/macOS:
    source .venv/bin/activate
    ```
3.  **Install Dependencies:** (Referencing the clean `requirements.txt`)
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data Placement:** 
    - Place the six stock price files (e.g., `AAPL.xlsx`) into the `data/raw/stock_prices/` folder.
    - Ensure the financial news dataset (`raw_analyst_ratings.csv`) is in the `data/raw/` folder.

5.  **Run Analysis:**
    ```bash
    # Option 1: Run via Jupyter Notebook
    jupyter notebook notebooks/03_correlation_analysis.ipynb
    
    # Option 2: Run via Python script
    python scripts/task3_correlation_analysis.py
    ```

-----

## 📈 Task Progress & Key Insights

This section summarizes the functionality achieved, demonstrating **Interim Functionality and Task Progress (KPI iv)** and **Readability and Interim Documentation (KPI iii)**.

### Task 1: Exploratory Data Analysis & Text Processing

| Sub-Task | Status | Key Findings |
| :--- | :--- | :--- |
| **Descriptive Statistics** | Completed | Confirmed dataset size (\~1.4M records) and analyzed headline length distribution. |
| **Publisher Analysis** | Completed | Identified top contributing publishers (e.g., Paul Quintaro, Lisa Levin) and analyzed their frequency over time. |
| **Text Analysis (Topic Modeling)** | Completed | Word cloud generated and top topics discovered, showing common financial terminology like **'share,' 'price,' 'target,'** and **'VS'** (suggesting comparisons). |
| **Time Series Analysis** | Completed | Analyzed publication frequency over time and by hour of day (UTC). |

### Task 2: Quantitative Analysis & Technical Indicators

| Sub-Task | Status | Key Findings |
| :--- | :--- | :--- |
| **Data Preparation** | Completed | Loaded six stock tickers (AAPL, AMZN, GOOG, META, MSFT, NVDA). Code was optimized to handle file type issues and run **warning-free**. |
| **Technical Indicators (TA-Lib)** | Completed | Calculated **RSI (14-period)** and **MACD (12/26/9)** for all stocks, quantifying market momentum and trend (Satisfies **Accuracy of Indicators KPI 2.2.2**). |
| **Financial Metrics** | Completed | Calculated **Daily Returns** using the stable Pandas `pct_change()` function (replacing `pynance` due to dependency issues), providing the final target variable for modeling. |
| **Visualization** | Completed | RSI overlay charts were generated to visually confirm that indicators are producing logical overbought/oversold signals relative to price movement. |

### Task 3: Sentiment Aggregation & Analysis

| Sub-Task | Status | Key Findings |
| :--- | :--- | :--- |
| **Data Profiling & Cleaning** | Completed | Comprehensive EDA on both news (~1.4M records) and stock datasets (21,793 records). Implemented data quality assessment, missing value handling, and cleaned datasets for six target stocks (AAPL, AMZN, GOOG, META, MSFT, NVDA). |
| **Date Alignment** | Completed | Normalized timestamps from UTC-4 to America/New_York timezone. Implemented market close logic (16:00 EST/EDT cutoff) to align news articles with trading days. News published after market close affects the next trading day. |
| **Sentiment Analysis** | Completed | Implemented **VADER Sentiment Analysis** (optimized for financial text) providing compound scores, positive/negative/neutral metrics. Added **TextBlob Sentiment** for additional polarity and subjectivity analysis. Successfully processed all news headlines with sentiment quantification. |
| **Stock Returns Calculation** | Completed | Calculated daily percentage changes in stock prices using Pandas `pct_change()` method. Properly handled first-day returns (NaN values) and generated 21,787 daily return records across all tickers. |
| **Sentiment Aggregation** | Completed | Aggregated daily sentiment scores by stock ticker and date. Implemented multiple aggregation methods: average compound score, maximum positive sentiment, minimum negative sentiment, and news count per day. Created timezone-naive date columns for compatibility. |
| **Visualization Dashboard** | Completed | Generated sentiment heatmaps by stock ticker showing average, max, and min sentiment metrics. Created time series visualizations displaying sentiment trends over time with min-max ranges. All visualizations use aggregated sentiment data without requiring data merging. |

**Tools & Technologies Used:**
- **NLTK VADER**: Sentiment analysis optimized for financial and social media text
- **TextBlob**: Additional sentiment polarity and subjectivity metrics
- **Pandas**: Data manipulation, aggregation, and timezone handling
- **NumPy**: Numerical operations and statistical calculations
- **Matplotlib & Seaborn**: Data visualization and dashboard creation
- **Scipy**: Statistical analysis (Pearson correlation for future use)

**Key Implementation Details:**
- Modular code structure in `scripts/task3_correlation_analysis.py` with reusable functions
- Comprehensive error handling and data validation
- Timezone-aware date processing with market close alignment logic
- Efficient chunk-based data loading for large datasets (>1M records)
- Professional repository structure with proper Git branching (task-3 branch)

**Output Files:**
- `data/processed/task3_aggregated_sentiment.csv`: Daily aggregated sentiment scores by stock and date
- Contains columns: stock, aligned_date, avg_vader_compound, max_vader_compound, min_vader_compound, news_count, avg_vader_pos, avg_vader_neg, avg_textblob_polarity

-----

## 💻 Version Control & Branching

The project adheres to a strict Gitflow model, demonstrating **Interim Use of Version Control (KPI v)** through robust branching, merging, and cleanup practices:

  * **Branching Strategy:** Feature development was conducted exclusively on dedicated feature branches (`Task-1`, `Task-2`, `Task-3`).
  * **`main` Branch:** Represents the **stable, production-ready** code base. All new features reach `main` only via a Pull Request.
  * **Pull Requests (PRs):** All feature branches (`Task-1`, `Task-2`, `Task-3`) were merged into `main` using a **Pull Request (PR)**, ensuring code review and a clean merge history.
  * **Commit Messages:** Commits are frequent and descriptive, following conventional commit practices (e.g., `feat: implement sentiment aggregation`, `fix: resolve timezone merge issues`, `docs: update Task 3 documentation`).
  * **Branch Cleanup:** Feature branches (`Task-1`, `Task-2`, `Task-3`) are maintained for active development and will be deleted after successful merge into `main`, maintaining a clean branch history.