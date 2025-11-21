# 🚀 Nova Financial Solutions - Stock Prediction Project

This repository hosts the initial development phases of a project aimed at predicting stock price movements by integrating **News Sentiment Analysis** (Task 1) with **Quantitative Technical Indicators** (Task 2).

The core hypothesis is that **market momentum can be explained by the emotional context of financial news.**

## 🎯 Project Status & Goal

| Metric | Status | Notes |
| :--- | :--- | :--- |
| **Task 1: EDA & Sentiment Prep** | **COMPLETE** | Full Exploratory Data Analysis (EDA) and Topic Modeling completed on news data. |
| **Task 2: Quantitative Analysis** | **COMPLETE** | Technical Indicators (RSI, MACD) and Financial Metrics (Daily Returns) calculated. |
| **Next Step** | **Task 3: Correlation & Modeling** | Integrating Task 1 and Task 2 results to build a predictive model. |

---

## 📁 Repository Structure and Organization

The repository adheres to the suggested professional data science structure, fulfilling the **Interim Repository Organization (KPI ii)** metric.

Nova_Financial_Analysis/
├── .github/                      # CI/CD workflows (unittests.yml)
├── data/                         # Raw and processed datasets
│   ├── raw/
│   │   └── stock_prices/         # 6 stock price files (.xlsx)
├── notebooks/                    # Analysis notebooks
│   ├── 01_eda_analysis.ipynb     # (Task 1) News Data Analysis
│   └── 02_quantitative_analysis.ipynb # (Task 2) Financial Data Analysis
├── requirements.txt              # Project dependencies
├── src/                          # Future source code/functions
└── tests/                        # Unit tests

---

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
4.  **Data Placement:** Place the six stock price files (e.g., `AAPL.xlsx`) into the `data/raw/stock_prices/` folder.

---

## 📈 Task Progress & Key Insights

This section summarizes the functionality achieved, demonstrating **Interim Functionality and Task Progress (KPI iv)** and **Readability and Interim Documentation (KPI iii)**.

### Task 1: Exploratory Data Analysis & Text Processing

| Sub-Task | Status | Key Findings |
| :--- | :--- | :--- |
| **Descriptive Statistics** | Completed | Confirmed dataset size (~1.4M records) and analyzed headline length distribution. |
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

---

## 💻 Version Control & Branching

The project follows a standard Git workflow, demonstrating **Interim Use of Version Control (KPI v)**:

* **`main`:** Stable, production-ready code base.
* **`Task-1`:** Branch used for initial EDA and Text Analysis (Merged into `main`).
* **`Task-2`:** Branch used for Quantitative Analysis and indicator calculation (Merged into `main`).
* Commits are frequent and descriptive (e.g., `feat: complete quantitative analysis...`).