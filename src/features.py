"""
features.py
-----------
Feature Engineering for TCS Stock Price Prediction.

This module takes the cleaned raw DataFrame and adds:
  - Calendar-based features (Year, Month, Day, Day_of_Week)
  - Lag features (previous day's closing price)
  - Rolling statistical features (7-day and 30-day moving averages)
  - Rolling volatility (standard deviation of Close over 7 days)

Returns a DataFrame ready to be fed into ML models.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────
# 1. Date / Calendar Features
# ──────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar components from the 'Date' column.

    New columns added:
      - Year        : calendar year  (e.g. 2023)
      - Month       : calendar month (1–12)
      - Day         : day of the month (1–31)
      - Day_of_Week : weekday number (0=Monday, 4=Friday, stock markets closed on weekends)

    Parameters
    ----------
    df : pd.DataFrame  — must contain a 'Date' column of datetime type.

    Returns
    -------
    pd.DataFrame with four new date-based columns appended.
    """
    df = df.copy()
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Day"]         = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.dayofweek   # 0 = Monday, 4 = Friday
    return df


# ──────────────────────────────────────────────────
# 2. Lag Features
# ──────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """
    Add lag features (previous N days' closing prices).

    Lag features let models use yesterday's price as a predictor for today's target.
    This is critical because stock prices follow autocorrelation (today is related to yesterday).

    New columns added:
      - Prev_Close_1  : Close price from 1 trading day ago
      - Prev_Close_3  : Close price from 3 trading days ago  (if lag_days >= 3)
      - Prev_Close_7  : Close price from 7 trading days ago  (if lag_days >= 7)

    Parameters
    ----------
    df       : pd.DataFrame — must contain 'Close' column, sorted chronologically.
    lag_days : int          — maximum lag window in days (default 1, minimum).

    Returns
    -------
    pd.DataFrame with lag columns appended.
    """
    df = df.copy()
    lags_to_create = [1, 3, 7]  # Create standard lags used in time-series ML

    for lag in lags_to_create:
        if lag <= lag_days or lag == 1:   # always create at least lag-1
            df[f"Prev_Close_{lag}"] = df["Close"].shift(lag)

    return df


# ──────────────────────────────────────────────────
# 3. Rolling / Moving Average Features
# ──────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window statistics over the 'Close' price column.

    New columns added:
      - MA_7    : 7-day rolling mean  (short-term trend)
      - MA_30   : 30-day rolling mean (medium-term trend)
      - STD_7   : 7-day rolling standard deviation (short-term volatility)
      - STD_30  : 30-day rolling standard deviation (medium-term volatility)

    Rolling statistics give the model awareness of recent price trends and volatility,
    which is more informative than raw prices alone.

    Parameters
    ----------
    df : pd.DataFrame  — must contain 'Close' column, sorted chronologically.

    Returns
    -------
    pd.DataFrame with rolling feature columns appended.
    """
    df = df.copy()

    # Moving averages
    df["MA_7"]  = df["Close"].rolling(window=7,  min_periods=1).mean()
    df["MA_30"] = df["Close"].rolling(window=30, min_periods=1).mean()

    # Rolling volatility (standard deviation)
    df["STD_7"]  = df["Close"].rolling(window=7,  min_periods=1).std()
    df["STD_30"] = df["Close"].rolling(window=30, min_periods=1).std()

    return df


# ──────────────────────────────────────────────────
# 4. Price Delta Feature
# ──────────────────────────────────────────────────

def add_price_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a daily price change feature.

    New columns added:
      - Daily_Return   : percentage daily return  = (Close - Prev_Close_1) / Prev_Close_1 * 100
      - Price_Range    : daily range = High - Low (intra-day volatility indicator)

    Parameters
    ----------
    df : pd.DataFrame — must contain 'Close', 'High', 'Low', and 'Prev_Close_1' columns.

    Returns
    -------
    pd.DataFrame with two new feature columns.
    """
    df = df.copy()

    # Guard: ensure Prev_Close_1 exists
    if "Prev_Close_1" not in df.columns:
        df["Prev_Close_1"] = df["Close"].shift(1)

    df["Daily_Return"] = ((df["Close"] - df["Prev_Close_1"]) / df["Prev_Close_1"]) * 100
    df["Price_Range"]  = df["High"] - df["Low"]

    return df


# ──────────────────────────────────────────────────
# 5. Master Pipeline — Build Feature DataFrame
# ──────────────────────────────────────────────────

def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Run all feature engineering steps in the correct order and return
    a complete feature-enriched DataFrame ready for ML modeling.

    Pipeline:
        1. add_date_features       → Year, Month, Day, Day_of_Week
        2. add_lag_features        → Prev_Close_1, Prev_Close_3, Prev_Close_7
        3. add_rolling_features    → MA_7, MA_30, STD_7, STD_30
        4. add_price_delta         → Daily_Return, Price_Range
        5. Drop rows with NaN      → caused by lag/rolling windows at start of series

    Parameters
    ----------
    df       : pd.DataFrame — clean raw DataFrame from data_loader.load_tcs_data().
    drop_na  : bool         — if True, drop rows with any NaN (default True).

    Returns
    -------
    pd.DataFrame  ready for ML feature/target split.
    """
    df = add_date_features(df)
    df = add_lag_features(df, lag_days=7)
    df = add_rolling_features(df)
    df = add_price_delta(df)

    if drop_na:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = before - len(df)
        print(f"[features] Dropped {dropped} rows with NaN (lag/rolling warm-up period).")

    print(f"[features] Final feature DataFrame shape: {df.shape}")
    print(f"[features] Columns: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────
# Module entry-point
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_tcs_data

    raw_df = load_tcs_data()
    feature_df = build_features(raw_df)
    print(feature_df.tail())
