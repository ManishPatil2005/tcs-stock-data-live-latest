"""
data_loader.py
--------------
Handles loading and preprocessing of TCS historical stock data.
This module is responsible for reading the CSV, cleaning data types,
handling missing values, and returning a ready-to-use DataFrame.
"""

import os
import pandas as pd
from typing import Optional

try:
    from src.download_latest_tcs_data import download_tcs_data, save_to_csv
except ImportError:
    from download_latest_tcs_data import download_tcs_data, save_to_csv


# ──────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────

# Root of the project (one level above /src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HISTORICAL_CSV = os.path.join(DATA_DIR, "TCS_stock_history.csv")
LATEST_CSV = os.path.join(DATA_DIR, "tcs_stock_latest.csv")


def ensure_latest_data(
    filepath: str = LATEST_CSV,
    period: str = "5y",
    interval: str = "1d",
) -> str:
    """
    Ensure a local latest CSV exists. If not, download from Yahoo Finance.

    Returns
    -------
    str
        Path to the CSV that should be used for loading.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(filepath):
        return filepath

    print("[data_loader] Latest CSV not found. Downloading online data...")
    downloaded_df = download_tcs_data(period=period, interval=interval)
    save_to_csv(downloaded_df, filepath=filepath)
    return filepath


# ──────────────────────────────────────────────────
# Core Loader Function
# ──────────────────────────────────────────────────

def load_tcs_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load TCS stock history from a CSV file and return a clean DataFrame.

    Steps performed:
    1. Read the CSV file into a DataFrame.
    2. Convert the 'Date' column to datetime format.
    3. Sort records chronologically by Date.
    4. Cast all OHLCV and other numeric columns to float64.
    5. Forward-fill any remaining NaN values (common at edges of time series).
    6. Reset the index so it starts from 0.

    Parameters
    ----------
    filepath : Optional[str]
        Path to a CSV file. If None, loader prefers data/tcs_stock_latest.csv.
        If that file is missing, it auto-downloads data from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with proper types and no leading NaN gaps.
    """
    # --- 1. Resolve CSV source (prefer latest CSV, fallback to historical CSV) ---
    if filepath is None:
        if os.path.exists(LATEST_CSV):
            filepath = LATEST_CSV
        elif os.path.exists(HISTORICAL_CSV):
            filepath = HISTORICAL_CSV
        else:
            filepath = ensure_latest_data()

    if not os.path.exists(filepath):
        if filepath == LATEST_CSV:
            filepath = ensure_latest_data(filepath=LATEST_CSV)
        else:
            raise FileNotFoundError(
                f"[data_loader] File not found: {filepath}\n"
                "Provide a valid CSV path or allow auto-download by using load_tcs_data() with no filepath."
            )

    df = pd.read_csv(filepath)
    print(f"[data_loader] Loaded {len(df)} rows from '{filepath}'")

    # --- 2. Parse and convert Date column ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop any rows where Date parsing failed entirely
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        print(f"[data_loader] WARNING: Dropping {invalid_dates} rows with unparseable dates.")
        df = df.dropna(subset=["Date"])

    # --- 3. Sort chronologically ---
    df = df.sort_values("Date").reset_index(drop=True)

    # --- 4. Ensure numeric columns have correct data types ---
    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 5. Handle missing values using forward fill then backward fill ---
    df = df.ffill().bfill()

    # --- 6. Final reset of index ---
    df = df.reset_index(drop=True)

    print(f"[data_loader] Data range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"[data_loader] Shape after cleaning: {df.shape}")
    return df


# ──────────────────────────────────────────────────
# Quick Info Utility
# ──────────────────────────────────────────────────

def display_basic_info(df: pd.DataFrame) -> None:
    """
    Print a quick overview of the DataFrame:
    - First 5 rows
    - Data types and non-null counts (.info())
    - Summary statistics (.describe())

    Parameters
    ----------
    df : pd.DataFrame
        The loaded TCS stock DataFrame.
    """
    print("\n" + "=" * 55)
    print("  BASIC DATA OVERVIEW")
    print("=" * 55)

    print("\n[HEAD — First 5 rows]")
    print(df.head())

    print("\n[INFO — Column types and null counts]")
    df.info()

    print("\n[DESCRIBE — Summary statistics]")
    print(df.describe().round(2))


# ──────────────────────────────────────────────────
# Module entry-point (for quick testing)
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    tcs_df = load_tcs_data()
    display_basic_info(tcs_df)
