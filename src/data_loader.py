"""
data_loader.py
--------------
Handles loading and preprocessing of TCS historical stock data.
This module is responsible for reading the CSV, cleaning data types,
handling missing values, and returning a ready-to-use DataFrame.
"""

import os
import pandas as pd


# ──────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────

# Root of the project (one level above /src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HISTORICAL_CSV = os.path.join(DATA_DIR, "TCS_stock_history.csv")


# ──────────────────────────────────────────────────
# Core Loader Function
# ──────────────────────────────────────────────────

def load_tcs_data(filepath: str = HISTORICAL_CSV) -> pd.DataFrame:
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
    filepath : str
        Path to the CSV file. Defaults to data/TCS_stock_history.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with proper types and no leading NaN gaps.
    """
    # --- 1. Read CSV ---
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[data_loader] File not found: {filepath}\n"
            "Please place 'TCS_stock_history.csv' in the data/ folder, "
            "or run src/download_latest_tcs_data.py to fetch live data."
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
    df = df.fillna(method="ffill").fillna(method="bfill")

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
