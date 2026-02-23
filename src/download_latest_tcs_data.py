"""
download_latest_tcs_data.py
----------------------------
Fetches the most recent TCS (Tata Consultancy Services) stock data
from Yahoo Finance using the `yfinance` library and saves it as a CSV.

Ticker : TCS.NS  (TCS listed on NSE – National Stock Exchange of India)
Output : data/tcs_stock_latest.csv

Usage:
    python src/download_latest_tcs_data.py

Requirements:
    pip install yfinance
"""

import os
import sys
import pandas as pd

# ── Resolve the project root path ──────────────────────────────────────────────
# This script lives in src/, so the project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE  = os.path.join(DATA_DIR, "tcs_stock_latest.csv")


def download_tcs_data(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download TCS stock data from Yahoo Finance and return it as a DataFrame.

    Parameters
    ----------
    period   : str — how far back to fetch data.
                     Options: '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'
                     Default : '5y' (last 5 years, good balance of size vs richness)
    interval : str — candle interval. '1d' = daily bars (recommended for this project).

    Returns
    -------
    pd.DataFrame with columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[downloader] ERROR: yfinance is not installed.")
        print("             Please run: pip install yfinance")
        sys.exit(1)

    ticker_symbol = "TCS.NS"
    print(f"[downloader] Fetching {ticker_symbol} data from Yahoo Finance ...")
    print(f"             Period   : {period}")
    print(f"             Interval : {interval}")

    # Create a Ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Download OHLCV + Actions (dividends & splits) as one DataFrame
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        print("[downloader] ERROR: No data returned. Please check your internet connection.")
        sys.exit(1)

    # Reset index so 'Date' becomes a regular column (not the DataFrame index)
    df = df.reset_index()

    # Ensure the Date column is timezone-naive (remove tz info if present)
    # This avoids issues when saving to CSV and reloading.
    if hasattr(df["Date"].dtype, "tz") and df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    # Rename columns consistently (yfinance occasionally returns 'Datetime' instead of 'Date')
    df = df.rename(columns={"Datetime": "Date"})

    # Reorder columns to match TCS_stock_history.csv structure
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols]

    print(f"[downloader] Downloaded {len(df)} rows of data.")
    print(f"[downloader] Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    return df


def save_to_csv(df: pd.DataFrame, filepath: str = OUTPUT_FILE) -> None:
    """
    Save the downloaded DataFrame to a CSV file.

    Parameters
    ----------
    df       : pd.DataFrame — the data to save.
    filepath : str          — output path (default: data/tcs_stock_latest.csv).
    """
    # Make sure the data/ directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"\n[downloader] ✓ Data saved to: {filepath}")


def main():
    """
    Main entry-point: download TCS stock data and save it to CSV.
    """
    print("\n" + "=" * 60)
    print("  TCS STOCK DATA DOWNLOADER")
    print("  Ticker: TCS.NS  |  Exchange: NSE India (via Yahoo Finance)")
    print("=" * 60)

    # Download data
    df = download_tcs_data(period="5y", interval="1d")

    # Preview a few rows in the terminal
    print("\n[downloader] Sample rows (last 5):")
    print(df.tail().to_string(index=False))

    # Save to CSV
    save_to_csv(df)

    print("\n[downloader] Done! You can now use 'data/tcs_stock_latest.csv' as your dataset.")
    print("             Rename it to 'TCS_stock_history.csv' to use with the notebooks.")


# Run when executed directly
if __name__ == "__main__":
    main()
