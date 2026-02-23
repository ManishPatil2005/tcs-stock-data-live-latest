"""
eda.py
------
Exploratory Data Analysis (EDA) functions for TCS stock data.
Each function creates a specific visualization or computes an insight.
Designed to be called from the notebook or run standalone.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# ──────────────────────────────────────────────────
# Global Plot Style
# ──────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-darkgrid")
FIGURE_SIZE = (14, 5)
COLOR_CLOSE = "#2196F3"   # blue  – closing price
COLOR_VOL   = "#4CAF50"   # green – volume
COLOR_MA30  = "#FF9800"   # orange – 30-day MA
COLOR_MA5   = "#E91E63"   # pink  – 5-day (short) MA
BUY_COLOR   = "#00C853"   # bright green – buy signal
SELL_COLOR  = "#D50000"   # bright red   – sell signal


# ──────────────────────────────────────────────────
# 1. Close Price Over Time
# ──────────────────────────────────────────────────

def plot_close_price(df: pd.DataFrame) -> None:
    """
    Plot the historical closing price of TCS as a time-series line chart.

    Parameters
    ----------
    df : pd.DataFrame  — must contain 'Date' and 'Close' columns.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(df["Date"], df["Close"], color=COLOR_CLOSE, linewidth=1.2, label="Close Price")

    ax.set_title("TCS Stock – Closing Price Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price (INR)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.legend()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────
# 2. Volume Over Time
# ──────────────────────────────────────────────────

def plot_volume(df: pd.DataFrame) -> None:
    """
    Plot trading Volume as a bar chart over time.

    Parameters
    ----------
    df : pd.DataFrame  — must contain 'Date' and 'Volume' columns.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(df["Date"], df["Volume"], color=COLOR_VOL, alpha=0.7, width=1.5, label="Volume")

    ax.set_title("TCS Stock – Trading Volume Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volume (Shares Traded)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.legend()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────
# 3. Dividends & Stock Splits Over Time
# ──────────────────────────────────────────────────

def plot_dividends_and_splits(df: pd.DataFrame) -> None:
    """
    Plot Dividends and Stock Splits on a dual-panel chart.
    These columns are often mostly zero, so we highlight non-zero events.

    Parameters
    ----------
    df : pd.DataFrame  — must contain 'Date', 'Dividends', 'Stock Splits' columns.
    """
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_SIZE[0], 7), sharex=True)

    # -- Dividends panel --
    axes[0].bar(df["Date"], df["Dividends"], color="#9C27B0", alpha=0.8, width=2, label="Dividends")
    axes[0].set_title("TCS – Dividends Over Time", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Dividend Amount (INR)", fontsize=11)
    axes[0].legend()

    # -- Stock Splits panel --
    axes[1].bar(df["Date"], df["Stock Splits"], color="#FF5722", alpha=0.8, width=2, label="Stock Splits")
    axes[1].set_title("TCS – Stock Splits Over Time", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].set_ylabel("Split Ratio", fontsize=11)
    axes[1].legend()

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────
# 4. Correlation Heatmap
# ──────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Compute and visualize a Pearson correlation heatmap for all numeric columns.
    This helps understand linear relationships between price features.

    Parameters
    ----------
    df : pd.DataFrame  — numeric columns of interest: Open, High, Low, Close, Volume, etc.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   # hide upper triangle (duplicate)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("TCS Stock – Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────
# 5. Scatter Plot: Close vs Volume
# ──────────────────────────────────────────────────

def plot_close_vs_volume(df: pd.DataFrame) -> None:
    """
    Scatter plot of closing price versus trading volume.
    Useful to see if high-volume days coincide with extreme prices.

    Parameters
    ----------
    df : pd.DataFrame  — must contain 'Close' and 'Volume' columns.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Volume"], df["Close"], alpha=0.3, s=10, color=COLOR_CLOSE, edgecolors="none")

    ax.set_title("TCS Stock – Close Price vs. Volume", fontsize=14, fontweight="bold")
    ax.set_xlabel("Volume (Shares Traded)", fontsize=12)
    ax.set_ylabel("Close Price (INR)", fontsize=12)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────
# 6. Moving Averages
# ──────────────────────────────────────────────────

def plot_moving_averages(df: pd.DataFrame, short_window: int = 5, long_window: int = 30) -> None:
    """
    Plot the closing price alongside short-term and long-term moving averages.
    Also marks buy (short MA crosses above long MA) and sell (short MA crosses below)
    signals using vertical green/red dots.

    Parameters
    ----------
    df           : pd.DataFrame — must contain 'Date' and 'Close' columns.
    short_window : int          — rolling window for short MA  (default 5 days).
    long_window  : int          — rolling window for long MA   (default 30 days).
    """
    df = df.copy()

    # Compute moving averages
    df[f"MA_{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"MA_{long_window}"]  = df["Close"].rolling(window=long_window).mean()

    # Generate buy/sell signals
    # Signal = 1 → short MA > long MA (uptrend), Signal = 0 → short MA < long MA (downtrend)
    df["Signal"] = 0
    df.loc[df[f"MA_{short_window}"] > df[f"MA_{long_window}"], "Signal"] = 1

    # Position = change in signal (1 = buy event, -1 = sell event)
    df["Position"] = df["Signal"].diff()

    buy_dates  = df[df["Position"] ==  1]["Date"]
    buy_prices = df[df["Position"] ==  1]["Close"]
    sell_dates = df[df["Position"] == -1]["Date"]
    sell_prices= df[df["Position"] == -1]["Close"]

    # Plot
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE[0], 6))
    ax.plot(df["Date"], df["Close"],            color=COLOR_CLOSE, linewidth=1.0, alpha=0.8, label="Close Price")
    ax.plot(df["Date"], df[f"MA_{short_window}"], color=COLOR_MA5,  linewidth=1.5, linestyle="--", label=f"{short_window}-Day MA")
    ax.plot(df["Date"], df[f"MA_{long_window}"],  color=COLOR_MA30, linewidth=1.5, linestyle="--", label=f"{long_window}-Day MA")

    # Buy/Sell markers
    ax.scatter(buy_dates,  buy_prices,  marker="^", color=BUY_COLOR,  s=80, zorder=5, label="Buy Signal")
    ax.scatter(sell_dates, sell_prices, marker="v", color=SELL_COLOR, s=80, zorder=5, label="Sell Signal")

    ax.set_title(
        f"TCS – Moving Average Crossover ({short_window}-Day vs {long_window}-Day)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (INR)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"[eda] Total Buy  signals: {len(buy_dates)}")
    print(f"[eda] Total Sell signals: {len(sell_dates)}")

    # Clear end-of-analysis guidance for the latest day
    latest_row = df.iloc[-1]
    latest_signal = int(latest_row["Signal"])
    latest_position = latest_row["Position"]

    if latest_signal == 1:
        print("[eda] Current trend bias: BUY (short MA is above long MA)")
    else:
        print("[eda] Current trend bias: SELL (short MA is below long MA)")

    if pd.isna(latest_position) or latest_position == 0:
        print("[eda] Latest event: No new crossover today → HOLD previous bias")
    elif latest_position == 1:
        print("[eda] Latest event: NEW BUY crossover today")
    elif latest_position == -1:
        print("[eda] Latest event: NEW SELL crossover today")


# ──────────────────────────────────────────────────
# 7. Price Distribution (Histogram)
# ──────────────────────────────────────────────────

def plot_price_distribution(df: pd.DataFrame) -> None:
    """
    Histogram of daily Close prices to understand the price distribution shape.

    Parameters
    ----------
    df : pd.DataFrame  — must contain 'Close' column.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Close"], bins=50, color=COLOR_CLOSE, edgecolor="white", alpha=0.85)
    ax.axvline(df["Close"].mean(),  color="orange", linewidth=2, linestyle="--", label=f"Mean  ≈ {df['Close'].mean():.0f}")
    ax.axvline(df["Close"].median(),color="red",    linewidth=2, linestyle=":",  label=f"Median ≈ {df['Close'].median():.0f}")

    ax.set_title("TCS Stock – Close Price Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Close Price (INR)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────
# Module entry-point
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_tcs_data

    df = load_tcs_data()
    plot_close_price(df)
    plot_volume(df)
    plot_dividends_and_splits(df)
    plot_correlation_heatmap(df)
    plot_close_vs_volume(df)
    plot_moving_averages(df)
    plot_price_distribution(df)
