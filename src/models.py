"""
models.py
---------
Machine Learning models for TCS Stock Close Price Prediction.

Contains:
  A) Linear Regression  — baseline scikit-learn model
  B) Random Forest      — ensemble tree model (improved baseline)
  C) LSTM               — deep learning time-series model (TensorFlow/Keras)

Each section has clear function-level comments so an intern
can understand and explain the code.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")


# ══════════════════════════════════════════════════
# SECTION A — LINEAR REGRESSION (BASELINE)
# ══════════════════════════════════════════════════

# Features used: we rely on OHLCV data + engineered lag & date features.
LR_FEATURES = [
    "Open", "High", "Low", "Volume",
    "Prev_Close_1", "Prev_Close_3", "Prev_Close_7",
    "MA_7", "MA_30",
    "Month", "Day_of_Week",
    "Price_Range", "Daily_Return",
]
TARGET = "Close"


def prepare_ml_data(df: pd.DataFrame, features: list = None, target: str = TARGET):
    """
    Split the feature DataFrame into X (features) and y (target), then
    perform a chronological 80/20 train-test split (no shuffling — time matters).

    Parameters
    ----------
    df       : pd.DataFrame — feature-enriched DataFrame from features.build_features().
    features : list         — list of column names to use as X (defaults to LR_FEATURES).
    target   : str          — column to predict (default 'Close').

    Returns
    -------
    Tuple (X_train, X_test, y_train, y_test, dates_test)
    """
    if features is None:
        features = LR_FEATURES

    # Only keep columns that actually exist in df
    available_features = [f for f in features if f in df.columns]
    missing = set(features) - set(available_features)
    if missing:
        print(f"[models] WARNING: Missing feature columns skipped: {missing}")

    X = df[available_features].values
    y = df[target].values
    dates = df["Date"].values

    # Chronological split — keep the last 20% as the test set
    split_idx = int(len(X) * 0.80)
    X_train, X_test   = X[:split_idx], X[split_idx:]
    y_train, y_test   = y[:split_idx], y[split_idx:]
    dates_test        = dates[split_idx:]

    print(f"[models] Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test, dates_test, available_features


def run_linear_regression(df: pd.DataFrame) -> LinearRegression:
    """
    Train a Linear Regression model to predict TCS closing price.

    Steps:
    1. Prepare X / y with chronological 80/20 split.
    2. Fit sklearn LinearRegression on training data.
    3. Predict on the test set.
    4. Print MSE, RMSE, MAE, and R² metrics.
    5. Plot Actual vs Predicted closing prices.

    Parameters
    ----------
    df : pd.DataFrame — output of features.build_features().

    Returns
    -------
    Fitted LinearRegression model.
    """
    print("\n" + "=" * 55)
    print("  MODEL A — LINEAR REGRESSION")
    print("=" * 55)

    X_train, X_test, y_train, y_test, dates_test, used_features = prepare_ml_data(df)

    # --- Train ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Predict ---
    y_pred = model.predict(X_test)

    # --- Metrics ---
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.4f}  (1.0 = perfect)")

    # --- Plot: Actual vs Predicted over time ---
    dates_dt = pd.to_datetime(dates_test)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_dt, y_test, color="#2196F3", linewidth=1.2, label="Actual Close")
    ax.plot(dates_dt, y_pred, color="#FF5722", linewidth=1.2, linestyle="--", label="Predicted Close (LR)")
    ax.set_title("Linear Regression – Actual vs Predicted Close Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- Scatter: Actual vs Predicted ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_test, y_pred, alpha=0.4, s=12, color="#9C27B0")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit")
    ax2.set_title("Linear Regression – Actual vs Predicted (Scatter)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Actual Close")
    ax2.set_ylabel("Predicted Close")
    ax2.legend()
    plt.tight_layout()
    plt.show()

    return model


# ══════════════════════════════════════════════════
# SECTION B — RANDOM FOREST REGRESSOR
# ══════════════════════════════════════════════════

def run_random_forest(df: pd.DataFrame, n_estimators: int = 100) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor — a more powerful ensemble model.

    Random Forest builds many decision trees and averages their predictions,
    which typically outperforms a single linear model on non-linear stock data.

    Parameters
    ----------
    df            : pd.DataFrame — output of features.build_features().
    n_estimators  : int          — number of trees (default 100).

    Returns
    -------
    Fitted RandomForestRegressor model.
    """
    print("\n" + "=" * 55)
    print("  MODEL B — RANDOM FOREST REGRESSOR")
    print("=" * 55)

    X_train, X_test, y_train, y_test, dates_test, used_features = prepare_ml_data(df)

    # --- Train ---
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Predict ---
    y_pred = model.predict(X_test)

    # --- Metrics ---
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.4f}  (1.0 = perfect)")

    # --- Feature Importance Plot ---
    importances = pd.Series(model.feature_importances_, index=used_features)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#4CAF50", edgecolor="white")
    ax.set_title("Random Forest – Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    # --- Plot: Actual vs Predicted ---
    dates_dt = pd.to_datetime(dates_test)
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(dates_dt, y_test, color="#2196F3", linewidth=1.2, label="Actual Close")
    ax2.plot(dates_dt, y_pred, color="#4CAF50", linewidth=1.2, linestyle="--", label="Predicted Close (RF)")
    ax2.set_title("Random Forest – Actual vs Predicted Close Price", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (INR)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig2.autofmt_xdate()
    ax2.legend()
    plt.tight_layout()
    plt.show()

    return model


# ══════════════════════════════════════════════════
# SECTION C — LSTM (DEEP LEARNING / TIME SERIES)
# ══════════════════════════════════════════════════

def run_lstm(df: pd.DataFrame, lookback: int = 60, epochs: int = 20, batch_size: int = 32):
    """
    Build and train a simple LSTM (Long Short-Term Memory) neural network
    to predict the next day's closing price from the past `lookback` days.

    LSTM is a type of Recurrent Neural Network (RNN) that is well-suited
    for sequential / time-series data like stock prices.

    Steps:
    1. Extract and normalise the 'Close' price series using MinMaxScaler.
    2. Build sequences: each input is the last `lookback` days of Close prices.
    3. Define a 2-layer LSTM model in Keras.
    4. Train for `epochs` epochs.
    5. Inverse-transform predictions and plot Actual vs Predicted.

    Parameters
    ----------
    df         : pd.DataFrame — clean DataFrame with 'Date' and 'Close' columns.
    lookback   : int          — number of past days used as input sequence (default 60).
    epochs     : int          — training epochs (default 20).
    batch_size : int          — mini-batch size (default 32).

    Returns
    -------
    Tuple (keras model, scaler) or None if TensorFlow is not installed.
    """
    print("\n" + "=" * 55)
    print("  MODEL C — LSTM (DEEP LEARNING)")
    print("=" * 55)

    # --- Optional Import ---
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        print(f"  TensorFlow version: {tf.__version__}")
    except ImportError:
        print("  [LSTM] TensorFlow not installed. Run: pip install tensorflow")
        print("  Skipping LSTM model.")
        return None

    # --- Prepare Data ---
    close_series = df["Close"].values.reshape(-1, 1)

    # Scale prices to range [0, 1] — LSTM trains better on normalised data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_series)

    # Build (X, y) sequences:
    # X[i] = scaled prices from day i to i+lookback-1
    # y[i] = scaled price on day i+lookback  (what we want to predict)
    def create_sequences(data, window):
        X_seq, y_seq = [], []
        for i in range(window, len(data)):
            X_seq.append(data[i - window : i, 0])
            y_seq.append(data[i, 0])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(scaled, lookback)

    # Reshape X to (samples, timesteps, features=1) — required by LSTM
    X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)

    # Chronological split — 80% train, 20% test
    split = int(len(X_seq) * 0.80)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"  LSTM Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")

    # --- Build LSTM Model ---
    model = Sequential([
        # First LSTM layer: 64 units, returns sequences so next LSTM can consume them
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),   # Dropout prevents overfitting by randomly zeroing 20% of neurons

        # Second LSTM layer: 32 units, only returns final output (not sequences)
        LSTM(32, return_sequences=False),
        Dropout(0.2),

        # Output layer: single neuron predicts next day's normalised price
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    # --- Train ---
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # --- Plot Training Loss ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title("LSTM – Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- Predict and Inverse Transform ---
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Align dates: test predictions map to days from (split + lookback) onward
    test_dates = df["Date"].values[split + lookback:]

    # --- Evaluation Metrics ---
    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    print(f"\n  [LSTM] MAE  : {mae:.2f}")
    print(f"  [LSTM] RMSE : {rmse:.2f}")

    # --- Plot: Actual vs Predicted ---
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(pd.to_datetime(test_dates), y_actual, color="#2196F3", linewidth=1.2, label="Actual Close")
    ax2.plot(pd.to_datetime(test_dates), y_pred,   color="#E91E63", linewidth=1.2, linestyle="--", label="LSTM Predicted")
    ax2.set_title("LSTM – Actual vs Predicted Close Price", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (INR)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig2.autofmt_xdate()
    ax2.legend()
    plt.tight_layout()
    plt.show()

    return model, scaler


# ══════════════════════════════════════════════════
# Module entry-point  (run all models)
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_tcs_data
    from src.features    import build_features

    raw_df     = load_tcs_data()
    feature_df = build_features(raw_df)

    run_linear_regression(feature_df)
    run_random_forest(feature_df)
    run_lstm(raw_df)         # LSTM uses raw Close series (not engineered features)
