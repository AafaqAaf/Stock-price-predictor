"""Data preparation and forecasting helpers for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


LOOKBACK = 60
LAG_COLUMNS = ["lag_1", "lag_2", "lag_3", "lag_5", "ma_5", "ma_20"]


@dataclass
class ModelResult:
    """One fitted model, its unseen-test results, and a recursive forecast."""

    actual: pd.Series
    predicted: pd.Series
    forecast: pd.Series
    mae: float
    rmse: float


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_absolute_error(actual, predicted)),
        float(np.sqrt(mean_squared_error(actual, predicted))),
    )


def _forecast_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Return weekday estimates for upcoming trading sessions."""
    return pd.bdate_range(last_date + pd.offsets.BDay(1), periods=horizon)


def _linear_features(close: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"target": close})
    for lag in (1, 2, 3, 5):
        frame[f"lag_{lag}"] = close.shift(lag)
    frame["ma_5"] = close.shift(1).rolling(5).mean()
    frame["ma_20"] = close.shift(1).rolling(20).mean()
    return frame.dropna()


def train_linear_regression(close: pd.Series, horizon: int) -> ModelResult:
    """Fit a lag-based linear baseline with a chronological holdout set."""
    data = _linear_features(close.astype(float))
    if len(data) < 80:
        raise ValueError("Linear Regression needs at least 80 usable trading days.")

    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    model = LinearRegression().fit(train[LAG_COLUMNS], train["target"])
    test_predictions = model.predict(test[LAG_COLUMNS])
    mae, rmse = _metrics(test["target"].to_numpy(), test_predictions)

    history = list(close.astype(float).to_numpy())
    future_prices: list[float] = []
    for _ in range(horizon):
        values = pd.Series(history)
        row = pd.DataFrame(
            [[
                values.iloc[-1], values.iloc[-2], values.iloc[-3], values.iloc[-5],
                values.iloc[-5:].mean(), values.iloc[-20:].mean(),
            ]],
            columns=LAG_COLUMNS,
        )
        next_price = float(model.predict(row)[0])
        future_prices.append(next_price)
        history.append(next_price)

    return ModelResult(
        actual=test["target"],
        predicted=pd.Series(test_predictions, index=test.index, name="Predicted"),
        forecast=pd.Series(future_prices, index=_forecast_dates(close.index[-1], horizon), name="Forecast"),
        mae=mae,
        rmse=rmse,
    )


def _lstm_sequences(values: np.ndarray, indexes: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    inputs, targets, target_indexes = [], [], []
    for end in range(LOOKBACK, len(values)):
        inputs.append(values[end - LOOKBACK:end])
        targets.append(values[end])
        target_indexes.append(indexes[end])
    return np.array(inputs), np.array(targets), pd.DatetimeIndex(target_indexes)


def train_lstm(
    close: pd.Series,
    horizon: int,
    progress_callback: Callable[[int, str], None] | None = None,
) -> ModelResult:
    """Fit the compact LSTM without fitting a scaler on holdout observations."""
    if len(close) < 160:
        raise ValueError("LSTM needs at least 160 trading days. Select a longer history period.")

    # TensorFlow is imported here so the overview still loads if it is unavailable.
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise RuntimeError("TensorFlow is not installed. Install requirements.txt and restart Streamlit.") from exc

    prices = close.astype(float)
    split = int(len(prices) * 0.8)
    train_prices, test_prices = prices.iloc[:split], prices.iloc[split:]
    if len(test_prices) < 10:
        raise ValueError("Not enough unseen observations for LSTM evaluation.")

    scaler = MinMaxScaler()
    scaler.fit(train_prices.to_numpy().reshape(-1, 1))
    scaled_all = scaler.transform(prices.to_numpy().reshape(-1, 1)).ravel()
    sequences, targets, target_dates = _lstm_sequences(scaled_all, prices.index)
    train_mask = target_dates < test_prices.index[0]
    test_mask = ~train_mask
    x_train, y_train = sequences[train_mask], targets[train_mask]
    x_test, y_test = sequences[test_mask], targets[test_mask]
    if len(x_train) < 20 or len(x_test) == 0:
        raise ValueError("Not enough observations to create LSTM train and test sequences.")

    model = Sequential([LSTM(50, input_shape=(LOOKBACK, 1)), Dropout(0.2), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    if progress_callback:
        progress_callback(5, "Training the LSTM…")

    class StreamlitProgress(EarlyStopping):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(min(95, int((epoch + 1) / 20 * 90) + 5), f"Training LSTM: epoch {epoch + 1}/20")

    model.fit(
        x_train.reshape((-1, LOOKBACK, 1)), y_train,
        validation_split=0.15,
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), StreamlitProgress()],
    )
    scaled_predictions = model.predict(x_test.reshape((-1, LOOKBACK, 1)), verbose=0).ravel()
    predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mae, rmse = _metrics(actual, predictions)

    rolling_window = scaled_all[-LOOKBACK:].copy()
    future_prices: list[float] = []
    for _ in range(horizon):
        next_scaled = float(model.predict(rolling_window.reshape(1, LOOKBACK, 1), verbose=0)[0, 0])
        future_prices.append(float(scaler.inverse_transform([[next_scaled]])[0, 0]))
        rolling_window = np.append(rolling_window[1:], next_scaled)
    if progress_callback:
        progress_callback(100, "LSTM forecast complete.")

    return ModelResult(
        actual=pd.Series(actual, index=target_dates[test_mask], name="Actual"),
        predicted=pd.Series(predictions, index=target_dates[test_mask], name="Predicted"),
        forecast=pd.Series(future_prices, index=_forecast_dates(prices.index[-1], horizon), name="Forecast"),
        mae=mae,
        rmse=rmse,
    )
