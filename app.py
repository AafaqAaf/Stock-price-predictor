"""A lightweight educational stock-price forecasting dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from models import ModelResult, train_linear_regression, train_lstm

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 58%, #2563eb 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
    }
    .hero h1 { margin: 0; color: white; font-size: 2.2rem; }
    .hero p { margin: 0.45rem 0 0; color: #dbeafe; font-size: 1rem; }
    .eyebrow {
        color: #93c5fd;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: rgba(248, 250, 252, 0.72);
    }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(148, 163, 184, 0.22); }
    .section-note { color: #64748b; font-size: 0.9rem; margin-top: -0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(ticker: str, period: str) -> pd.DataFrame:
    """Download and tidy daily OHLCV market data."""
    try:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    except Exception as exc:
        raise ValueError(f"Could not download data for {ticker}: {exc}") from exc
    if data.empty:
        raise ValueError("No price data was returned. Check the ticker symbol and try again.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = {"Close", "Volume"}
    if not required.issubset(data.columns):
        raise ValueError("The provider returned incomplete price data for this ticker.")
    data = data.dropna(subset=["Close", "Volume"])
    if len(data) < 30:
        raise ValueError("This ticker has too few observations for analysis.")
    return data


def currency(value: float) -> str:
    return f"{value:,.2f}"


def price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Closing price", line={"color": "#2563eb"}))
    chart.add_trace(go.Scatter(x=data.index, y=data["Close"].rolling(20).mean(), name="20-day average", line={"color": "#f59e0b"}))
    chart.update_layout(
        title=f"{ticker} closing price",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=420,
        template="plotly_white",
        margin={"t": 60, "r": 20, "b": 40, "l": 60},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return chart


def evaluation_chart(result: ModelResult) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=result.actual.index, y=result.actual, name="Actual", line={"color": "#2563eb"}))
    chart.add_trace(go.Scatter(x=result.predicted.index, y=result.predicted, name="Predicted", line={"color": "#dc2626"}))
    chart.update_layout(
        title="Actual versus predicted price (unseen test data)",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=400,
        template="plotly_white",
        margin={"t": 60, "r": 20, "b": 40, "l": 60},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return chart


def forecast_chart(close: pd.Series, result: ModelResult) -> go.Figure:
    context = close.tail(60)
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=context.index, y=context, name="Historical closing price", line={"color": "#2563eb"}))
    chart.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast, name="Forecast estimate", line={"color": "#16a34a", "dash": "dash"}))
    chart.update_layout(
        title="Future price estimate",
        xaxis_title="Trading-day estimate",
        yaxis_title="Price",
        hovermode="x unified",
        height=400,
        template="plotly_white",
        margin={"t": 60, "r": 20, "b": 40, "l": 60},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return chart


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Market intelligence studio</div>
        <h1>📈 Stock Price Predictor</h1>
        <p>Explore historical price action and compare data-driven forecasts in one focused workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Forecast settings")
    st.caption("Configure the market and model, then run a forecast.")
    ticker = st.text_input("Ticker symbol", value="AAPL", help="Examples: AAPL, MSFT, RELIANCE.NS").upper().strip()
    period_label = st.selectbox("Historical period", ["6 months", "1 year", "2 years", "5 years"], index=2)
    period = {"6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y"}[period_label]
    model_name = st.selectbox("Model", ["Linear Regression", "LSTM"])
    horizon = st.selectbox("Forecast horizon", [1, 7, 30], index=1, format_func=lambda n: f"{n} trading day{'s' if n > 1 else ''}")
    st.divider()
    predict = st.button("✨ Generate forecast", type="primary", use_container_width=True)
    st.caption("Forecasts are educational estimates and are not financial advice.")

if not ticker:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()

try:
    with st.spinner("Fetching historical market data…"):
        history = load_history(ticker, period)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

close = history["Close"].astype(float)
latest, previous = close.iloc[-1], close.iloc[-2]
change = latest - previous

st.header(f"{ticker} · Market Overview")
st.markdown(
    f'<div class="section-note">Latest available data · {period_label} lookback · {len(history):,} trading sessions</div>',
    unsafe_allow_html=True,
)
cols = st.columns(4)
daily_percent = change / previous * 100
cols[0].metric("Latest close", currency(latest), "Market price")
cols[1].metric("Previous close", currency(previous), "Prior session")
cols[2].metric("Daily change", currency(change), f"{daily_percent:+.2f}%", delta_color="normal")
cols[3].metric("Latest trading date", history.index[-1].strftime("%Y-%m-%d"))

st.header("Historical Analysis")
st.markdown("Price trend, moving average, and trading activity across the selected period.", unsafe_allow_html=True)
st.plotly_chart(price_chart(history, ticker), use_container_width=True)
volume = go.Figure(go.Bar(x=history.index, y=history["Volume"], name="Volume", marker_color="#64748b"))
volume.update_layout(
    title="Daily trading volume",
    xaxis_title="Date",
    yaxis_title="Shares",
    height=300,
    template="plotly_white",
    margin={"t": 60, "r": 20, "b": 40, "l": 60},
)
st.plotly_chart(volume, use_container_width=True)

st.header("Price Prediction")
st.markdown(f"**{model_name}** · {horizon} trading day{'s' if horizon > 1 else ''} ahead", unsafe_allow_html=True)
if not predict:
    st.info("Choose a model and forecast horizon, then select **Predict**. Forecasts are estimates, not guarantees.")
else:
    try:
        progress = st.progress(0, text="Preparing model…")
        if model_name == "Linear Regression":
            result = train_linear_regression(close, horizon)
            progress.progress(100, text="Linear Regression forecast complete.")
        else:
            result = train_lstm(close, horizon, lambda value, message: progress.progress(value, text=message))
        st.session_state["prediction"] = {"ticker": ticker, "period": period, "model": model_name, "horizon": horizon, "result": result}
        progress.empty()
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))

saved = st.session_state.get("prediction")
if saved and saved["ticker"] == ticker and saved["period"] == period:
    result: ModelResult = saved["result"]
    with st.container(border=True):
        st.subheader(f"✅ {saved['model']} results")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Latest observed close", currency(latest))
        metric_cols[1].metric("Test MAE", currency(result.mae))
        metric_cols[2].metric("Test RMSE", currency(result.rmse))
        st.caption("Lower MAE and RMSE indicate smaller errors on unseen test data.")
        st.plotly_chart(evaluation_chart(result), use_container_width=True)
        st.plotly_chart(forecast_chart(close, result), use_container_width=True)
        forecast_table = result.forecast.rename("Forecasted price").to_frame()
        forecast_table.index.name = "Trading day"
        st.dataframe(forecast_table.style.format("{:.2f}"), use_container_width=True)
    st.warning("Forecasts are estimates based on historical prices. Uncertainty usually increases with the forecast horizon; this app is for education, not financial advice.")
