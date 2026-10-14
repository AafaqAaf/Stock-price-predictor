# 📈 Stock Price Predictor

A lightweight, interactive stock-market dashboard built with **Streamlit**. It retrieves historical daily prices from Yahoo Finance, visualizes market trends, and forecasts future closing prices using **Linear Regression** or an **LSTM neural network**.

> **Disclaimer:** This project is for educational purposes only. Forecasts are estimates based on historical prices and are not financial advice or investment recommendations.

## Features

- **Flexible ticker input** — analyze symbols such as `AAPL`, `MSFT`, or `RELIANCE.NS`.
- **Historical data ranges** — choose 6 months, 1 year, 2 years, or 5 years of daily data.
- **Market overview** — view latest close, previous close, daily change, percentage change, and latest trading date.
- **Interactive charts** — explore closing prices, a 20-day moving average, and daily trading volume.
- **Two forecasting models:**
  - **Linear Regression:** uses lagged closing prices and 5-day/20-day trailing averages.
  - **LSTM:** uses a rolling window of the previous 60 closing prices.
- **Forecast horizons** — estimate 1, 7, or 30 upcoming trading days.
- **Model evaluation** — compare predictions against a chronological holdout set using MAE and RMSE.
- **Forecast table** — inspect estimated future closing prices.

## Tech Stack

| Component | Technology |
| --- | --- |
| User interface | Streamlit |
| Market data | Yahoo Finance via `yfinance` |
| Data processing | Python, Pandas, NumPy |
| Baseline model | Scikit-learn Linear Regression |
| Deep learning model | TensorFlow / Keras LSTM |
| Scaling and metrics | Scikit-learn |
| Visualizations | Plotly |

## Project Structure

```text
Stock-price-predictor/
├── app.py             # Streamlit dashboard and charts
├── models.py          # Model training, evaluation, and forecasting helpers
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores virtual environments, caches, and secrets
└── README.md          # Project documentation
```

## Getting Started

### Requirements

- Python **3.11** is a recommended starting point.
- `pip`
- An internet connection to retrieve market data from Yahoo Finance.

TensorFlow installation support varies by operating system and hardware. If installation fails, check the [official TensorFlow installation guide](https://www.tensorflow.org/install).

### 1. Clone the repository

```bash
git clone https://github.com/AafaqAaf/Stock-price-predictor.git
cd Stock-price-predictor
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## How to Use

1. Enter a ticker in the sidebar, for example `AAPL` or `RELIANCE.NS`.
2. Select the historical period.
3. Review the market overview, closing-price chart, moving average, and volume chart.
4. Choose **Linear Regression** or **LSTM**.
5. Select a forecast horizon of 1, 7, or 30 trading days.
6. Click **Generate forecast** to train the selected model and view its results.
7. Review the holdout-set MAE/RMSE, actual-versus-predicted chart, future forecast chart, and forecast table.

## Models and Evaluation

### Linear Regression

The baseline model uses previous closing prices at 1-, 2-, 3-, and 5-day lags, plus 5-day and 20-day trailing averages. It is trained on earlier observations and evaluated on the newest 20% of usable observations. The implementation requires at least 80 usable trading days after feature preparation.

### LSTM

The LSTM model uses the previous 60 closing prices to predict the next close. It fits a `MinMaxScaler` on the training portion, then trains a compact network with a 50-unit LSTM layer, 20% dropout, and a dense output layer. Early stopping is used during training. The implementation requires at least 160 trading days.

### Metrics

- **MAE (Mean Absolute Error):** average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** square root of the average squared prediction error; larger errors have more influence.

Lower values indicate smaller errors on the selected holdout data. They do not guarantee better future performance.

## Important Notes and Limitations

- Market data must be available from Yahoo Finance for the selected ticker and period.
- Forecasts are generated recursively: each estimated price is used as input for the next forecast step. Errors can accumulate as the horizon grows.
- Forecast dates are weekday estimates and do not account for exchange-specific holidays.
- Stock prices are affected by news, earnings, market sentiment, macroeconomic conditions, and other factors not modeled here.
- Historical patterns do not guarantee future results. Do not use this application as the sole basis for financial decisions.

## Troubleshooting

- **No data returned:** verify the ticker symbol and internet connection.
- **Not enough observations:** select a longer historical period.
- **TensorFlow not installed:** reinstall dependencies in the active virtual environment and restart Streamlit. Consult the official TensorFlow installation guide if needed.
- **PowerShell activation blocked:** use an approved Python environment activation method for your system, or run the environment's Python executable directly.

## Dependencies

The main dependencies are listed in [`requirements.txt`](requirements.txt): Streamlit, yfinance, Pandas, NumPy, scikit-learn, Plotly, and TensorFlow.

## License

No license is specified in this repository yet. Add a `LICENSE` file if you want to define how others may use, modify, and distribute the project.
