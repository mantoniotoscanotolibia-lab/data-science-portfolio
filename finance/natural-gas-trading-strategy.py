import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# ====
# 1) Load data
# ===

raw_csv = """Dates,Prices
10/31/20,10.1
11/30/20,10.3
12/31/20,10.5
01/31/21,10.7
02/28/21,11.0
03/31/21,10.8
04/30/21,10.6
05/31/21,10.4
06/30/21,10.2
07/31/21,10.1
08/31/21,10.0
09/30/21,10.2
10/31/21,10.6
11/30/21,11.0
12/31/21,11.4
01/31/22,11.8
02/28/22,12.3
03/31/22,12.0
04/30/22,11.7
05/31/22,11.3
06/30/22,11.0
07/31/22,10.8
08/31/22,10.7
09/30/22,10.9
10/31/22,11.2
11/30/22,11.6
12/31/22,12.0
01/31/23,12.5
02/28/23,12.9
03/31/23,12.4
04/30/23,12.0
05/31/23,11.7
06/30/23,11.4
07/31/23,11.1
08/31/23,11.0
09/30/23,11.3
10/31/23,11.7
11/30/23,12.0
12/31/23,12.5
01/31/24,13.0
02/29/24,13.4
03/31/24,12.9
04/30/24,12.5
05/31/24,12.1
06/30/24,11.9
07/31/24,11.6
08/31/24,11.4
09/30/24,11.7
"""

from io import StringIO
df = pd.read_csv(StringIO(raw_csv))
df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
df = df.sort_values("Dates").reset_index(drop=True)

# ==
# 2) Forecasting model: log(price) = quadratic trend + seasonality
# ===

df["t"] = np.arange(len(df))
df["log_price"] = np.log(df["Prices"])
df["month"] = df["Dates"].dt.month

month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True).astype(float)

X = pd.concat([
    pd.Series(1.0, index=df.index, name="intercept"),
    df["t"].astype(float),
    (df["t"] ** 2).astype(float).rename("t2"),
    month_dummies
], axis=1)

y = df["log_price"].values
beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)

df["fitted_log"] = X.values @ beta
df["fitted_price"] = np.exp(df["fitted_log"])

# ===
# 3) Build future monthly forecasts
# ====

future_months = 12
last_date = df["Dates"].max()

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthEnd(1),
    periods=future_months,
    freq="ME"
)

future_df = pd.DataFrame({"Dates": future_dates})
future_df["t"] = np.arange(len(df), len(df) + len(future_df))
future_df["month"] = future_df["Dates"].dt.month

future_dummies = pd.get_dummies(future_df["month"], prefix="m", drop_first=True).astype(float)

for col in month_dummies.columns:
    if col not in future_dummies.columns:
        future_dummies[col] = 0.0
future_dummies = future_dummies[month_dummies.columns]

X_future = pd.concat([
    pd.Series(1.0, index=future_df.index, name="intercept"),
    future_df["t"].astype(float),
    (future_df["t"] ** 2).astype(float).rename("t2"),
    future_dummies
], axis=1)

future_df["forecast_log"] = X_future.values @ beta
future_df["forecast_price"] = np.exp(future_df["forecast_log"])

# ===
# 4) Generate trading signals
# ====

signals_df = future_df.copy()
signals_df["current_price"] = np.nan

last_known_price = df["Prices"].iloc[-1]
signals_df.loc[0, "current_price"] = last_known_price

for i in range(1, len(signals_df)):
    signals_df.loc[i, "current_price"] = signals_df.loc[i - 1, "forecast_price"]

signals_df["expected_return"] = (
    signals_df["forecast_price"] - signals_df["current_price"]
) / signals_df["current_price"]

buy_threshold = 0.02
sell_threshold = -0.02

def get_signal(x):
    if x > buy_threshold:
        return 1
    elif x < sell_threshold:
        return -1
    return 0

signals_df["signal"] = signals_df["expected_return"].apply(get_signal)

# ===
# 5) Simple backtest
# ====

# Use historical fitted changes as if signals were generated sequentially
bt = df.copy()
bt["next_price"] = bt["Prices"].shift(-1)
bt["predicted_next_price"] = df["fitted_price"].shift(-1)

bt = bt.dropna().reset_index(drop=True)

bt["expected_return"] = (
    bt["predicted_next_price"] - bt["Prices"]
) / bt["Prices"]

bt["signal"] = bt["expected_return"].apply(get_signal)

# Strategy return:
# + actual return if long
# - actual return if short
# 0 if hold
bt["actual_return"] = (bt["next_price"] - bt["Prices"]) / bt["Prices"]
bt["strategy_return"] = bt["signal"] * bt["actual_return"]

initial_capital = 10000
bt["equity_curve"] = initial_capital * (1 + bt["strategy_return"]).cumprod()
bt["buy_hold_curve"] = initial_capital * (bt["next_price"] / bt["Prices"].iloc[0])

# =====
# 6) Performance metrics
# =====

def annualized_volatility(returns, periods_per_year=12):
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns, periods_per_year=12, risk_free_rate=0.0):
    excess = returns - (risk_free_rate / periods_per_year)
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)

def max_drawdown(equity_curve):
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

metrics = {
    "Strategy Total Return": bt["equity_curve"].iloc[-1] / initial_capital - 1,
    "Buy & Hold Return": bt["buy_hold_curve"].iloc[-1] / initial_capital - 1,
    "Strategy Volatility": annualized_volatility(bt["strategy_return"]),
    "Strategy Sharpe": sharpe_ratio(bt["strategy_return"]),
    "Strategy Max Drawdown": max_drawdown(bt["equity_curve"]),
    "Number of Trades": (bt["signal"] != 0).sum()
}

metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
print(metrics_df)

# ===
# 7) Save outputs
# ===

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Actual vs fitted
plt.figure(figsize=(10, 5))
plt.plot(df["Dates"], df["Prices"], label="Actual")
plt.plot(df["Dates"], df["fitted_price"], label="Fitted")
plt.title("Monthly Actual vs Fitted Natural Gas Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "monthly_actual_vs_fitted.png")
plt.close()

# Future forecast
plt.figure(figsize=(10, 5))
plt.plot(df["Dates"], df["Prices"], label="Historical Prices")
plt.plot(future_df["Dates"], future_df["forecast_price"], label="Forecasted Prices")
plt.title("Natural Gas Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "daily_curve_with_extrapolation.png")
plt.close()

# Strategy vs benchmark
plt.figure(figsize=(10, 5))
plt.plot(bt["Dates"], bt["equity_curve"], label="Strategy")
plt.plot(bt["Dates"], bt["buy_hold_curve"], label="Buy & Hold")
plt.title("Strategy vs Buy-and-Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "strategy_vs_buyhold.png")
plt.close()

# Equity curve only
plt.figure(figsize=(10, 5))
plt.plot(bt["Dates"], bt["equity_curve"], label="Strategy Equity Curve")
plt.title("Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "equity_curve.png")
plt.close()

print("Project completed. Outputs saved in /outputs.")
