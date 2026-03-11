# Natural Gas Price Interpolation and 1-Year Extrapolation
# 1) Loads monthly natural gas price data
# 2) Uses simple linear interpolation for dates between known points
# 3) Uses average monthly seasonality for future estimates
# 4) Creates basic plots
# 5) Exports daily estimates to CSV

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1) Load the data
# ---------------------------------------------------------------

raw_csv = """Dates,Prices
10/31/20,1.01E+01
11/30/20,1.03E+01
12/31/20,1.10E+01
1/31/21,1.09E+01
2/28/21,1.09E+01
3/31/21,1.09E+01
4/30/21,1.04E+01
5/31/21,9.84E+00
6/30/21,1.00E+01
7/31/21,1.01E+01
8/31/21,1.03E+01
9/30/21,1.02E+01
10/31/21,1.01E+01
11/30/21,1.12E+01
12/31/21,1.14E+01
1/31/22,1.15E+01
2/28/22,1.18E+01
3/31/22,1.15E+01
4/30/22,1.07E+01
5/31/22,1.07E+01
6/30/22,1.04E+01
7/31/22,1.05E+01
8/31/22,1.04E+01
9/30/22,1.08E+01
10/31/22,1.10E+01
11/30/22,1.16E+01
12/31/22,1.16E+01
1/31/23,1.21E+01
2/28/23,1.17E+01
3/31/23,1.20E+01
4/30/23,1.15E+01
5/31/23,1.12E+01
6/30/23,1.09E+01
7/31/23,1.14E+01
8/31/23,1.11E+01
9/30/23,1.15E+01
10/31/23,1.18E+01
11/30/23,1.22E+01
12/31/23,1.28E+01
1/31/24,1.26E+01
2/29/24,1.24E+01
3/31/24,1.27E+01
4/30/24,1.21E+01
5/31/24,1.14E+01
6/30/24,1.15E+01
7/31/24,1.16E+01
8/31/24,1.15E+01
9/30/24,1.18E+01
"""

monthly = pd.read_csv(io.StringIO(raw_csv))
monthly["Dates"] = pd.to_datetime(monthly["Dates"], format="%m/%d/%y")
monthly["Prices"] = pd.to_numeric(monthly["Prices"])
monthly = monthly.sort_values("Dates").reset_index(drop=True)

# Add month column
monthly["Month"] = monthly["Dates"].dt.month

# ---------------------------------------------------------------
# 2) Simple monthly seasonality
# ---------------------------------------------------------------

# Average price for each calendar month
monthly_avg = monthly.groupby("Month")["Prices"].mean()

# Overall average price
overall_avg = monthly["Prices"].mean()

# Seasonal factor = month average / overall average
seasonal_factors = monthly_avg / overall_avg

seasonal_df = pd.DataFrame({
    "Month": monthly_avg.index,
    "AveragePrice": monthly_avg.values,
    "SeasonalFactor": seasonal_factors.values
})

# ---------------------------------------------------------------
# 3) Price estimation function
# ---------------------------------------------------------------

def estimate_price(date_like):
    """
    Estimate price for any date.
    
    Rules:
    - If date is inside historical range: use linear interpolation
    - If date is up to 1 year after last observed date:
      use last known price adjusted by monthly seasonal factor
    """
    dt = pd.to_datetime(date_like)

    first_date = monthly["Dates"].min()
    last_date = monthly["Dates"].max()

    if dt < first_date:
        raise ValueError("Date is before the first available observation.")

    if dt <= last_date:
        # Interpolation inside known history
        temp = monthly.set_index("Dates")["Prices"].reindex(
            monthly.set_index("Dates").index.union([dt])
        ).sort_index()

        temp = temp.interpolate(method="time")
        return float(temp.loc[dt])

    if dt > last_date + pd.Timedelta(days=365):
        raise ValueError("Date is more than 1 year after the last observation.")

    # Simple extrapolation for future dates
    last_price = float(monthly.loc[monthly.index[-1], "Prices"])
    future_month = dt.month
    factor = float(seasonal_factors.loc[future_month])

    return last_price * factor

# ---------------------------------------------------------------
# 4) Create daily estimated curve
# ---------------------------------------------------------------

start_date = monthly["Dates"].min()
last_date = monthly["Dates"].max()
end_date = last_date + pd.Timedelta(days=365)

daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
daily_prices = [estimate_price(d) for d in daily_index]

daily_df = pd.DataFrame({
    "Date": daily_index,
    "EstimatedPrice": daily_prices
})

daily_df.to_csv("daily_price_estimates_simple.csv", index=False)

# ---------------------------------------------------------------
# 5) Plot 1: Historical monthly prices
# ---------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(monthly["Dates"], monthly["Prices"], marker="o")
plt.title("Monthly Natural Gas Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_prices_simple.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 6) Plot 2: Seasonal factors by month
# ---------------------------------------------------------------

plt.figure(figsize=(10, 4))
plt.bar(seasonal_df["Month"], seasonal_df["SeasonalFactor"])
plt.axhline(1.0, color="black", linewidth=1)
plt.title("Seasonal Factors by Month")
plt.xlabel("Month")
plt.ylabel("Seasonal Factor")
plt.tight_layout()
plt.savefig("seasonal_factors_simple.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 7) Plot 3: Daily estimated curve
# ---------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(daily_df["Date"], daily_df["EstimatedPrice"], label="Daily estimated price")
plt.scatter(monthly["Dates"], monthly["Prices"], color="black", s=20, label="Monthly observations")
plt.title("Daily Estimated Natural Gas Price Curve")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("daily_curve_simple.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 8) Sample estimates
# ---------------------------------------------------------------

sample_dates = [
    "2020-10-31",
    "2020-11-15",
    "2023-02-15",
    "2024-09-30",
    "2025-03-30"
]

print("Sample estimates:")
for d in sample_dates:
    print(d, "->", round(estimate_price(d), 4))
