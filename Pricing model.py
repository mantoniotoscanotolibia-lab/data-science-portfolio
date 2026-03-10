# Natural Gas Storage Contract Prototype Pricing Model
# ---------------------------------------------------
# This script:
# 1) Loads monthly natural gas prices
# 2) Fits a simple time-series model on log prices:
#       log(price_t) = quadratic trend + month-of-year effects
# 3) Builds estimate_price(date_like) to estimate prices on arbitrary dates
# 4) Builds price_storage_contract(...) to value a gas storage contract
# 5) Runs sample test cases
#
# Assumptions:
# - No transport delay
# - Interest rates are zero
# - No need to account for weekends/holidays
# - Manual oversight remains in place; this is a prototype model

import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------
# 1) LOAD MONTHLY PRICE DATA
# ---------------------------------------------------------------------

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
monthly["Dates"] = pd.to_datetime(monthly["Dates"], format="%m/%d/%y", errors="raise")
monthly["Prices"] = pd.to_numeric(monthly["Prices"], errors="raise")
monthly = monthly.sort_values("Dates").reset_index(drop=True)

monthly["Month"] = monthly["Dates"].dt.month
monthly["Year"] = monthly["Dates"].dt.year

start_date = monthly["Dates"].min()
monthly["t_months"] = (
    (monthly["Dates"].dt.year - start_date.year) * 12
    + (monthly["Dates"].dt.month - start_date.month)
).astype(float)

monthly["log_price"] = np.log(monthly["Prices"])

# ---------------------------------------------------------------------
# 2) FIT PRICE MODEL: QUADRATIC TREND + MONTH DUMMIES
# ---------------------------------------------------------------------

month_dummies = pd.get_dummies(
    monthly["Month"].astype(int),
    prefix="m",
    drop_first=True
).astype(float)

monthly["t"] = monthly["t_months"]
monthly["t2"] = monthly["t"] ** 2

X = pd.concat(
    [
        pd.Series(1.0, index=monthly.index, name="const"),
        monthly[["t", "t2"]].astype(float),
        month_dummies
    ],
    axis=1
)

y = monthly["log_price"].values

beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
coef = pd.Series(beta, index=X.columns)

monthly["fitted_log_price"] = X.values @ coef.values
monthly["fitted_price"] = np.exp(monthly["fitted_log_price"])

# ---------------------------------------------------------------------
# 3) BUILD SEASONAL MULTIPLIERS TABLE
# ---------------------------------------------------------------------

month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

seasonal_rel = {}
for m in range(1, 13):
    d = {col: 0.0 for col in month_dummies.columns}
    if m > 1:
        key = f"m_{m}"
        if key in d:
            d[key] = 1.0

    d_vec = np.array([d.get(col, 0.0) for col in month_dummies.columns], dtype=float)
    dummy_effect = float(np.dot(coef[month_dummies.columns].values, d_vec))
    seasonal_rel[m] = math.exp(dummy_effect)

geom_mean = math.exp(np.mean(np.log(list(seasonal_rel.values()))))
for m in seasonal_rel:
    seasonal_rel[m] /= geom_mean

seasonal_df = pd.DataFrame({
    "Month": [month_names[m] for m in range(1, 13)],
    "SeasonalMultiplier": [seasonal_rel[m] for m in range(1, 13)]
})

# ---------------------------------------------------------------------
# 4) PRICE ESTIMATION FUNCTION
# ---------------------------------------------------------------------

def _design_row_for_date(dt: pd.Timestamp) -> np.ndarray:
    delta_days = (dt - start_date).days
    t = delta_days / 30.4375
    t2 = t ** 2

    d = {col: 0.0 for col in month_dummies.columns}
    m = int(dt.month)
    if m > 1:
        key = f"m_{m}"
        if key in d:
            d[key] = 1.0

    row = [1.0, t, t2] + [d[c] for c in month_dummies.columns]
    return np.array(row, dtype=float)

def _predict_log_price(dt: pd.Timestamp) -> float:
    row = _design_row_for_date(dt)
    return float(np.dot(row, coef.values))

def estimate_price(date_like):
    """
    Estimate the gas price on an arbitrary date using the fitted model.

    Parameters
    ----------
    date_like : str | datetime | pd.Timestamp

    Returns
    -------
    float
    """
    if isinstance(date_like, str):
        dt = pd.to_datetime(date_like)
    elif isinstance(date_like, datetime):
        dt = pd.Timestamp(date_like)
    elif isinstance(date_like, pd.Timestamp):
        dt = date_like
    else:
        raise ValueError("Unsupported date type. Use string, datetime, or pd.Timestamp.")

    last_obs = monthly["Dates"].max()
    if dt > last_obs + pd.Timedelta(days=365):
        raise ValueError(
            f"Date {dt.date()} exceeds the supported forecast horizon "
            f"(up to one year after {last_obs.date()})."
        )

    lp = _predict_log_price(dt)
    return float(np.exp(lp))

# ---------------------------------------------------------------------
# 5) PROTOTYPE STORAGE CONTRACT PRICING FUNCTION
# ---------------------------------------------------------------------

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    price_func,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_unit_per_day=0.0,
    injection_volumes=None,
    withdrawal_volumes=None,
    enforce_empty_end=False,
    verbose=True
):
    """
    Price a prototype natural gas storage contract.

    Parameters
    ----------
    injection_dates : list-like
        Dates on which gas is injected into storage.
    withdrawal_dates : list-like
        Dates on which gas is withdrawn from storage.
    price_func : callable
        Function returning price for a given date, e.g. estimate_price.
    injection_rate : float
        Maximum volume that can be injected per injection date.
    withdrawal_rate : float
        Maximum volume that can be withdrawn per withdrawal date.
    max_volume : float
        Maximum inventory that can be stored.
    storage_cost_per_unit_per_day : float
        Storage cost per unit of gas per day.
    injection_volumes : list-like or None
        Requested injection volumes. If None, use max feasible volume.
    withdrawal_volumes : list-like or None
        Requested withdrawal volumes. If None, use max feasible volume.
    enforce_empty_end : bool
        If True, raise error if final inventory is not zero.
    verbose : bool
        If True, print summary.

    Returns
    -------
    dict
        contract_value, cash_flow_table, total_injected, total_withdrawn, final_inventory
    """

    injection_dates = pd.to_datetime(injection_dates)
    withdrawal_dates = pd.to_datetime(withdrawal_dates)

    if len(injection_dates) == 0 and len(withdrawal_dates) == 0:
        raise ValueError("At least one injection or one withdrawal date is required.")

    injection_dates = pd.Series(injection_dates).sort_values().reset_index(drop=True)
    withdrawal_dates = pd.Series(withdrawal_dates).sort_values().reset_index(drop=True)

    if injection_volumes is None:
        injection_volumes = [None] * len(injection_dates)
    elif len(injection_volumes) != len(injection_dates):
        raise ValueError("injection_volumes must have same length as injection_dates.")

    if withdrawal_volumes is None:
        withdrawal_volumes = [None] * len(withdrawal_dates)
    elif len(withdrawal_volumes) != len(withdrawal_dates):
        raise ValueError("withdrawal_volumes must have same length as withdrawal_dates.")

    events = []
    for dt, vol in zip(injection_dates, injection_volumes):
        events.append({
            "date": pd.Timestamp(dt),
            "type": "inject",
            "requested_volume": vol
        })

    for dt, vol in zip(withdrawal_dates, withdrawal_volumes):
        events.append({
            "date": pd.Timestamp(dt),
            "type": "withdraw",
            "requested_volume": vol
        })

    type_order = {"inject": 0, "withdraw": 1}
    events = sorted(events, key=lambda x: (x["date"], type_order[x["type"]]))

    inventory = 0.0
    total_value = 0.0
    total_injected = 0.0
    total_withdrawn = 0.0
    records = []

    prev_date = events[0]["date"]

    for event in events:
        current_date = event["date"]

        # Storage cost accrued between events
        days_held = (current_date - prev_date).days
        storage_cost = inventory * storage_cost_per_unit_per_day * max(days_held, 0)
        total_value -= storage_cost

        if days_held > 0 and storage_cost != 0:
            records.append({
                "date": current_date,
                "action": "storage_cost",
                "price": np.nan,
                "volume": 0.0,
                "inventory_after": inventory,
                "cash_flow": -storage_cost
            })

        price = float(price_func(current_date))

        if event["type"] == "inject":
            requested = event["requested_volume"]

            if requested is None:
                volume = min(injection_rate, max_volume - inventory)
            else:
                if requested < 0:
                    raise ValueError("Injection volumes must be non-negative.")
                if requested > injection_rate:
                    raise ValueError(
                        f"Requested injection volume {requested} exceeds injection_rate {injection_rate} on {current_date.date()}."
                    )
                if inventory + requested > max_volume:
                    raise ValueError(f"Injection on {current_date.date()} exceeds max storage capacity.")
                volume = requested

            cash_flow = -volume * price
            inventory += volume
            total_value += cash_flow
            total_injected += volume

            records.append({
                "date": current_date,
                "action": "inject",
                "price": price,
                "volume": volume,
                "inventory_after": inventory,
                "cash_flow": cash_flow
            })

        elif event["type"] == "withdraw":
            requested = event["requested_volume"]

            if requested is None:
                volume = min(withdrawal_rate, inventory)
            else:
                if requested < 0:
                    raise ValueError("Withdrawal volumes must be non-negative.")
                if requested > withdrawal_rate:
                    raise ValueError(
                        f"Requested withdrawal volume {requested} exceeds withdrawal_rate {withdrawal_rate} on {current_date.date()}."
                    )
                if requested > inventory:
                    raise ValueError(f"Withdrawal on {current_date.date()} exceeds available inventory.")
                volume = requested

            cash_flow = volume * price
            inventory -= volume
            total_value += cash_flow
            total_withdrawn += volume

            records.append({
                "date": current_date,
                "action": "withdraw",
                "price": price,
                "volume": volume,
                "inventory_after": inventory,
                "cash_flow": cash_flow
            })

        prev_date = current_date

    if enforce_empty_end and abs(inventory) > 1e-9:
        raise ValueError(f"Ending inventory is {inventory:.4f}, but enforce_empty_end=True.")

    cash_flow_table = pd.DataFrame(records)

    result = {
        "contract_value": total_value,
        "cash_flow_table": cash_flow_table,
        "total_injected": total_injected,
        "total_withdrawn": total_withdrawn,
        "final_inventory": inventory
    }

    if verbose:
        print("\n================ CONTRACT SUMMARY ================")
        print(f"Total injected   : {total_injected:.4f}")
        print(f"Total withdrawn  : {total_withdrawn:.4f}")
        print(f"Final inventory  : {inventory:.4f}")
        print(f"Contract value   : {total_value:.4f}")
        print("==================================================\n")

    return result

# ---------------------------------------------------------------------
# 6) OPTIONAL OUTPUTS FROM PRICE MODEL
# ---------------------------------------------------------------------

last_date = monthly["Dates"].max()
end_date = last_date + pd.Timedelta(days=365)
daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
daily_pred = pd.Series([estimate_price(d) for d in daily_index], index=daily_index)

out_df = pd.DataFrame({
    "Date": daily_pred.index,
    "EstimatedPrice": daily_pred.values
})
out_df.to_csv("daily_price_estimates.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(monthly["Dates"], monthly["Prices"], marker="o", label="Actual (month-end)")
plt.plot(monthly["Dates"], monthly["fitted_price"], marker="s", label="Model fit (monthly)")
plt.title("Natural Gas: Actual vs Fitted (Month-End)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("monthly_actual_vs_fitted.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 4))
plt.bar(seasonal_df["Month"], seasonal_df["SeasonalMultiplier"])
plt.axhline(1.0, color="black", linewidth=1)
plt.title("Multiplicative Seasonality by Month")
plt.xlabel("Month")
plt.ylabel("Seasonal Multiplier")
plt.tight_layout()
plt.savefig("seasonal_multipliers.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(daily_pred.index, daily_pred.values, label="Daily estimate (+1 year)")
plt.scatter(monthly["Dates"], monthly["Prices"], color="black", s=20, zorder=3, label="Monthly observations")
plt.title("Estimated Daily Natural Gas Price Curve")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("daily_curve_with_extrapolation.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 7) TEST CASES FOR THE ACTIVITY
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing storage contract pricing model...\n")

    # Sample test 1
    inj_dates_1 = ["2024-04-30", "2024-05-31", "2024-06-30"]
    wd_dates_1 = ["2024-10-31", "2024-11-30", "2024-12-31"]

    result_1 = price_storage_contract(
        injection_dates=inj_dates_1,
        withdrawal_dates=wd_dates_1,
        price_func=estimate_price,
        injection_rate=1000,
        withdrawal_rate=1000,
        max_volume=3000,
        storage_cost_per_unit_per_day=0.01,
        injection_volumes=[1000, 1000, 1000],
        withdrawal_volumes=[1000, 1000, 1000],
        enforce_empty_end=True,
        verbose=True
    )

    print("Test 1 cash flows:")
    print(result_1["cash_flow_table"])
    print("\n")

    # Sample test 2
    inj_dates_2 = ["2024-03-31", "2024-04-30", "2024-05-31"]
    wd_dates_2 = ["2024-09-30", "2024-10-31", "2024-11-30"]

    result_2 = price_storage_contract(
        injection_dates=inj_dates_2,
        withdrawal_dates=wd_dates_2,
        price_func=estimate_price,
        injection_rate=800,
        withdrawal_rate=800,
        max_volume=2400,
        storage_cost_per_unit_per_day=0.015,
        injection_volumes=None,
        withdrawal_volumes=None,
        enforce_empty_end=True,
        verbose=True
    )

    print("Test 2 cash flows:")
    print(result_2["cash_flow_table"])
    print("\n")

    # Sample test 3
    inj_dates_3 = ["2024-05-01", "2024-06-01"]
    wd_dates_3 = ["2024-11-01"]

    result_3 = price_storage_contract(
        injection_dates=inj_dates_3,
        withdrawal_dates=wd_dates_3,
        price_func=estimate_price,
        injection_rate=500,
        withdrawal_rate=500,
        max_volume=1000,
        storage_cost_per_unit_per_day=0.02,
        injection_volumes=[500, 500],
        withdrawal_volumes=[500],
        enforce_empty_end=False,
        verbose=True
    )

    print("Test 3 cash flows:")
    print(result_3["cash_flow_table"])
    print("\n")

    sample_dates = [
        start_date,
        start_date + pd.Timedelta(days=15),
        pd.Timestamp("2023-02-15"),
        last_date,
        last_date + pd.Timedelta(days=180),
    ]

    print("Sample estimated prices:")
    for d in sample_dates:
        print(f"{d.date()}: {estimate_price(d):.4f}")