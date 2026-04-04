import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ========================
# CONFIGURATION

BANXICO_TOKEN = "****"
BANXICO_SERIE = "SF43718"

FRED_API_KEY = "*****"
FRED_SERIES_ID = "FEDFUNDS"

# ========================
# LOAD MXN/USD DATA FROM BANXICO


banxico_url = (
    f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
    f"{BANXICO_SERIE}/datos"
)

headers = {"Bmx-Token": BANXICO_TOKEN}

response = requests.get(banxico_url, headers=headers)
print("Banxico status code:", response.status_code)

banxico_data = response.json()

if "bmx" not in banxico_data:
    print("Error en la respuesta de Banxico:")
    print(banxico_data)
    raise SystemExit()

series_data = banxico_data["bmx"]["series"][0]["datos"]
df = pd.DataFrame(series_data)

df = df.rename(columns={"fecha": "date", "dato": "value"})
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df["value"] = pd.to_numeric(df["value"].replace("N/E", pd.NA), errors="coerce")
df = df.dropna().sort_values("date").reset_index(drop=True)

print("\nPrimeras filas de MXN/USD:")
print(df.head())

print("\nEstadísticas básicas:")
print(df["value"].describe())

# ========================
# VISUALIZE MXN/USD


plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["value"], linewidth=2, label="MXN/USD")
plt.title("MXN/USD Exchange Rate (Banco de México)")
plt.xlabel("Date")
plt.ylabel("MXN per USD")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

df["returns"] = df["value"].pct_change()

plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["returns"])
plt.title("Daily Returns MXN/USD")
plt.xlabel("Date")
plt.ylabel("Daily return")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ========================
# LOAD FED FUNDS RATE FROM FRED


fred_url = (
    f"https://api.stlouisfed.org/fred/series/observations"
    f"?series_id={FRED_SERIES_ID}"
    f"&api_key={FRED_API_KEY}"
    f"&file_type=json"
)

fred_response = requests.get(fred_url)
print("\nFRED status code:", fred_response.status_code)

fred_data = fred_response.json()

if "observations" not in fred_data:
    print("Error en la respuesta de FRED:")
    print(fred_data)
    raise SystemExit()

fred_df = pd.DataFrame(fred_data["observations"])
fred_df = fred_df[["date", "value"]].copy()
fred_df["date"] = pd.to_datetime(fred_df["date"], errors="coerce")
fred_df["value"] = pd.to_numeric(fred_df["value"], errors="coerce")
fred_df = fred_df.dropna().rename(columns={"value": "fed_funds_rate"})
fred_df = fred_df.sort_values("date").reset_index(drop=True)

print("\nPrimeras filas de FEDFUNDS:")
print(fred_df.head())

# ========================
# MERGE DATASETS


merged = pd.merge_asof(
    df.sort_values("date"),
    fred_df.sort_values("date"),
    on="date",
    direction="backward"
)

print("\nDatos unidos:")
print(merged[["date", "value", "fed_funds_rate"]].head())

# ========================
# FEATURE ENGINEERING


# Target:
# 1 = el peso se aprecia (MXN/USD baja al día siguiente)
# 0 = el peso se deprecia (MXN/USD sube al día siguiente)
merged["target"] = (merged["value"].shift(-1) < merged["value"]).astype(int)

# Basic lag and moving-average features
merged["lag_1"] = merged["value"].shift(1)
merged["lag_5"] = merged["value"].shift(5)
merged["ma_5"] = merged["value"].rolling(5).mean()
merged["ma_10"] = merged["value"].rolling(10).mean()

# Return and volatility features
merged["return_1"] = merged["value"].pct_change(1)
merged["return_5"] = merged["value"].pct_change(5)
merged["volatility_5"] = merged["return_1"].rolling(5).std()
merged["volatility_10"] = merged["return_1"].rolling(10).std()

# Macro feature changes
merged["fed_change"] = merged["fed_funds_rate"].diff()

merged = merged.dropna().reset_index(drop=True)

# ========================
# FINAL DATASET


features = [
    "lag_1",
    "lag_5",
    "ma_5",
    "ma_10",
    "fed_funds_rate",
    "return_1",
    "return_5",
    "volatility_5",
    "volatility_10",
    "fed_change"
]

X = merged[features]
y = merged["target"]

# Time-based split
split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ========================
# BASELINE


baseline = y_test.mode()[0]
baseline_acc = (y_test == baseline).mean()
print("\nBaseline accuracy:", baseline_acc)

# ========================
# RANDOM FOREST MODEL

rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)

# Feature importance
importances = rf_model.feature_importances_

plt.figure(figsize=(10, 5))
plt.bar(features, importances)
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========================
# LOGISTIC REGRESSION MODEL


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)

print("\n=== MODEL COMPARISON ===")
print(f"Baseline: {baseline_acc:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")
print(f"Logistic Regression: {lr_accuracy:.4f}")


