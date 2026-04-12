import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Create dataset inline
# -----------------------------
data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Carbon intensity refining (kgCO2e/CWT)": [43.0, 42.5, 41.8, 40.9, 39.5, 38.7, 37.8, 37.0, 36.5, 36.2],
    "Operational GHG emissions (million tCO2e)": [78, 75, 73, 70, 68, 60, 55, 52, 49, 47],
    "Energy consumption (TJ)": [850000, 830000, 820000, 810000, 800000, 780000, 790000, 795000, 800000, 805000],
    "Freshwater withdrawn (million m3)": [210, 205, 200, 198, 195, 190, 188, 185, 182, 180],
    "Hazardous waste generated (thousand metric tons/year)": [320, 310, 300, 290, 285, 270, 260, 255, 250, 245]
}

df = pd.DataFrame(data)

# -----------------------------
# Derived metrics
# -----------------------------
df["Emissions per TJ"] = (
    df["Operational GHG emissions (million tCO2e)"] * 1_000_000
) / df["Energy consumption (TJ)"]

df["Carbon Intensity YoY (%)"] = (
    df["Carbon intensity refining (kgCO2e/CWT)"].pct_change() * 100
)

df["GHG Emissions YoY (%)"] = (
    df["Operational GHG emissions (million tCO2e)"].pct_change() * 100
)

df["Energy YoY (%)"] = (
    df["Energy consumption (TJ)"].pct_change() * 100
)

df["Freshwater YoY (%)"] = (
    df["Freshwater withdrawn (million m3)"].pct_change() * 100
)

df["Hazardous Waste YoY (%)"] = (
    df["Hazardous waste generated (thousand metric tons/year)"].pct_change() * 100
)

# -----------------------------
# Print dataframe
# -----------------------------
print("\nDataset preview:\n")
print(df)

# -----------------------------
# Plot 1: Refining carbon intensity
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["Year"], df["Carbon intensity refining (kgCO2e/CWT)"], marker="o")
plt.title("Petrobras Refining Carbon Intensity (2015-2024)")
plt.xlabel("Year")
plt.ylabel("kgCO2e/CWT")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: Operational GHG emissions
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["Year"], df["Operational GHG emissions (million tCO2e)"], marker="o")
plt.title("Operational GHG Emissions (2015-2024)")
plt.xlabel("Year")
plt.ylabel("Million tCO2e")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 3: Energy consumption
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["Year"], df["Energy consumption (TJ)"], marker="o")
plt.title("Energy Consumption (2015-2024)")
plt.xlabel("Year")
plt.ylabel("TJ")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 4: Hazardous waste
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(df["Year"], df["Hazardous waste generated (thousand metric tons/year)"])
plt.title("Hazardous Waste Generated (2015-2024)")
plt.xlabel("Year")
plt.ylabel("Thousand metric tons")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 5: Freshwater withdrawn
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["Year"], df["Freshwater withdrawn (million m3)"], marker="o")
plt.title("Freshwater Withdrawn (2015-2024)")
plt.xlabel("Year")
plt.ylabel("Million m3")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 6: Emissions per TJ
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["Year"], df["Emissions per TJ"], marker="o")
plt.title("Operational Emissions per Unit of Energy (2015-2024)")
plt.xlabel("Year")
plt.ylabel("tCO2e per TJ")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Summary metrics
# -----------------------------
summary = {
    "Carbon intensity reduction (%)": round(
        (df.iloc[-1]["Carbon intensity refining (kgCO2e/CWT)"] /
         df.iloc[0]["Carbon intensity refining (kgCO2e/CWT)"] - 1) * 100, 2
    ),
    "GHG emissions reduction (%)": round(
        (df.iloc[-1]["Operational GHG emissions (million tCO2e)"] /
         df.iloc[0]["Operational GHG emissions (million tCO2e)"] - 1) * 100, 2
    ),
    "Energy consumption change (%)": round(
        (df.iloc[-1]["Energy consumption (TJ)"] /
         df.iloc[0]["Energy consumption (TJ)"] - 1) * 100, 2
    ),
    "Freshwater withdrawn change (%)": round(
        (df.iloc[-1]["Freshwater withdrawn (million m3)"] /
         df.iloc[0]["Freshwater withdrawn (million m3)"] - 1) * 100, 2
    ),
    "Hazardous waste change (%)": round(
        (df.iloc[-1]["Hazardous waste generated (thousand metric tons/year)"] /
         df.iloc[0]["Hazardous waste generated (thousand metric tons/year)"] - 1) * 100, 2
    ),
    "Emissions per TJ change (%)": round(
        (df.iloc[-1]["Emissions per TJ"] / df.iloc[0]["Emissions per TJ"] - 1) * 100, 2
    ),
}

print("\nSummary Insights:\n")
for metric, value in summary.items():
    print(f"{metric}: {value}%")
