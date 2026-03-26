import pandas as pd
import matplotlib.pyplot as plt

# ======
# 1. Example dataset
# ===

data = {
    "property_id": ["A01", "A02", "A03", "A04", "A05"],

    # Material quantities (kg)
    "wood_kg": [1200, 800, 1500, 1000, 600],
    "metal_kg": [300, 200, 400, 250, 150],
    "plastic_kg": [150, 100, 200, 120, 80],
    "glass_kg": [200, 150, 250, 180, 100],

    # Reusable percentages (0–1)
    "wood_reuse_pct": [0.7, 0.5, 0.8, 0.6, 0.4],
    "metal_reuse_pct": [0.9, 0.85, 0.95, 0.9, 0.8],
    "plastic_reuse_pct": [0.3, 0.2, 0.35, 0.25, 0.15],
    "glass_reuse_pct": [0.6, 0.5, 0.7, 0.6, 0.4]
}

df = pd.DataFrame(data)

# ===
# 2. Material prices (USD per kg)
# ===

prices = {
    "wood": 0.1,
    "metal": 0.5,
    "plastic": 0.2,
    "glass": 0.15
}

# ===
# 3. Calculate reusable materials
# ====

df["wood_reusable"] = df["wood_kg"] * df["wood_reuse_pct"]
df["metal_reusable"] = df["metal_kg"] * df["metal_reuse_pct"]
df["plastic_reusable"] = df["plastic_kg"] * df["plastic_reuse_pct"]
df["glass_reusable"] = df["glass_kg"] * df["glass_reuse_pct"]

# Total materials
df["total_material_kg"] = (
    df["wood_kg"] + df["metal_kg"] +
    df["plastic_kg"] + df["glass_kg"]
)

# Total reusable
df["total_reusable_kg"] = (
    df["wood_reusable"] + df["metal_reusable"] +
    df["plastic_reusable"] + df["glass_reusable"]
)

# Recovery rate (%)
df["recovery_rate_pct"] = (
    df["total_reusable_kg"] / df["total_material_kg"]
) * 100

# ====
# 4. Economic value of recovered materials
# ===

df["wood_value"] = df["wood_reusable"] * prices["wood"]
df["metal_value"] = df["metal_reusable"] * prices["metal"]
df["plastic_value"] = df["plastic_reusable"] * prices["plastic"]
df["glass_value"] = df["glass_reusable"] * prices["glass"]

df["total_recovery_value"] = (
    df["wood_value"] + df["metal_value"] +
    df["plastic_value"] + df["glass_value"]
)

# ===
# 5. Rank properties
# ===

df = df.sort_values("total_recovery_value", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# ===
# 6. Output results
# ===

output_cols = [
    "rank",
    "property_id",
    "total_material_kg",
    "total_reusable_kg",
    "recovery_rate_pct",
    "total_recovery_value"
]

print("\nMaterial Recovery Analysis Results\n")
print(df[output_cols].round(2))

# Save CSV
df[output_cols].to_csv("material_recovery_results.csv", index=False)

# ==
# 7. Visualization
# ===

plt.figure(figsize=(8,5))
plt.bar(df["property_id"], df["total_recovery_value"])
plt.title("Recovery Value by Property")
plt.xlabel("Property ID")
plt.ylabel("Recovery Value (USD)")
plt.tight_layout()
plt.savefig("recovery_value_bar_chart.png")
plt.show()

print("\nFiles saved:")
print("- material_recovery_results.csv")
print("- recovery_value_bar_chart.png")
