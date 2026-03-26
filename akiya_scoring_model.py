import pandas as pd
import numpy as np

# ====
# 1. Example dataset
# ===

data = {
    "property_id": ["A01", "A02", "A03", "A04", "A05"],
    "structural_condition": [4, 2, 5, 3, 1],      # 1 = very poor, 5 = very good
    "location_score": [5, 3, 4, 2, 3],            # 1 = weak location, 5 = strong location
    "accessibility_score": [4, 2, 5, 3, 2],       # 1 = difficult access, 5 = easy access
    "reuse_potential": [5, 3, 4, 2, 2],           # 1 = low potential, 5 = high potential
    "estimated_renovation_cost": [45000, 90000, 60000, 70000, 120000],  # USD or local equivalent
    "reusable_material_pct": [70, 40, 80, 55, 30] # 0 to 100
}

df = pd.DataFrame(data)

# ===
# 2. Normalize cost so that lower cost gets a higher score
# ====
# We convert cost into a 0-100 score where lower renovation cost is better.

cost_min = df["estimated_renovation_cost"].min()
cost_max = df["estimated_renovation_cost"].max()

if cost_max == cost_min:
    df["cost_score"] = 100
else:
    df["cost_score"] = 100 * (cost_max - df["estimated_renovation_cost"]) / (cost_max - cost_min)

# ===
# 3. Convert 1-5 scores to 0-100 scale
# ===

score_columns_1_to_5 = [
    "structural_condition",
    "location_score",
    "accessibility_score",
    "reuse_potential"
]

for col in score_columns_1_to_5:
    df[col + "_scaled"] = (df[col] - 1) / 4 * 100

# reusable_material_pct is already close to a 0-100 scale
df["reusable_material_score"] = df["reusable_material_pct"]

# ===
# 4. Weighted scoring model
# ===
# Adjust weights if you want. Total should add to 1.00.

weights = {
    "structural_condition_scaled": 0.25,
    "location_score_scaled": 0.20,
    "accessibility_score_scaled": 0.15,
    "reuse_potential_scaled": 0.20,
    "cost_score": 0.10,
    "reusable_material_score": 0.10
}

df["final_score"] = (
    df["structural_condition_scaled"] * weights["structural_condition_scaled"] +
    df["location_score_scaled"] * weights["location_score_scaled"] +
    df["accessibility_score_scaled"] * weights["accessibility_score_scaled"] +
    df["reuse_potential_scaled"] * weights["reuse_potential_scaled"] +
    df["cost_score"] * weights["cost_score"] +
    df["reusable_material_score"] * weights["reusable_material_score"]
)

# ====
# 5. Classify priority level
# ==

def classify_priority(score):
    if score >= 75:
        return "High Priority"
    elif score >= 50:
        return "Medium Priority"
    return "Low Priority"

df["priority_level"] = df["final_score"].apply(classify_priority)

# ====
# 6. Rank properties
# ====

df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# ====
# 7. Display final output
# ====

output_columns = [
    "rank",
    "property_id",
    "final_score",
    "priority_level",
    "structural_condition",
    "location_score",
    "accessibility_score",
    "reuse_potential",
    "estimated_renovation_cost",
    "reusable_material_pct"
]

print("\nAkiya Redevelopment Prioritization Results\n")
print(df[output_columns].round(2))

# ====
# 8. Save results to CSV
# ====

df[output_columns].to_csv("akiya_property_scoring_results.csv", index=False)
print("\nResults saved to 'akiya_property_scoring_results.csv'")
