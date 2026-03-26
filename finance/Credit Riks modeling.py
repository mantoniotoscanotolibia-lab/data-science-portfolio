import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1) EXAMPLE DATA
# Replace this with your real dataset if needed
# The dataset must have:
#   - fico_score
#   - default (0 or 1)
# ---------------------------------------------------------

data = {
    "fico_score": [520, 540, 560, 580, 600, 620, 640, 660, 680, 700,
                   720, 740, 760, 780, 800, 820, 840, 610, 630, 650,
                   670, 690, 710, 730, 750, 770, 790, 810],
    "default":    [1,   1,   1,   1,   1,   0,   1,   0,   0,   0,
                   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,
                   0,   0,   0,   0,   0,   0,   0,   0]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2) PREPARE DATA
# Group by FICO score
# For each unique score:
#   n_i = number of borrowers
#   k_i = number of defaults
# ---------------------------------------------------------

grouped = df.groupby("fico_score").agg(
    n=("default", "count"),
    k=("default", "sum")
).reset_index()

fico_values = grouped["fico_score"].values
n_vals = grouped["n"].values
k_vals = grouped["k"].values

m = len(grouped)

# Prefix sums for fast range calculations
cum_n = np.zeros(m + 1)
cum_k = np.zeros(m + 1)

for i in range(1, m + 1):
    cum_n[i] = cum_n[i - 1] + n_vals[i - 1]
    cum_k[i] = cum_k[i - 1] + k_vals[i - 1]

# ---------------------------------------------------------
# 3) LOG-LIKELIHOOD FUNCTION FOR ONE BUCKET
# bucket = scores from i to j inclusive
# ---------------------------------------------------------

def bucket_loglik(i, j):
    total_n = cum_n[j + 1] - cum_n[i]
    total_k = cum_k[j + 1] - cum_k[i]

    if total_n == 0:
        return -np.inf

    p = total_k / total_n

    # avoid log(0)
    if p == 0:
        return (total_n - total_k) * np.log(1e-12 + 1.0)
    if p == 1:
        return total_k * np.log(1e-12 + 1.0)

    return total_k * np.log(p) + (total_n - total_k) * np.log(1 - p)

# ---------------------------------------------------------
# 4) DYNAMIC PROGRAMMING TO FIND BEST BUCKETS
# dp[b][j] = best log-likelihood using b buckets up to j
# ---------------------------------------------------------

def find_optimal_buckets(num_buckets):
    dp = np.full((num_buckets + 1, m), -np.inf)
    split = np.full((num_buckets + 1, m), -1, dtype=int)

    # Base case: 1 bucket from 0 to j
    for j in range(m):
        dp[1][j] = bucket_loglik(0, j)

    # Fill DP table
    for b in range(2, num_buckets + 1):
        for j in range(b - 1, m):
            for i in range(b - 2, j):
                candidate = dp[b - 1][i] + bucket_loglik(i + 1, j)
                if candidate > dp[b][j]:
                    dp[b][j] = candidate
                    split[b][j] = i

    # Reconstruct bucket ranges
    buckets = []
    j = m - 1
    b = num_buckets

    while b > 1:
        i = split[b][j]
        buckets.append((i + 1, j))
        j = i
        b -= 1

    buckets.append((0, j))
    buckets.reverse()

    return buckets, dp[num_buckets][m - 1]

# ---------------------------------------------------------
# 5) BUILD RATING MAP
# lower rating = better credit score
# So highest FICO bucket gets rating 1
# ---------------------------------------------------------

def build_rating_map(num_buckets):
    buckets, best_ll = find_optimal_buckets(num_buckets)

    rows = []

    # Buckets come from low FICO to high FICO
    # But ratings should be reversed:
    # highest fico -> rating 1
    for idx, (start, end) in enumerate(buckets):
        fico_min = fico_values[start]
        fico_max = fico_values[end]
        total_n = int(cum_n[end + 1] - cum_n[start])
        total_k = int(cum_k[end + 1] - cum_k[start])
        pd_bucket = total_k / total_n if total_n > 0 else np.nan

        rows.append({
            "bucket_number": idx + 1,
            "fico_min": fico_min,
            "fico_max": fico_max,
            "num_records": total_n,
            "num_defaults": total_k,
            "pd": round(pd_bucket, 4)
        })

    rating_map = pd.DataFrame(rows)

    # Reverse ratings so lower rating means better score
    rating_map = rating_map.sort_values("fico_min", ascending=False).reset_index(drop=True)
    rating_map["rating"] = range(1, len(rating_map) + 1)

    # Put in nicer order
    rating_map = rating_map[[
        "rating", "fico_min", "fico_max",
        "num_records", "num_defaults", "pd"
    ]].sort_values("rating").reset_index(drop=True)

    return rating_map, best_ll

# ---------------------------------------------------------
# 6) FUNCTION TO ASSIGN RATING TO A NEW FICO SCORE
# ---------------------------------------------------------

def assign_rating(fico_score, rating_map):
    for _, row in rating_map.iterrows():
        if row["fico_min"] <= fico_score <= row["fico_max"]:
            return int(row["rating"])

    # handle values outside observed range
    if fico_score < rating_map["fico_min"].min():
        return int(rating_map.loc[rating_map["fico_min"].idxmin(), "rating"])

    if fico_score > rating_map["fico_max"].max():
        return int(rating_map.loc[rating_map["fico_max"].idxmax(), "rating"])

    return None

# ---------------------------------------------------------
# 7) RUN EXAMPLE
# ---------------------------------------------------------

num_buckets = 5
rating_map, best_ll = build_rating_map(num_buckets)

print("Optimal Rating Map")
print(rating_map)
print("\nBest Log-Likelihood:", round(best_ll, 4))

# Example new borrowers
test_scores = [545, 615, 685, 755, 825]

print("\nSample FICO to Rating Mapping:")
for score in test_scores:
    rating = assign_rating(score, rating_map)
    print(f"FICO {score} -> Rating {rating}")