# FICO Rating Model

# 1) Loads borrower data
# 2) Splits FICO scores into buckets
# 3) Calculates default probability in each bucket
# 4) Creates a rating map
# 5) Assigns ratings to new borrowers

import pandas as pd

# -------------------------------------------------
# 1) LOAD DATA
# -------------------------------------------------

data = {
    "fico_score": [520,540,560,580,600,620,640,660,680,700,
                   720,740,760,780,800,820,840,610,630,650,
                   670,690,710,730,750,770,790,810],
    
    "default":    [1,1,1,1,1,0,1,0,0,0,
                   0,0,0,0,0,0,0,1,0,0,
                   0,0,0,0,0,0,0,0]
}

df = pd.DataFrame(data)

# -------------------------------------------------
# 2) CREATE BUCKETS
# -------------------------------------------------

num_buckets = 5

# Divide FICO scores into equal-sized buckets
df["bucket"] = pd.qcut(df["fico_score"], num_buckets, labels=False)

# -------------------------------------------------
# 3) CALCULATE DEFAULT PROBABILITY PER BUCKET
# -------------------------------------------------

bucket_stats = df.groupby("bucket").agg(
    min_fico=("fico_score","min"),
    max_fico=("fico_score","max"),
    num_records=("default","count"),
    num_defaults=("default","sum")
).reset_index()

bucket_stats["pd"] = bucket_stats["num_defaults"] / bucket_stats["num_records"]

# -------------------------------------------------
# 4) CREATE RATING MAP
# lower rating = better credit score
# -------------------------------------------------

bucket_stats = bucket_stats.sort_values("min_fico", ascending=False).reset_index(drop=True)

bucket_stats["rating"] = range(1, len(bucket_stats)+1)

rating_map = bucket_stats[[
    "rating",
    "min_fico",
    "max_fico",
    "num_records",
    "num_defaults",
    "pd"
]]

print("Rating Map")
print(rating_map)
print()

# -------------------------------------------------
# 5) FUNCTION TO ASSIGN RATING
# -------------------------------------------------

def assign_rating(fico_score):

    for _, row in rating_map.iterrows():

        if row["min_fico"] <= fico_score <= row["max_fico"]:
            return int(row["rating"])

    return None


# -------------------------------------------------
# 6) TEST EXAMPLES
# -------------------------------------------------

test_scores = [550, 620, 690, 740, 820]

print("Sample Ratings")
for score in test_scores:

    rating = assign_rating(score)

    print("FICO:", score, "-> Rating:", rating)
