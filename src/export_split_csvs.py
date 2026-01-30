import pandas as pd

# 1) Read labeled CSV
df = pd.read_csv("flights_split.csv")

# 2) Split into train/val/test
train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
test_df  = df[df["split"] == "test"]

# 3) Save to disk
train_df.to_csv("data/flights_train.csv", index=False)
val_df.to_csv("data/flights_val.csv", index=False)
test_df.to_csv("data/flights_test.csv", index=False)

# 4) Quick sanity counts
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))
