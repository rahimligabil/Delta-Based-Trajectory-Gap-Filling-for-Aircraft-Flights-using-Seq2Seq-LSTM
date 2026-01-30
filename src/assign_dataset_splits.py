import pandas as pd
import numpy as np

INPUT  = "flights_clean.csv"
OUTPUT = "flights_split.csv"

# Load cleaned manifest
df = pd.read_csv(INPUT)

# Shuffle rows for a random split (reproducible seed)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
train_end = int(n * 0.80)
val_end   = int(n * 0.90)

# Default all rows to test, then overwrite earlier ranges
df["split"] = "test"
df.loc[:train_end-1, "split"] = "train"
df.loc[train_end:val_end-1, "split"] = "val"

# Save with split labels
df.to_csv(OUTPUT, index=False)

print("OK!")
print(df["split"].value_counts())
