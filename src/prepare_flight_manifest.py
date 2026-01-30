import pandas as pd
import csv

INPUT_PATH  = "flights_LTFM_LFPG_1years.csv"       # source dump provided by user
OUTPUT_PATH = "flights_clean.csv"

# 1) Read the pipe-delimited dump ("col | col | col")
# - sep="|" because columns are pipe-separated
# - engine="python" is more tolerant for messy rows
df = pd.read_csv(
    INPUT_PATH,
    sep="|",
    engine="python",
    header=None,
    skiprows=2,          # skip the header/separator rows from the export
)

# 2) Assign column names
df.columns = ["day", "icao24", "callsign", "firstseen", "lastseen"]

# 3) Trim whitespace that follows each pipe
for c in df.columns:
    df[c] = df[c].astype(str).str.strip()

# 4) Cast numeric fields; invalid rows become NaN
df["day"]       = pd.to_numeric(df["day"], errors="coerce")
df["firstseen"] = pd.to_numeric(df["firstseen"], errors="coerce")
df["lastseen"]  = pd.to_numeric(df["lastseen"], errors="coerce")

# 5) Drop rows missing critical fields
df = df.dropna(subset=["day", "icao24", "firstseen", "lastseen"])

# 6) Normalize empty callsigns
df["callsign"] = df["callsign"].replace({"NULL": None, "": None})

# 7) Compute flight duration
df["duration_sec"] = df["lastseen"] - df["firstseen"]

# 8) Keep only plausible durations (10?30000 seconds)
df = df[(df["duration_sec"] >= 600) & (df["duration_sec"] <= 30000)]

# 9) Build a unique flight_id (icao24 can appear multiple times per day)
df["flight_id"] = df["icao24"].astype(str) + "_" + df["firstseen"].astype(int).astype(str)

# 10) Sort for readability
df = df.sort_values(["day", "firstseen", "icao24"]).reset_index(drop=True)

# 11) Save cleaned manifest
df.to_csv(OUTPUT_PATH, index=False)

print("OK!")
print("Rows:", len(df))
print("Duration(min,max):", df["duration_sec"].min(), df["duration_sec"].max())
print("Unique flights:", df["flight_id"].nunique())
