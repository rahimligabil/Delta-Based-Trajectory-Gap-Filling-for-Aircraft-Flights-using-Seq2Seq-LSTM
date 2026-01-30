import glob
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Parameters
PAST_LEN = 6
FUTURE_LEN = 6
GAP_LEN = 8

FEATURES = [
    "lat",
    "lon",
    "geoaltitude",
    "velocity",
    "heading_sin",
    "heading_cos",
    "vertrate",
    "onground",
]

OUTPUT_X = os.path.join("data", "X_gap_v3.npy")
OUTPUT_Y = os.path.join("data", "Y_gap_v3.npy")
OUTPUT_SCALER = os.path.join("data", "scaler_v3.pkl")

# Read and basic clean
def read_flight_csv(path):
    df = pd.read_csv(path)
    df = df[
        [
            "time",
            "lat",
            "lon",
            "geoaltitude",
            "velocity",
            "heading",
            "vertrate",
            "onground",
        ]
    ]
    df["onground"] = df["onground"].astype(float)
    df = df.sort_values("time")
    df = df.dropna()
    return df

# Resample to fixed dt and add heading sin/cos
def resample_flight(df, dt=5):
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time_dt")
    df = df.resample(f"{dt}S").mean()
    df["onground"] = df["onground"].ffill()
    df = df.interpolate()
    df = df.dropna()

    heading_rad = np.deg2rad(df["heading"])
    df["heading_sin"] = np.sin(heading_rad)
    df["heading_cos"] = np.cos(heading_rad)
    df = df.drop(columns=["heading"])

    df = df[FEATURES]
    return df

# 1) Fit scaler on training flights only
scaler = StandardScaler()
train_files = glob.glob("state_vectors/train/*.csv")

print("Train flights:", len(train_files))

for f in train_files:
    df = resample_flight(read_flight_csv(f))
    if len(df) < 50:
        continue
    scaler.partial_fit(df[FEATURES].values)

# 2) Generate windows and gap targets
X_all = []
Y_all = []

for f in train_files:
    df = resample_flight(read_flight_csv(f))
    if len(df) < 50:
        continue

    data = scaler.transform(df[FEATURES].values)

    total_len = PAST_LEN + GAP_LEN + FUTURE_LEN

    for i in range(0, len(data) - total_len, 8):
        past   = data[i : i + PAST_LEN]
        gap    = data[i + PAST_LEN : i + PAST_LEN + GAP_LEN]
        future = data[i + PAST_LEN + GAP_LEN : i + total_len]

        x = np.vstack([past, future])     # (12,8)

        # delta target in scaled space
        abs_gap = gap[:, 0:3]                               # (8,3)
        prev0   = past[-1, 0:3]                             # last before gap
        abs_all = np.vstack([prev0, abs_gap])               # (9,3)
        y_delta = np.diff(abs_all, axis=0)                  # (8,3) delta per step
        y = y_delta

        X_all.append(x)
        Y_all.append(y)

X_all = np.array(X_all)
Y_all = np.array(Y_all)

print("X shape:", X_all.shape)
print("Y shape:", Y_all.shape)

os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)

np.save(OUTPUT_X, X_all)
np.save(OUTPUT_Y, Y_all)
joblib.dump(scaler, OUTPUT_SCALER)

print(f"Saved {OUTPUT_X} and {OUTPUT_Y}")
print(f"Saved scaler to {OUTPUT_SCALER}")
