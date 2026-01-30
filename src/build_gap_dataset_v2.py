import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1) Read and basic clean of a flight CSV
def read_flight_csv(path):
    df = pd.read_csv(path)
    df = df[["time", "lat", "lon", "geoaltitude", "velocity", "heading", "vertrate", "onground"]]
    df = df.sort_values("time")
    df = df.dropna(subset=["lat", "lon", "geoaltitude", "velocity", "heading", "vertrate"])
    # ensure boolean onground
    df["onground"] = df["onground"].astype(str).str.lower().isin(["true", "1"])
    return df

# 2) Resample to fixed dt and time-interpolate
def resample_flight(df, dt_seconds=5):
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time_dt")
    df = df.resample(f"{dt_seconds}S").mean()
    df = df.interpolate(method="time")
    df = df.dropna()
    return df

# 3) Convert heading to sin/cos features
def add_heading_features(df):
    df = df.copy()
    heading_rad = np.deg2rad(df["heading"].values)
    df["heading_sin"] = np.sin(heading_rad)
    df["heading_cos"] = np.cos(heading_rad)
    df = df.drop(columns=["heading"])
    df = df[["lat", "lon", "geoaltitude", "velocity", "heading_sin", "heading_cos", "vertrate", "onground"]]
    return df

# 4) Build gap windows (past + gap + future)
def make_gap_windows(arr, window_size=20, gap=8):
    past = (window_size - gap) // 2   # 6
    future = window_size - gap - past # 6

    X_all = []
    Y_all = []

    for i in range(len(arr) - window_size + 1):
        w = arr[i:i + window_size]   # (20, F)
        past_part   = w[:past]
        gap_part    = w[past:past + gap]
        future_part = w[past + gap:]

        # input = past + future
        X = np.concatenate([past_part, future_part], axis=0)   # (12, F)
        # target = gap lat/lon/alt only
        Y = gap_part[:, 0:3]   # (8, 3)

        X_all.append(X)
        Y_all.append(Y)

    return np.array(X_all), np.array(Y_all)

# 5) Locate all training flights
flight_files = glob.glob("state_vectors/train/*.csv")

# 6) Fit scaler only on training flights
scaler = StandardScaler()

for path in flight_files:
    df = read_flight_csv(path)
    df = df[df["onground"] == False]     # airborne only
    df = resample_flight(df)
    df = add_heading_features(df)

    if len(df) < 30:
        continue

    features = df[["lat", "lon", "geoaltitude", "velocity", "heading_sin", "heading_cos", "vertrate", "onground"]]
    scaler.partial_fit(features.values)

# 7) Build dataset windows
X_all = []
Y_all = []

for path in flight_files:
    df = read_flight_csv(path)
    df = df[df["onground"] == False]
    df = resample_flight(df)
    df = add_heading_features(df)

    if len(df) < 30:
        continue

    features = df[["lat", "lon", "geoaltitude", "velocity", "heading_sin", "heading_cos", "vertrate", "onground"]]
    arr = scaler.transform(features.values)

    X, Y = make_gap_windows(arr, window_size=20, gap=8)

    if len(X) == 0:
        continue

    X_all.append(X)
    Y_all.append(Y)

# 8) Concatenate and save
X_all = np.concatenate(X_all, axis=0)
Y_all = np.concatenate(Y_all, axis=0)

np.save("X_gap_v2.npy", X_all)
np.save("Y_gap_v2.npy", Y_all)

print("DONE")
print("X_gap_v2 shape:", X_all.shape)   # (N, 12, F)
print("Y_gap_v2 shape:", Y_all.shape)   # (N, 8, 3)
