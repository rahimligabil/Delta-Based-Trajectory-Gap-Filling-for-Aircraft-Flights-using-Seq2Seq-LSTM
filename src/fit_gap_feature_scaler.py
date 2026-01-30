import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

TRAIN_DIR = "state_vectors/train"
SCALER_OUT = "data/scaler_v2.pkl"

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
DT_SECONDS = 5

def to01(x):
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    s = str(x).strip().lower()
    if s in ["true", "1", "t", "yes", "y"]:
        return 1.0
    if s in ["false", "0", "f", "no", "n"]:
        return 0.0
    return np.nan

def resample_flight(df, dt_seconds=5):
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time_dt")

    # numeric columns: mean per window
    df = df.resample(f"{dt_seconds}S").mean(numeric_only=True)

    # onground is boolean; forward-fill separately if present
    if "onground" in df.columns:
        pass

    # interpolate over time gaps
    df = df.interpolate(method="time")
    df = df.dropna()

    heading_rad = np.deg2rad(df["heading"])
    df["heading_sin"] = np.sin(heading_rad)
    df["heading_cos"] = np.cos(heading_rad)
    df = df.drop(columns=["heading"])

    df = df[FEATURES]
    return df

scaler = StandardScaler()

files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".csv")]
print("train flights:", len(files))

for f in files:
    path = os.path.join(TRAIN_DIR, f)
    raw = pd.read_csv(path)

    raw["onground"] = raw["onground"].apply(to01)
    raw = raw.dropna(subset=["time"])

    # resample + interpolate (same as training pipeline)
    df = raw.copy()
    df["time_dt"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time_dt")

    num = df.resample(f"{DT_SECONDS}S").mean(numeric_only=True)
    ong = df["onground"].resample(f"{DT_SECONDS}S").ffill()

    num["onground"] = ong
    num = num.interpolate(method="time").dropna()

    heading_rad = np.deg2rad(num["heading"])
    num["heading_sin"] = np.sin(heading_rad)
    num["heading_cos"] = np.cos(heading_rad)
    num = num.drop(columns=["heading"])

    feat = num[FEATURES].dropna()

    if len(feat) >= 20:
        scaler.partial_fit(feat.values)

joblib.dump(scaler, SCALER_OUT)
print(" saved:", SCALER_OUT)
print("mean:", scaler.mean_)
print("std :", scaler.scale_)
