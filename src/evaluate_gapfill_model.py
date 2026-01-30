import sys, os, argparse
sys.path.append(os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
import joblib
import folium
from math import radians, sin, cos, sqrt, atan2
from gapfill_lstm_model import TwoEncoderSeq2SeqGapFill

# -------------------------
# CONFIG
# -------------------------
TEST_DIR = "state_vectors/test"
MODEL_PATH = "gapfill_model.pt"
SCALER_PATH = "data/scaler_v2.pkl"

PAST_LEN = 6
GAP_LEN = 8
FUTURE_LEN = 6
DT = 5

FEATURES = ["lat","lon","geoaltitude","velocity","heading_sin","heading_cos","vertrate","onground"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# HELPERS
# -------------------------
def resample(df):
    df["time_dt"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time_dt")
    df = df.resample(f"{DT}S").mean()
    df["onground"] = df["onground"].ffill()
    df = df.interpolate()
    df = df.dropna()

    heading_rad = np.deg2rad(df["heading"])
    df["heading_sin"] = np.sin(heading_rad)
    df["heading_cos"] = np.cos(heading_rad)
    df = df.drop(columns=["heading"])

    return df[FEATURES]

def inverse_latlonalt(x3, scaler):
    tmp = np.zeros((x3.shape[0], len(FEATURES)))
    tmp[:,0:3] = x3
    return scaler.inverse_transform(tmp)[:,0:3]

def haversine(p, q):
    """Great-circle distance between two lat/lon points in km."""
    lat1, lon1 = p
    lat2, lon2 = q
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))

def evaluate_one(fpath, save_map=True, map_name="gapfill_debug_map.html"):
    """Run one flight file; return error DataFrame and mean error."""
    df = pd.read_csv(fpath)
    df = resample(df)

    data = df[FEATURES].values
    data_scaled = scaler.transform(data)

    total_len = PAST_LEN + GAP_LEN + FUTURE_LEN
    if len(data_scaled) < total_len:
        raise ValueError(f"Flight too short: {len(data_scaled)} < {total_len}")

    # pick a centered window inside the flight
    start = len(data)//2 - total_len//2
    window = data_scaled[start:start+total_len]

    past   = window[:PAST_LEN]
    gap_gt = window[PAST_LEN:PAST_LEN+GAP_LEN]
    future = window[PAST_LEN+GAP_LEN:]

    X = np.vstack([past, future])
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(X_t, y_true=None, teacher_forcing=0.0).squeeze(0).cpu().numpy()

    # --- reconstruct absolute positions from deltas (all in scaled space) ---
    start_abs = past[-1, 0:3]  # scaled last before gap

    # ground truth delta in scaled space
    abs_all_true = np.vstack([start_abs, gap_gt[:,0:3]])        # (9,3) abs scaled
    true_delta   = np.diff(abs_all_true, axis=0)                 # (8,3)

    pred_delta   = pred                                          # model outputs delta scaled

    true_abs_scaled = start_abs + np.cumsum(true_delta, axis=0)  # (8,3)
    pred_abs_scaled = start_abs + np.cumsum(pred_delta, axis=0)  # (8,3)

    pred_llh = inverse_latlonalt(pred_abs_scaled, scaler)
    true_llh = inverse_latlonalt(true_abs_scaled, scaler)

    rows = []
    for i in range(len(pred_llh)):
        err_km = haversine(true_llh[i, :2], pred_llh[i, :2])
        rows.append({
            "step": i,
            "true_lat":  true_llh[i,0],
            "true_lon":  true_llh[i,1],
            "true_alt":  true_llh[i,2],
            "pred_lat":  pred_llh[i,0],
            "pred_lon":  pred_llh[i,1],
            "pred_alt":  pred_llh[i,2],
            "err_km":    err_km
        })

    df_err = pd.DataFrame(rows)

    if save_map:
        center = [true_llh[:,0].mean(), true_llh[:,1].mean()]
        m = folium.Map(location=center, zoom_start=7)

        # Full flight track (magenta) for sanity check
        full_track_coords = [[row[0], row[1]] for row in inverse_latlonalt(data[:,0:3], scaler)]
        folium.PolyLine(full_track_coords, color="#b300ff", weight=4, opacity=0.8, tooltip="Full track (all points)").add_to(m)

        # Past (blue) + Future (gray)
        pf = inverse_latlonalt(window[:,0:3], scaler)
        past_coords   = [[x[0], x[1]] for x in pf[:PAST_LEN]]
        future_coords = [[x[0], x[1]] for x in pf[PAST_LEN+GAP_LEN:]]
        folium.PolyLine(past_coords, color="blue", weight=3, tooltip="Past (6)").add_to(m)
        folium.PolyLine(future_coords, color="gray", weight=3, tooltip="Future (6)").add_to(m)

        folium.CircleMarker(past_coords[-1], radius=4, color="blue", fill=True, fill_opacity=1, tooltip="Past end").add_to(m)
        folium.CircleMarker(future_coords[0], radius=4, color="gray", fill=True, fill_opacity=1, tooltip="Future start").add_to(m)

        # True gap (green thick)
        true_coords = [[x[0], x[1]] for x in true_llh]
        folium.PolyLine(true_coords, color="green", weight=6, tooltip="Real gap").add_to(m)

        # Predicted gap (red)
        pred_coords = [[x[0], x[1]] for x in pred_llh]
        folium.PolyLine(pred_coords, color="red", weight=4, tooltip="Predicted gap").add_to(m)

        all_coords = past_coords + true_coords + future_coords + pred_coords + full_track_coords
        m.fit_bounds(all_coords, padding=(20,20))
        m.save(map_name)

    return df_err, df_err["err_km"].mean()


# -------------------------
# LOAD MODEL
# -------------------------
scaler = joblib.load(SCALER_PATH)

model = TwoEncoderSeq2SeqGapFill(
    input_size=8, output_size=3,
    hidden_size=128, num_layers=2,
    dropout=0.1,
    past_len=PAST_LEN,
    future_len=FUTURE_LEN,
    gap_len=GAP_LEN
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all flights in TEST_DIR")
    parser.add_argument("--no-map", action="store_true", help="Skip saving maps")
    parser.add_argument("--file", type=str, help="Single flight csv to run (overrides default first file)")
    args = parser.parse_args()

    if args.all:
        means = []
        files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".csv")])
        for f in files:
            path = os.path.join(TEST_DIR, f)
            try:
                _, mean_err = evaluate_one(path, save_map=not args.no_map, map_name=f"gapfill_{os.path.splitext(f)[0]}.html")
                means.append(mean_err)
                print(f"{f}: mean gap err {mean_err:.3f} km")
            except Exception as e:
                print(f"{f}: SKIPPED ({e})")
        if means:
            print(f"\nOverall mean across {len(means)} flights: {np.mean(means):.3f} km")
    else:
        if args.file:
            fpath = args.file if os.path.isabs(args.file) else os.path.join(TEST_DIR, args.file)
        else:
            fname = os.listdir(TEST_DIR)[0]
            fpath = os.path.join(TEST_DIR, fname)
        df_err, mean_err = evaluate_one(fpath, save_map=not args.no_map, map_name="gapfill_debug_map.html")
        print("\nGap comparison (true vs predicted) and Haversine error (km):")
        print(df_err.to_string(index=False, formatters={"err_km": lambda x: f"{x:6.3f}"}))
        print(f"\nMean gap error: {mean_err:.3f} km")
        if not args.no_map:
            print("Saved: gapfill_debug_map.html")
